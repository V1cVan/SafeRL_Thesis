import tensorflow as tf
from tensorflow import keras
import datetime
from tensorflow.keras import layers, Sequential
import numpy as np
from HelperClasses import EpisodeBuffer, DataLogger
from DqnBuffers import TrainingBuffer, PerTrainingBuffer

class DqnAgent(keras.models.Model):
    """
    double deep q network trainer
    """

    def __init__(self, network, target_network, training_param, tb_logger):
        super(DqnAgent, self).__init__()
        tf.random.set_seed(training_param["seed"])
        np.random.seed(training_param["seed"])
        self.tb_logger = tb_logger

        self.latest_experience = None
        self.latest_reward = 0
        self.is_action_taken = False
        self.Q_target_net = target_network
        self.Q_actual_net = network
        self.reward_weights = training_param["reward_weights"]
        self.training = True
        self.training_param = training_param
        self.stop_flags = None
        self.eps_final = 0.1
        self.decay = training_param["decay_rate"]
        self.gamma = training_param["gamma"]
        self.epsilon = training_param["epsilon_min"]
        self.prev_epsilon = self.epsilon
        self.epsilon_decay_count = 1
        self.evaluation = False
        self.episode = 1

        # Set parameter which changes behaviour of the training buffer if CNN or deepset models are used
        if training_param["use_deepset"] or training_param["use_CNN"]:
            self.use_deepset_or_cnn = True
        else:
            self.use_deepset_or_cnn = False

        # Set parameter which changes sampling behaviour of training buffer if a certain CNN or LSTM are used
        if training_param["use_LSTM"] or \
                (training_param["use_CNN"] == True and self.Q_actual_net.model_param["cnn_param"]["config"]==3):
            use_frame_stacking = True
        else:
            use_frame_stacking = False

        if training_param["use_per"]:
            self.buffer = PerTrainingBuffer(buffer_size=training_param["buffer_size"],
                                            batch_size=training_param["batch_size"],
                                            alpha=training_param["alpha"],
                                            beta=training_param["beta"],
                                            beta_increment=training_param["beta_increment"],
                                            use_deepset_or_cnn=self.use_deepset_or_cnn,
                                            stack_frames=use_frame_stacking)
        else:
            self.buffer = TrainingBuffer(buffer_size=training_param["buffer_size"],
                                         batch_size=training_param["batch_size"],
                                         use_deepset_or_cnn=self.use_deepset_or_cnn,
                                         stack_frames=use_frame_stacking)

    def set_neg_collision_reward(self, timestep, punishment):
        """ Sets a negative reward if a collision occurs. """
        self.buffer.alter_reward_at_timestep(timestep, punishment)

    def add_experience(self, done):
        states, actions, rewards, next_states = self.latest_experience
        experience = (states, actions, rewards, next_states, done)

        if self.training_param["use_per"]:
            # TODO add deepset implementation for PER
            # Calculate the TD-error for the Prioritised Replay Buffer
            if self.use_deepset_or_cnn:
                dyn_states = tf.convert_to_tensor(states[0], dtype=np.float32)
                stat_states = tf.convert_to_tensor(states[1], dtype=np.float32)
                # states = (dyn_states, stat_states)

                actions = tf.squeeze(tf.convert_to_tensor(actions, dtype=np.float32))
                rewards = tf.squeeze(tf.convert_to_tensor(rewards, dtype=np.float32))

                dyn_next_states = tf.convert_to_tensor(next_states[0], dtype=np.float32)
                stat_next_states = tf.convert_to_tensor(next_states[1], dtype=np.float32)
                # next_states = (dyn_next_states, stat_next_states)

                done = tf.cast(done, dtype=tf.float32)
                td_error = self.compute_td_error(states=states,
                                                 rewards=rewards,
                                                 next_states=next_states,
                                                 done=done)
                self.buffer.add_experience(td_error, (dyn_states, stat_states,
                                                      actions, rewards,
                                                      dyn_next_states, stat_next_states,
                                                      done))
            else:
                states = tf.convert_to_tensor(states, dtype=np.float32)
                actions = tf.squeeze(tf.convert_to_tensor(actions, dtype=np.float32))
                rewards = tf.squeeze(tf.convert_to_tensor(rewards, dtype=np.float32))
                next_states = tf.convert_to_tensor(next_states, dtype=np.float32)
                done = tf.cast(done, dtype=tf.float32)
                td_error = self.compute_td_error(states=states,
                                                 rewards=rewards,
                                                 next_states=next_states,
                                                 done=done)
                self.buffer.add_experience(td_error, (states, actions, rewards, next_states, done))
        else:
            self.buffer.add_experience(experience)

    @tf.function
    def compute_td_error(self,
                         states: tf.Tensor,
                         rewards: tf.Tensor,
                         next_states: tf.Tensor,
                         done: tf.Tensor):
        ones = tf.ones(tf.shape(done), dtype=tf.dtypes.float32)
        target_Q = self.Q_target_net(next_states)
        target_output = rewards + (ones - done) * (self.gamma * tf.reduce_max(target_Q, axis=1))
        predicted_Q = self.Q_actual_net(states)
        predicted_output = tf.reduce_max(predicted_Q, axis=1)
        return target_output - predicted_output

    def calc_epsilon(self):
        if self.evaluation:
            self.epsilon = 0
        elif not self.buffer.is_buffer_min_size() and not self.evaluation:
            self.epsilon = 1
        else:
            if self.epsilon_decay_count > 1 and self.epsilon > self.eps_final:
                self.epsilon = self.epsilon * self.decay
        return self.epsilon

    def get_action_choice(self, Q):
        """ Randomly choose from the available actions."""
        epsilon = self.calc_epsilon()

        if np.random.rand() < epsilon and not self.evaluation:
            self.latest_action = np.random.randint(0, 5)
        else:
            # Otherwise, query the DQN for an action
            self.latest_action = np.argmax(Q, axis=1)[0]
        return self.latest_action

    def update_target_net(self):
        self.Q_target_net.set_weights(self.Q_actual_net.get_weights())

    def train_step(self):
        """ Performs a training step. """
        if self.training:
            n_actions = self.Q_actual_net.model_param["n_actions"]
            # Gather and convert data from the buffer (data from simulation):
            # Sample mini-batch from memory
            if self.training_param["use_per"]:
                # TODO add deepset implementation
                states, actions, rewards, next_states, done, idxs, is_weight = self.buffer.get_training_samples()
                one_hot_actions = tf.keras.utils.to_categorical(actions, num_classes=n_actions)
                mean_batch_reward, loss, td_error, grads, clipped_grads = self.run_tape(
                    states=states,
                    actions=one_hot_actions,
                    rewards=rewards,
                    next_states=next_states,
                    done=done,
                    is_weight=is_weight)
            else:
                states, actions, rewards, next_states, done = self.buffer.get_training_samples()
                one_hot_actions = tf.keras.utils.to_categorical(actions, num_classes=n_actions)
                mean_batch_reward, loss, td_error, grads, clipped_grads = self.run_tape(
                    states=states,
                    actions=one_hot_actions,
                    rewards=rewards,
                    next_states=next_states,
                    done=done)

            if self.training_param["use_per"]:
                self.buffer.update(idxs, td_error)
            # self.buffer.set_training_variables(
            #     episode_num=self.episode,
            #     episode_reward=episode_reward,
            #     losses=loss,
            #     advantage=tf.squeeze(tf.convert_to_tensor(advantage)),
            #     returns=returns,
            #     gradients=np.squeeze(grads),
            #     model_weights=self.actor_critic_net.weights)

            return mean_batch_reward, loss, td_error, grads, clipped_grads

    @tf.function
    def run_tape(self,
                 states: tf.Tensor,
                 actions: tf.Tensor,
                 rewards: tf.Tensor,
                 next_states: tf.Tensor,
                 done: tf.Tensor,
                 is_weight: tf.Tensor = None):
        """ Performs the training calculations in a tf.function. """
        ones = tf.ones(tf.shape(done), dtype=tf.dtypes.float32)

        Q_output = self.Q_target_net(next_states)
        Q_target = rewards + (ones - done) * (self.gamma * tf.reduce_max(Q_output, axis=1))

        if self.training_param["standardise_returns"]:
            eps = np.finfo(np.float32).eps.item()
            Q_target = (Q_target - tf.math.reduce_mean(Q_target)) / (tf.math.reduce_std(Q_target) + eps)

        with tf.GradientTape() as tape:
            Q_output = self.Q_actual_net(states)
            Q_predicted = tf.reduce_sum(Q_output * actions, axis=1)

            td_error = Q_target - Q_predicted

            # loss_value = self.training_param["loss_func"](Q_target, Q_predicted)
            # loss_value = tf.losses.MSE(y_true=target_output, y_pred=predicted_output)
            # loss_value = tf.reduce_mean(tf.square(Q_target - Q_predicted))
            if self.training_param["use_per"]:
                # TODO compute loss for each item in experience individually and then perform the huber loss calculation
                loss_value = tf.reduce_mean(tf.square( is_weight * (Q_target - Q_predicted)))
            else:
                loss_value = self.training_param["loss_func"](y_true=Q_target, y_pred=Q_predicted)

        grads = tape.gradient(loss_value, self.Q_actual_net.trainable_variables)


        # Clip gradients
        if self.training_param["clip_gradients"]:
            norm = self.training_param["clip_norm"]
            clipped_grads = [tf.clip_by_norm(g, norm)
                     for g in grads]

        if self.training_param["clip_gradients"]:
            grads= clipped_grads

        self.training_param["optimiser"].apply_gradients(zip(grads, self.Q_actual_net.trainable_variables))
        sum_reward = tf.math.reduce_sum(rewards)
        mean_batch_reward = sum_reward / self.buffer.batch_size

        if self.training_param["clip_gradients"]:
            return mean_batch_reward, loss_value, td_error, grads, clipped_grads
        else:
            return mean_batch_reward, loss_value, td_error, grads, grads


# class SpgAgentSingle(keras.models.Model):
#     """
#     Gradient ascent training algorithm.
#     https://spinningup.openai.com/en/latest/algorithms/vpg.html
#     """
#
#     def __init__(self, network, training_param):
#         super(SpgAgentSingle, self).__init__()
#         tf.random.set_seed(training_param["seed"])
#         np.random.seed(training_param["seed"])
#         self.actor_critic_net = network
#         self.reward_weights = training_param["reward_weights"]
#         # TODO implement data logging class for debugging training
#         self.training = True
#         self.training_param = training_param
#         self.episode = 1
#         self.buffer = EpisodeBuffer()
#         self.actions = []
#         self.states = []
#         self.rewards = []
#
#     def set_neg_collision_reward(self, timestep, punishment):
#         """ Sets a negative reward if a collision occurs. """
#         self.buffer.alter_reward_at_timestep(timestep, punishment)
#
#     def get_action_choice(self, action_probs):
#         """ Randomly choose from the available actions."""
#         # TODO make sure that you add is_evaluation to the choosing of actions
#         action_choice = np.random.choice(5, p=np.squeeze(action_probs))
#         return action_choice
#
#     @tf.function
#     def get_expected_returns(self,
#                              rewards: tf.Tensor) -> tf.Tensor:
#         """ Computes expected returns per timestep. """
#
#         # Initialise returns array outside of tf.function
#         n = tf.shape(rewards)[0]
#         returns = tf.TensorArray(dtype=tf.float32, size=n)
#
#         eps = np.finfo(np.float32).eps.item()
#         # Start from the end of rewards and accumulate reward sums into the returns array
#         rewards = tf.cast(rewards[::-1], dtype=tf.float32)
#         discounted_sum = tf.constant(0.0)
#         discounted_sum_shape = discounted_sum.shape
#         for i in tf.range(tf.shape(rewards)[0]):
#             discounted_sum = rewards[i] + self.training_param["gamma"] * discounted_sum
#             discounted_sum.set_shape(discounted_sum_shape)
#             returns = returns.write(i, discounted_sum)
#         returns = returns.stack()[::-1]
#
#         if self.training_param["standardise_rewards"]:
#             returns = ((returns - tf.math.reduce_mean(returns)) / (tf.math.reduce_std(returns) + eps))
#
#         return returns
#
#     @tf.function
#     def compute_loss(
#             self,
#             action_probs: tf.Tensor,
#             critic_values: tf.Tensor,
#             returns: tf.Tensor):
#         """ Computes the combined actor-critic loss."""
#
#         # Advantage: How much better an action is given a state over a random action selected by the policy
#         advantage = returns - critic_values
#
#         action_log_probs = tf.math.log(action_probs)
#
#         actor_loss = -tf.math.reduce_sum(action_log_probs * advantage)
#
#         # critic_loss = tf.losses.MSE(y_true=critic_values, y_pred=returns)
#         # critic_loss = tf.reduce_mean(tf.square(critic_values - returns))
#         critic_loss = self.training_param["loss_func"](critic_values, returns)
#
#         loss = critic_loss + actor_loss
#
#         return loss, advantage
#
#     def train_step(self):
#         """ Performs a training step. """
#         if self.training:
#             # Gather and convert data from the buffer (data from simulation):
#             timesteps, sim_states, rewards, sim_action_choices \
#                 = self.buffer.experience
#
#             # TODO make sure that n_actions are reflected in categorical choice!
#             sim_action_choices = tf.keras.utils.to_categorical(sim_action_choices)
#
#             episode_reward, loss, advantage, grads, returns = self.run_tape(
#                 sim_states=sim_states,
#                 action_choices=sim_action_choices,
#                 rewards=rewards)
#
#             self.buffer.set_training_variables(
#                 episode_num=self.episode,
#                 episode_reward=episode_reward,
#                 losses=loss,
#                 advantage=tf.squeeze(tf.convert_to_tensor(advantage)),
#                 returns=returns,
#                 gradients=np.squeeze(grads),
#                 model_weights=self.actor_critic_net.weights)
#
#             return episode_reward, loss
#
#     # @tf.function
#     def run_tape(self,
#                  sim_states: tf.Tensor,
#                  action_choices: tf.Tensor,
#                  rewards: tf.Tensor):
#         """ Performs the training calculations in a tf.function. """
#
#         with tf.GradientTape() as tape:
#             # Forward Pass - (Re)Calculation of actions that caused saved states
#             action_probs, critic_values = self.actor_critic_net(sim_states)
#             critic_values = tf.squeeze(critic_values)
#
#             # Choose actions based on what was previously (randomly) sampled during simulation
#             actions = tf.reduce_sum(action_choices * action_probs, axis=1)
#
#             # Calculate expected returns
#             returns = self.get_expected_returns(rewards=rewards)
#
#             # Calculating loss values to update our network
#             loss, advantage = self.compute_loss(actions, critic_values, returns)
#
#         for x in self.actor_critic_net.weights:
#             if tf.reduce_any(tf.math.is_nan(x)):
#                 print("NAN detected in network weight")
#
#         # Compute the gradients from the loss
#         grads = tape.gradient(loss, self.actor_critic_net.trainable_variables)
#
#         # Clip gradients
#         if self.training_param["clip_gradients"]:
#             norm = self.training_param["clip_norm"]
#             grads = [tf.clip_by_norm(g, norm)
#                      for g in grads]
#
#         # Apply the gradients to the model's parameters
#         self.training_param["optimiser"].apply_gradients(
#             zip(grads, self.actor_critic_net.trainable_variables))
#
#         episode_reward = tf.math.reduce_sum(rewards)
#
#         return episode_reward, loss, advantage, returns, grads


# class SpgAgentDouble(keras.models.Model):
#     """
#     Gradient ascent training algorithm.
#     https://spinningup.openai.com/en/latest/algorithms/vpg.html
#     """
#
#     def __init__(self, network, training_param):
#         super(SpgAgentDouble, self).__init__()
#         tf.random.set_seed(training_param["seed"])
#         np.random.seed(training_param["seed"])
#         self.actor_critic_net = network
#         self.reward_weights = training_param["reward_weights"]
#         # TODO implement data logging class for debugging training
#         self.training = True
#         self.training_param = training_param
#         self.episode = 1
#         self.buffer = EpisodeBuffer()
#         self.actions = []
#         self.states = []
#         self.rewards = []
#
#     def set_neg_collision_reward(self, timestep, punishment):
#         """ Sets a negative reward if a collision occurs. """
#         self.buffer.alter_reward_at_timestep(timestep, punishment)
#
#     def get_action_choice(self, action_probs):
#         """ Randomly choose from the available actions."""
#         # TODO make sure that you add is_evaluation to the choosing of actions
#         action_choice = np.random.choice(5, p=np.squeeze(action_probs))
#         return action_choice
#
#     @tf.function
#     def get_expected_returns(self,
#                              rewards: tf.Tensor) -> tf.Tensor:
#         """ Computes expected returns per timestep. """
#
#         # Initialise returns array outside of tf.function
#         n = tf.shape(rewards)[0]
#         returns = tf.TensorArray(dtype=tf.float32, size=n)
#
#         eps = np.finfo(np.float32).eps.item()
#         # Start from the end of rewards and accumulate reward sums into the returns array
#         rewards = tf.cast(rewards[::-1], dtype=tf.float32)
#         discounted_sum = tf.constant(0.0)
#         discounted_sum_shape = discounted_sum.shape
#         for i in tf.range(tf.shape(rewards)[0]):
#             discounted_sum = rewards[i] + self.training_param["gamma"] * discounted_sum
#             discounted_sum.set_shape(discounted_sum_shape)
#             returns = returns.write(i, discounted_sum)
#         returns = returns.stack()[::-1]
#
#         if self.training_param["standardise_rewards"]:
#             returns = ((returns - tf.math.reduce_mean(returns)) / (tf.math.reduce_std(returns) + eps))
#
#         return returns
#
#     # @tf.function
#     def compute_loss(
#             self,
#             action_probs: tf.Tensor,
#             critic_values: tf.Tensor,
#             returns: tf.Tensor):
#         """ Computes the combined actor-critic loss."""
#
#         # Advantage: How much better an action is given a state over a random action selected by the policy
#         advantage = returns - critic_values
#
#         action_log_probs = tf.math.log(action_probs)
#
#         actor_loss = -tf.math.reduce_sum(action_log_probs * advantage)
#
#         # critic_loss = tf.losses.MSE(y_true=critic_values, y_pred=returns)
#         # critic_loss = tf.reduce_mean(tf.square(critic_values - returns))
#         critic_loss = self.training_param["loss_func"](critic_values, returns)
#
#         loss = critic_loss + actor_loss
#
#         return loss, advantage
#
#     def train_step(self):
#         """ Performs a training step. """
#         if self.training:
#             # Gather and convert data from the buffer (data from simulation):
#             timesteps, sim_states, rewards, sim_action_choices \
#                 = self.buffer.experience
#
#             # TODO make sure that n_actions are reflected in categorical choice!
#             sim_action_choices = tf.keras.utils.to_categorical(sim_action_choices)
#
#             episode_reward, loss, advantage, grads, returns = self.run_tape(
#                 sim_states=sim_states,
#                 action_choices=sim_action_choices,
#                 rewards=rewards)
#
#             self.buffer.set_training_variables(
#                 episode_num=self.episode,
#                 episode_reward=episode_reward,
#                 losses=loss,
#                 advantage=tf.squeeze(tf.convert_to_tensor(advantage)),
#                 returns=returns,
#                 gradients=np.squeeze(grads),
#                 model_weights=self.actor_critic_net.weights)
#
#             return episode_reward, loss
#
#     # @tf.function
#     def run_tape(self,
#                  sim_states: tf.Tensor,
#                  action_choices: tf.Tensor,
#                  rewards: tf.Tensor):
#         """ Performs the training calculations in a tf.function. """
#
#         with tf.GradientTape() as tape:
#             # Forward Pass - (Re)Calculation of actions that caused saved states
#             action_probs, critic_values = self.actor_critic_net(sim_states)
#             critic_values = tf.squeeze(critic_values)
#
#             # Choose actions based on what was previously (randomly) sampled during simulation
#             actions = tf.reduce_sum(action_choices * action_probs, axis=1)
#
#             # Calculate expected returns
#             returns = self.get_expected_returns(rewards=rewards)
#
#             # Calculating loss values to update our network
#             loss, advantage = self.compute_loss(actions, critic_values, returns)
#
#         for x in self.actor_critic_net.weights:
#             if tf.reduce_any(tf.math.is_nan(x)):
#                 print("NAN detected in network weight")
#
#         # Compute the gradients from the loss
#         grads = tape.gradient(loss, self.actor_critic_net.trainable_variables)
#
#         # Clip gradients
#         if self.training_param["clip_gradients"]:
#             norm = self.training_param["clip_norm"]
#             grads = [tf.clip_by_norm(g, norm)
#                      for g in grads]
#
#         # Apply the gradients to the model's parameters
#         self.training_param["optimiser"].apply_gradients(
#             zip(grads, self.actor_critic_net.trainable_variables))
#
#         episode_reward = tf.math.reduce_sum(rewards)
#
#         return episode_reward, loss, advantage, returns, grads
