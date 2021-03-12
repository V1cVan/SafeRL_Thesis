import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Sequential
import numpy as np
from HelperClasses import EpisodeBuffer, DataLogger, TrainingBuffer


class DqnAgent(keras.models.Model):
    """
    double deep q network trainer
    """

    def __init__(self, network, training_param):
        super(DqnAgent, self).__init__()
        tf.random.set_seed(training_param["seed"])
        np.random.seed(training_param["seed"])
        self.latest_action = None
        self.Q_target_net = network
        self.Q_actual_net = network
        self.reward_weights = training_param["reward_weights"]
        # TODO implement data logging class for debugging training
        self.training = True
        self.training_param = training_param
        self.stop_flags = None
        self.eps_final = 0.1
        self.decay = training_param["decay_rate"]

        self.epsilon = training_param["epsilon_max"]
        self.prev_epsilon = self.epsilon
        self.epsilon_decay_count = 1
        self.evaluation = False
        self.episode = 1
        self.buffer = TrainingBuffer(max_mem_size=training_param["buffer_size"],
                                     batch_size=training_param["batch_size"])
        self.gamma = training_param["gamma"]


    def set_neg_collision_reward(self, timestep, punishment):
        """ Sets a negative reward if a collision occurs. """
        self.buffer.alter_reward_at_timestep(timestep, punishment)

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

        if np.random.rand() < epsilon:
            return np.random.randint(0, 5)
        else:
            # Otherwise, query the DQN for an action
            self.latest_action = np.argmax(Q, axis=1)[0]
            return self.latest_action


    def update_target_net(self):
        self.Q_target_net.set_weights(self.Q_actual_net.get_weights())

    def train_step(self):
        """ Performs a training step. """
        if self.training:

            # Gather and convert data from the buffer (data from simulation):
            # Sample mini-batch from memory
            states, actions, rewards, next_states, done = self.buffer.get_training_samples()

            one_hot_actions = tf.keras.utils.to_categorical(actions, num_classes=5)

            episode_reward, loss = self.run_tape(
                states=states,
                actions=one_hot_actions,
                rewards=rewards,
                next_states=next_states,
                done=done)

            # self.buffer.set_training_variables(
            #     episode_num=self.episode,
            #     episode_reward=episode_reward,
            #     losses=loss,
            #     advantage=tf.squeeze(tf.convert_to_tensor(advantage)),
            #     returns=returns,
            #     gradients=np.squeeze(grads),
            #     model_weights=self.actor_critic_net.weights)

            return episode_reward, loss

    @tf.function
    def run_tape(self,
                 states: tf.Tensor,
                 actions: tf.Tensor,
                 rewards: tf.Tensor,
                 next_states: tf.Tensor,
                 done: tf.Tensor):
        """ Performs the training calculations in a tf.function. """
        ones = tf.ones(tf.shape(done), dtype=tf.dtypes.float32)

        Q_output = self.Q_target_net(next_states)
        Q_target = rewards + (ones - done) * (self.gamma * tf.reduce_max(Q_output, axis=1))

        if self.training_param["standardise_returns"]:
            eps = np.finfo(np.float32).eps.item()
            Q_target = Q_target - tf.math.reduce_mean(Q_target) / (tf.math.reduce_std(Q_target) + eps)

        with tf.GradientTape() as tape:
            Q_output = self.Q_actual_net(states)
            Q_predicted = tf.reduce_sum(Q_output * actions, axis=1)

            loss_value = self.training_param["loss_func"](Q_target, Q_predicted)
            # loss_value = tf.losses.MSE(y_true=target_output, y_pred=predicted_output)
            # loss_value = tf.reduce_mean(tf.square(Q_target - Q_predicted))
            # Choose actions based on what was previously (randomly) sampled during simulation

        grads = tape.gradient(loss_value, self.Q_actual_net.trainable_variables)
        # Clip gradients
        if self.training_param["clip_gradients"]:
            norm = self.training_param["clip_norm"]
            grads = [tf.clip_by_norm(g, norm)
                     for g in grads]

        self.training_param["optimiser"].apply_gradients(zip(grads, self.Q_actual_net.trainable_variables))
        reward = tf.math.reduce_sum(rewards)
        reward = reward/len(self.buffer.buffer)

        return reward, loss_value





class SpgAgentSingle(keras.models.Model):
    """
    Gradient ascent training algorithm.
    https://spinningup.openai.com/en/latest/algorithms/vpg.html
    """
    def __init__(self, network, training_param):
        super(SpgAgentSingle, self).__init__()
        tf.random.set_seed(training_param["seed"])
        np.random.seed(training_param["seed"])
        self.actor_critic_net = network
        self.reward_weights = training_param["reward_weights"]
        # TODO implement data logging class for debugging training
        self.training = True
        self.training_param = training_param
        self.episode = 1
        self.buffer = EpisodeBuffer()
        self.actions = []
        self.states = []
        self.rewards = []

    def set_neg_collision_reward(self, timestep, punishment):
        """ Sets a negative reward if a collision occurs. """
        self.buffer.alter_reward_at_timestep(timestep, punishment)

    def get_action_choice(self, action_probs):
        """ Randomly choose from the available actions."""
        action_choice = np.random.choice(5, p=np.squeeze(action_probs))
        return action_choice

    @tf.function
    def get_expected_returns(self,
                             rewards: tf.Tensor) -> tf.Tensor:
        """ Computes expected returns per timestep. """

        # Initialise returns array outside of tf.function
        n = tf.shape(rewards)[0]
        returns = tf.TensorArray(dtype=tf.float32, size=n)

        eps = np.finfo(np.float32).eps.item()
        # Start from the end of rewards and accumulate reward sums into the returns array
        rewards = tf.cast(rewards[::-1], dtype=tf.float32)
        discounted_sum = tf.constant(0.0)
        discounted_sum_shape = discounted_sum.shape
        for i in tf.range(tf.shape(rewards)[0]):
            discounted_sum = rewards[i] + self.training_param["gamma"] * discounted_sum
            discounted_sum.set_shape(discounted_sum_shape)
            returns = returns.write(i, discounted_sum)
        returns = returns.stack()[::-1]

        if self.training_param["standardise_rewards"]:
            returns = ((returns - tf.math.reduce_mean(returns)) / (tf.math.reduce_std(returns) + eps))

        return returns

    #@tf.function
    def compute_loss(
            self,
            action_probs: tf.Tensor,
            critic_values: tf.Tensor,
            returns: tf.Tensor):
        """ Computes the combined actor-critic loss."""

        # Advantage: How much better an action is given a state over a random action selected by the policy
        advantage = returns - critic_values

        action_log_probs = tf.math.log(action_probs)

        actor_loss = -tf.math.reduce_sum(action_log_probs * advantage)

        # critic_loss = tf.losses.MSE(y_true=critic_values, y_pred=returns)
        # critic_loss = tf.reduce_mean(tf.square(critic_values - returns))
        critic_loss = self.training_param["loss_func"](critic_values, returns)

        loss = critic_loss + actor_loss

        return loss, advantage

    def train_step(self):
        """ Performs a training step. """
        if self.training:
            # Gather and convert data from the buffer (data from simulation):
            timesteps, sim_states, rewards, sim_action_choices \
                = self.buffer.experience

            sim_action_choices = tf.keras.utils.to_categorical(sim_action_choices)

            episode_reward, loss, advantage, grads, returns = self.run_tape(
                sim_states=sim_states,
                action_choices=sim_action_choices,
                rewards=rewards)

            self.buffer.set_training_variables(
                episode_num=self.episode,
                episode_reward=episode_reward,
                losses=loss,
                advantage=tf.squeeze(tf.convert_to_tensor(advantage)),
                returns=returns,
                gradients=np.squeeze(grads),
                model_weights=self.actor_critic_net.weights)

            return episode_reward, loss

    #@tf.function
    def run_tape(self,
                 sim_states: tf.Tensor,
                 action_choices: tf.Tensor,
                 rewards: tf.Tensor):
        """ Performs the training calculations in a tf.function. """


        with tf.GradientTape() as tape:
            # Forward Pass - (Re)Calculation of actions that caused saved states
            action_probs, critic_values = self.actor_critic_net(sim_states)
            critic_values = tf.squeeze(critic_values)

            # Choose actions based on what was previously (randomly) sampled during simulation
            actions = tf.reduce_sum(action_choices * action_probs, axis=1)

            # Calculate expected returns
            returns = self.get_expected_returns(rewards=rewards)

            # Calculating loss values to update our network
            loss, advantage = self.compute_loss(actions, critic_values, returns)

        for x in self.actor_critic_net.weights:
            if tf.reduce_any(tf.math.is_nan(x)):
                print("NAN detected in network weight")

        # Compute the gradients from the loss
        grads = tape.gradient(loss, self.actor_critic_net.trainable_variables)

        # Clip gradients
        if self.training_param["clip_gradients"]:
            norm = self.training_param["clip_norm"]
            grads = [tf.clip_by_norm(g, norm)
                     for g in grads]

        # Apply the gradients to the model's parameters
        self.training_param["optimiser"].apply_gradients(
            zip(grads, self.actor_critic_net.trainable_variables))

        episode_reward = tf.math.reduce_sum(rewards)

        return episode_reward, loss, advantage, returns, grads

class SpgAgentDouble(keras.models.Model):
    """
    Gradient ascent training algorithm.
    https://spinningup.openai.com/en/latest/algorithms/vpg.html
    """

    def __init__(self, network, training_param):
        super(SpgAgentDouble, self).__init__()
        tf.random.set_seed(training_param["seed"])
        np.random.seed(training_param["seed"])
        self.actor_critic_net = network
        self.reward_weights = training_param["reward_weights"]
        # TODO implement data logging class for debugging training
        self.training = True
        self.training_param = training_param
        self.episode = 1
        self.buffer = EpisodeBuffer()
        self.actions = []
        self.states = []
        self.rewards = []

    def set_neg_collision_reward(self, timestep, punishment):
        """ Sets a negative reward if a collision occurs. """
        self.buffer.alter_reward_at_timestep(timestep, punishment)

    def get_action_choice(self, action_probs):
        """ Randomly choose from the available actions."""
        action_choice = np.random.choice(5, p=np.squeeze(action_probs))
        return action_choice

    @tf.function
    def get_expected_returns(self,
                             rewards: tf.Tensor) -> tf.Tensor:
        """ Computes expected returns per timestep. """

        # Initialise returns array outside of tf.function
        n = tf.shape(rewards)[0]
        returns = tf.TensorArray(dtype=tf.float32, size=n)

        eps = np.finfo(np.float32).eps.item()
        # Start from the end of rewards and accumulate reward sums into the returns array
        rewards = tf.cast(rewards[::-1], dtype=tf.float32)
        discounted_sum = tf.constant(0.0)
        discounted_sum_shape = discounted_sum.shape
        for i in tf.range(tf.shape(rewards)[0]):
            discounted_sum = rewards[i] + self.training_param["gamma"] * discounted_sum
            discounted_sum.set_shape(discounted_sum_shape)
            returns = returns.write(i, discounted_sum)
        returns = returns.stack()[::-1]

        if self.training_param["standardise_rewards"]:
            returns = ((returns - tf.math.reduce_mean(returns)) / (tf.math.reduce_std(returns) + eps))

        return returns

    # @tf.function
    def compute_loss(
            self,
            action_probs: tf.Tensor,
            critic_values: tf.Tensor,
            returns: tf.Tensor):
        """ Computes the combined actor-critic loss."""

        # Advantage: How much better an action is given a state over a random action selected by the policy
        advantage = returns - critic_values

        action_log_probs = tf.math.log(action_probs)

        actor_loss = -tf.math.reduce_sum(action_log_probs * advantage)

        # critic_loss = tf.losses.MSE(y_true=critic_values, y_pred=returns)
        # critic_loss = tf.reduce_mean(tf.square(critic_values - returns))
        critic_loss = self.training_param["loss_func"](critic_values, returns)

        loss = critic_loss + actor_loss

        return loss, advantage

    def train_step(self):
        """ Performs a training step. """
        if self.training:
            # Gather and convert data from the buffer (data from simulation):
            timesteps, sim_states, rewards, sim_action_choices \
                = self.buffer.experience

            sim_action_choices = tf.keras.utils.to_categorical(sim_action_choices)

            episode_reward, loss, advantage, grads, returns = self.run_tape(
                sim_states=sim_states,
                action_choices=sim_action_choices,
                rewards=rewards)

            self.buffer.set_training_variables(
                episode_num=self.episode,
                episode_reward=episode_reward,
                losses=loss,
                advantage=tf.squeeze(tf.convert_to_tensor(advantage)),
                returns=returns,
                gradients=np.squeeze(grads),
                model_weights=self.actor_critic_net.weights)

            return episode_reward, loss

    # @tf.function
    def run_tape(self,
                 sim_states: tf.Tensor,
                 action_choices: tf.Tensor,
                 rewards: tf.Tensor):
        """ Performs the training calculations in a tf.function. """

        with tf.GradientTape() as tape:
            # Forward Pass - (Re)Calculation of actions that caused saved states
            action_probs, critic_values = self.actor_critic_net(sim_states)
            critic_values = tf.squeeze(critic_values)

            # Choose actions based on what was previously (randomly) sampled during simulation
            actions = tf.reduce_sum(action_choices * action_probs, axis=1)

            # Calculate expected returns
            returns = self.get_expected_returns(rewards=rewards)

            # Calculating loss values to update our network
            loss, advantage = self.compute_loss(actions, critic_values, returns)

        for x in self.actor_critic_net.weights:
            if tf.reduce_any(tf.math.is_nan(x)):
                print("NAN detected in network weight")

        # Compute the gradients from the loss
        grads = tape.gradient(loss, self.actor_critic_net.trainable_variables)

        # Clip gradients
        if self.training_param["clip_gradients"]:
            norm = self.training_param["clip_norm"]
            grads = [tf.clip_by_norm(g, norm)
                     for g in grads]

        # Apply the gradients to the model's parameters
        self.training_param["optimiser"].apply_gradients(
            zip(grads, self.actor_critic_net.trainable_variables))

        episode_reward = tf.math.reduce_sum(rewards)

        return episode_reward, loss, advantage, returns, grads
