import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Sequential
import numpy as np
from HelperClasses import EpisodeBuffer, DataLogger


class GradAscentTrainerDiscrete(keras.models.Model):
    """
    Gradient ascent training algorithm.
    https://spinningup.openai.com/en/latest/algorithms/vpg.html
    """

    def __init__(self, actor_critic_net, training_param):
        super(GradAscentTrainerDiscrete, self).__init__()
        tf.random.set_seed(training_param["seed"])
        np.random.seed(training_param["seed"])
        self.actor_critic_net = actor_critic_net
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
        action_vel_probs, action_off_probs = action_probs

        vel_action_choice = np.random.choice(3, p=np.squeeze(action_vel_probs))
        off_action_choice = np.random.choice(3, p=np.squeeze(action_off_probs))
        return vel_action_choice, off_action_choice

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

        if self.training_param["standardise_returns"]:
            returns = ((returns - tf.math.reduce_mean(returns)) / (tf.math.reduce_std(returns) + eps))

        return returns

    @tf.function
    def compute_loss(
            self,
            action_vel_probs: tf.Tensor,
            action_off_probs: tf.Tensor,
            critic_values: tf.Tensor,
            returns: tf.Tensor):
        """ Computes the combined actor-critic loss."""

        # Advantage: How much better an action is given a state over a random action selected by the policy
        advantage = returns - critic_values

        action_vel_log_probs = tf.math.log(action_vel_probs)
        action_off_log_probs = tf.math.log(action_off_probs)

        actor_vel_loss = -tf.math.reduce_sum(action_vel_log_probs * advantage)
        actor_off_loss = -tf.math.reduce_sum(action_off_log_probs * advantage)

        critic_loss = self.training_param["loss_func"](critic_values, returns)

        loss = critic_loss + actor_vel_loss + actor_off_loss

        return loss, advantage

    def train_step(self):
        """ Performs a training step. """
        if self.training:
            # Gather and convert data from the buffer (data from simulation):
            timesteps, sim_states, rewards, sim_action_vel_choices, sim_action_off_choices \
                = self.buffer.experience

            sim_action_vel_choices = tf.keras.utils.to_categorical(sim_action_vel_choices)
            sim_action_off_choices = tf.keras.utils.to_categorical(sim_action_off_choices)

            episode_reward, loss, advantage, grads, returns = self.run_tape(
                sim_states=sim_states,
                action_vel_choices=sim_action_vel_choices,
                action_off_choices=sim_action_off_choices,
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

    @tf.function
    def run_tape(self,
                 sim_states: tf.Tensor,
                 action_vel_choices: tf.Tensor,
                 action_off_choices: tf.Tensor,
                 rewards: tf.Tensor):
        """ Performs the training calculations in a tf.function. """
        with tf.GradientTape() as tape:
            # Forward Pass - (Re)Calculation of actions that caused saved states
            action_vel_probs, action_off_probs, critic_values = self.actor_critic_net(sim_states)
            critic_values = tf.squeeze(critic_values)

            # Choose actions based on what was previously (randomly) sampled during simulation
            action_vel = tf.reduce_sum(action_vel_choices * action_vel_probs, axis=1)
            action_off = tf.reduce_sum(action_off_choices * action_off_probs, axis=1)

            # Calculate expected returns
            returns = self.get_expected_returns(rewards=rewards)

            # Calculating loss values to update our network
            loss, advantage = self.compute_loss(action_vel, action_off, critic_values, returns)

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
