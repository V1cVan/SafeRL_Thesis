import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import logging



class ActorCriticNetDiscrete(keras.Model):
    """
    Neural network architecture for the actor.
    """
    def __init__(self, modelParam):
        super(ActorCriticNetDiscrete, self).__init__()
        # TODO Add variability in depth.
        # Actor net
        self.inputLayer = layers.Input(shape=(modelParam["n_inputs"],),
                                       name="inputStateLayer")

        self.denseActorLayer1 = layers.Dense(modelParam["n_nodes"][0],
                                             activation=tf.nn.relu,
                                             kernel_initializer='random_normal',
                                             bias_initializer='zeros',
                                             name="densActorLayer1")(self.inputLayer)
        self.denseActorLayer2 = layers.Dense(modelParam["n_nodes"][1], activation=tf.nn.relu,
                                             kernel_initializer='random_normal',
                                             bias_initializer='zeros',
                                             name="denseActorLayer2")(self.denseActorLayer1)
        self.outputLayerVel = layers.Dense(3, activation=tf.nn.softmax,
                                           kernel_initializer='random_normal',
                                           bias_initializer='zeros',
                                           name="outputActorLayerVel")(self.denseActorLayer1)
        self.outputLayerOff = layers.Dense(3, activation=tf.nn.softmax,
                                           kernel_initializer='random_normal',
                                           bias_initializer='zeros',
                                           name="outputActorLayerOff")(self.denseActorLayer1)

        self.denseCriticLayer1 = layers.Dense(modelParam["n_nodes"][0], activation=tf.nn.relu,
                                              kernel_initializer='random_normal',
                                              bias_initializer='zeros',
                                              name="denseCriticLayer1")(self.inputLayer)
        self.outputLayerCritic = layers.Dense(1, activation=tf.nn.softmax,
                                              kernel_initializer='random_normal',
                                              bias_initializer='zeros',
                                              name="outputCriticLayer")(self.denseCriticLayer1)

        self.model = keras.Model(inputs=self.inputLayer,
                                 outputs=[self.outputLayerVel, self.outputLayerOff, self.outputLayerCritic],
                                 name="ActorCriticNetwork")

    def call(self, inputs: tf.Tensor):
        y = self.model(inputs)
        return y

    def display_overview(self):
        # Display overview of model
        print("\nActor network model summary:")
        self.model.summary()
        print("############################\n")


class GradAscentTrainerDiscrete(keras.models.Model):
    """
    Gradient ascent training algorithm.
    https://spinningup.openai.com/en/latest/algorithms/vpg.html
    """

    def __init__(self, actor_critic_net, training_param):
        super(GradAscentTrainerDiscrete, self).__init__()
        self.actor_critic_net = actor_critic_net
        # self.cfg = cfg
        #self.buffer = Buffer()  # Buffer class defined below
        self.training = True
        self.training_param = training_param
        self.state = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        # Old implementation - Gives tensorflow warnings...
        # self.action_vel_probs = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        # self.action_off_probs = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        # self.action_choices = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        # self.critic_values = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        # self.rewards = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        self.states = []
        self.rewards = []
        self.actions_vel = []
        self.actions_off = []
        self.action_choices = []
        self.timestep = 1

        # Logging
        logging.basicConfig(level=logging.INFO, filename="./python/implementation/logfiles/trainer.log")
        with open('./python/implementation/logfiles/trainer.log', 'w'):
            pass  # Clear the log file of previous run

    def add_experience(self, s0, a0, a_choices, r, c):
        """ Saves the experience after the sim.step() call. """
        if self.training:
            self.states.append(s0)
            self.actions_vel.append(a0[0])
            self.actions_off.append(a0[1])
            self.action_choices.append(a_choices)
            self.rewards.append(r)
            # Old implementation. Gives TF warnings...
            # self.state.write(self.timestep, s0)
            # self.action_vel_probs.write(self.timestep, a0[0])
            # self.action_off_probs.write(self.timestep, a0[1])
            # self.action_choices.write(self.timestep, a_choices)
            # self.critic_values.write(self.timestep, c)
            # self.rewards.write(self.timestep, r)

    def get_experience(self):
        """ Returns the experiences """
        action_choices = np.array(self.action_choices, dtype=np.float32)
        action_vel_choice = action_choices[:, 0]
        action_off_choice = action_choices[:, 1]
        rewards = np.array(self.rewards, dtype=np.float32)
        states = np.array(self.states, dtype=np.float32)
        actions_vel = np.array(self.actions_vel, dtype=np.float32)
        actions_off = np.array(self.actions_off, dtype=np.float32)
        # state = self.state.stack()
        # action_vel_probs = self.action_vel_probs.stack()
        # action_off_probs = self.action_off_probs.stack()
        # action_choices = self.action_choices.stack()
        # critic_values = self.critic_values.stack()
        # rewards = self.rewards.stack()
        return states, actions_vel, actions_off, action_vel_choice, action_off_choice, rewards

    def set_tf_action_choices(self, states, actions_vel, actions_off, action_vel_choice, action_off_choice, rewards):
        self.states = tf.convert_to_tensor(states)
        self.actions_vel = tf.convert_to_tensor(actions_vel)
        self.actions_off = tf.convert_to_tensor(actions_off)
        self.action_vel_choice = tf.convert_to_tensor(action_vel_choice)
        self.action_off_choice = tf.convert_to_tensor(action_off_choice)
        self.rewards = tf.convert_to_tensor(rewards)

    def clear_experience(self):
        self.states.clear()
        self.rewards.clear()
        self.action_choices.clear()
        # self.action_vel_probs = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        # self.action_off_probs = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        # self.action_choices = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        # self.critic_values = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        # self.rewards = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)

    def set_neg_collision_reward(self, punishment):
        self.rewards[-1] = self.rewards[-1] - punishment

    def get_action_choice(self, action_probs):
        """ Randomly choose from the available actions."""
        action_vel_probs, action_off_probs = action_probs
        vel_actions_choice = tf.random.categorical(action_vel_probs, 1)[0,0]
        off_actions_choice = tf.random.categorical(action_off_probs, 1)[0,0]
        return vel_actions_choice, off_actions_choice

    def get_expected_returns(self, rewards: tf.Tensor) -> tf.Tensor:
        """
        Compute expected returns per timestep.
        """
        n = tf.shape(rewards)[0]
        returns = tf.TensorArray(dtype=tf.float32, size=n)
        eps = np.finfo(np.float32).eps.item()

        # Start from the end of `rewards` and accumulate reward sums
        # into the `returns` array
        rewards = tf.cast(rewards[::-1], dtype=tf.float32)
        discounted_sum = tf.constant(0.0)
        discounted_sum_shape = discounted_sum.shape
        for i in tf.range(n):
            reward = rewards[i]
            discounted_sum = reward + self.training_param["gamma"] * discounted_sum
            discounted_sum.set_shape(discounted_sum_shape)
            returns = returns.write(i, discounted_sum)
        returns = returns.stack()[::-1]

        returns = ((returns - tf.math.reduce_mean(returns))/(tf.math.reduce_std(returns) + eps))
        return returns

    def compute_loss(
            self,
            action_vel_probs: tf.Tensor,
            action_off_probs: tf.Tensor,
            critic_values: tf.Tensor,
            returns: tf.Tensor) -> tf.Tensor:
        """ Computes the combined actor-critic loss."""
        advantage = returns - critic_values

        action_vel_log_probs = tf.math.log(action_vel_probs)
        action_off_log_probs = tf.math.log(action_off_probs)
        # TODO PROBLEM LIES HERE ! query with bram if it is correct to do the actor losses like this?
        # TODO Probably need multiple critics!
        actor_vel_loss = tf.math.reduce_sum(tf.math.multiply(-action_vel_log_probs, advantage))
        actor_off_loss = tf.math.reduce_sum(tf.math.multiply(-action_off_log_probs, advantage))
        # actor_loss = tf.math.reduce_sum(-(action_vel_log_probs+action_off_log_probs)*advantage)  # ERROR!!!
        critic_loss = self.training_param["huber_loss"](critic_values, returns)
        loss = critic_loss + actor_vel_loss + actor_off_loss
        return loss


    #@tf.function
    def train_step(self):
        """ Performs a training step. """
        if self.training:
            with tf.GradientTape() as tape:
                action_vel_values = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
                action_off_values = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
                # Data from simulation:
                sim_states = tf.convert_to_tensor(self.states)
                rewards = tf.convert_to_tensor(self.rewards)
                sim_action_vel_choices = tf.convert_to_tensor(np.array(self.action_choices)[:,0])
                sim_action_off_choices = tf.convert_to_tensor(np.array(self.action_choices)[:,1])
                # Forward Pass - (Re)Calculation of actions that caused saved states
                # TODO log the values from ACPolicy class to ensure actions+critic correspond to calculations done here (indices etc.)
                action_vel_probs, action_off_probs, critic_values = self.actor_critic_net(sim_states)
                for t in tf.range(tf.size(sim_states[:, 0])):
                    vel_choice = tf.get_static_value(sim_action_vel_choices[t])
                    off_choice = tf.get_static_value(sim_action_off_choices[t])
                    action_vel_values.write(t, action_vel_probs[tf.get_static_value(t), vel_choice])
                    action_off_values.write(t, action_off_probs[tf.get_static_value(t), off_choice])
                critic_values = tf.squeeze(critic_values)
                action_vel_values = action_vel_values.stack()
                action_off_values = action_off_values.stack()

                # Calculate expected returns
                returns = self.get_expected_returns(rewards)

                # Calculating loss values to update our network
                loss = self.compute_loss(action_vel_values, action_off_values, critic_values, returns)

                # Compute the gradients from the loss
                grads = tape.gradient(loss, self.actor_critic_net.trainable_variables)

                # Apply the gradients to the model's parameters
            self.training_param["adam_optimiser"].apply_gradients(
                zip(grads, self.actor_critic_net.trainable_variables))
            episode_reward = tf.math.reduce_sum(rewards)

            return episode_reward




class Buffer(object):
    def __init__(self):
        self.state_hist = []
        self.action_vel_hist = []
        self.action_off_hist = []
        self.critic_hist = []
        self.reward_hist = []

    def addToBuffer(self, s0, a0, r, s1, c):
        self.state_hist.append(s0)
        self.action_vel_hist.append(a0[0])
        self.action_off_hist.append(a0[1])
        self.critic_hist.append(c)
        self.reward_hist.append(r)

