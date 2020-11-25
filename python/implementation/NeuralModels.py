import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

class ActorNetDiscrete(keras.Model):
    """
    Neural network architecture for the actor.
    """
    def __init__(self, modelParam):
        super(ActorNetDiscrete, self).__init__()
        # TODO Add variability in depth.
        self.inputLayer = layers.Input(shape=(modelParam["n_inputs"],))
        self.denseLayer1 = layers.Dense(modelParam["n_nodes"][0], activation=tf.nn.relu)(self.inputLayer)
        self.denseLayer2 = layers.Dense(modelParam["n_nodes"][1], activation=tf.nn.relu)(self.denseLayer1)
        self.outputLayerVel = layers.Dense(3, activation=tf.nn.softmax)(self.denseLayer2)
        self.outputLayerOff = layers.Dense(3, activation=tf.nn.softmax)(self.denseLayer2)
        self.model = keras.Model(inputs=self.inputLayer, outputs=[self.outputLayerVel, self.outputLayerOff])

    def call(self, inputs):
        y = self.model(inputs)
        return y

    def displayOverview(self):
        # Display overview of model
        print("\nActor network model summary:")
        self.model.summary()
        print("############################\n")

class CriticNetDiscrete(keras.Model):
    """
    Neural network architecture for the critic.
    """
    def __init__(self, modelParam):
        super(CriticNetDiscrete, self).__init__()
        # TODO Add variability in depth.
        self.inputLayer = layers.Input(shape=(modelParam["n_inputs"],))
        self.denseLayer1 = layers.Dense(modelParam["n_nodes"][0], activation=tf.nn.relu)(self.inputLayer)
        self.denseLayer2 = layers.Dense(modelParam["n_nodes"][1], activation=tf.nn.relu)(self.denseLayer1)
        self.outputLayer = layers.Dense(1, activation=tf.nn.softmax)(self.denseLayer2)
        self.model = keras.Model(inputs=self.inputLayer, outputs=self.outputLayer)

    def call(self, inputs):
        y = self.model(inputs)
        return y

    def displayOverview(self):
        # Display overview of model
        print("\nCritic network model summary:")
        self.model.summary()
        print("############################\n")



class GradAscentTrainerDiscrete(keras.models.Model):
    """
    Gradient ascent training algorithm.
    https://spinningup.openai.com/en/latest/algorithms/vpg.html
    """

    def __init__(self, actor, critic, training_param):
        super(GradAscentTrainerDiscrete, self).__init__()
        self.actor = actor
        self.critic = critic
        # self.cfg = cfg
        self.buffer = []  # [[s0, a, r, s1, critic], [] , ...]
        self.action_hist = []
        self.critic_hist = []
        self.reward_hist = []
        self.training = True
        self.training_param = training_param

    def addExperience(self, s0, a0, r, s1, critic):
        if self.training:
            critic = critic[0, 0]
            self.buffer.append(np.array([s0, a0, r, s1, critic]))
            self.action_hist.append(a0)
            self.critic_hist.append(critic)
            self.reward_hist.append(r)

    def trainStep(self):
        print("trainStep")
        eps = np.finfo(np.float32).eps.item()  # Smallest number such that 1.0 + eps != 1.0
        if self.training:
            with tf.GradientTape() as tape:
                gamma = self.training_param["gamma"]
                returns = []
                discounted_sum = 0
                # Return = SUM_t=0^inf (gamma*reward_t)
                for r in self.reward_hist[::-1]:
                    discounted_sum = r + gamma*discounted_sum
                    returns.insert(0, discounted_sum)
                # Normalise
                returns = np.array(returns)
                returns = (returns - np.mean(returns))/(np.std(returns) + eps)
                returns = returns.tolist()

                # Fetch experience from buffer and calculate loss values:
                history = zip(self.action_hist[0], self.action_hist[1],
                              self.critic_hist, returns)
                actor_losses_vel = []
                actor_losses_off = []
                critic_losses = []
                for actor_log_prob_vel, actor_log_prob_off, critic_val, ret in history:
                    diff = ret - critic_val
                    # if diff neg weaken, if diff pos strengthen connections
                    actor_losses_vel.append(-actor_log_prob_vel*diff)  # actor velocity loss
                    actor_losses_off.append(-actor_log_prob_off*diff)  # actor off loss
                    critic_losses.append(self.training_param["loss_function"](tf.expand_dims(critic_val, 0), tf.expand_dims(ret, 0)))

                # Backpropogation
                loss_value_actor = sum(actor_losses_vel) + sum(actor_losses_off)
                loss_value_critic = sum(critic_losses)

        print()

