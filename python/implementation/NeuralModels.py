import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

class ActorNet(keras.Model):
    """
    Neural network architecture for the actor.
    """
    def __init__(self, modelParam):
        super(ActorNet, self).__init__()
        # TODO Add variability in depth.
        self.inputLayer = layers.Input(shape=(modelParam["n_inputs"],))
        self.denseLayer1 = layers.Dense(modelParam["n_nodes"][0], activation=tf.nn.relu)(self.inputLayer)
        self.denseLayer2 = layers.Dense(modelParam["n_nodes"][1], activation=tf.nn.relu)(self.denseLayer1)
        self.outputLayer = layers.Dense(modelParam["n_actions"], activation=tf.nn.tanh)(self.denseLayer2)
        self.model = keras.Model(inputs=self.inputLayer, outputs=self.outputLayer)

    def call(self, inputs):
        y = self.model(inputs)
        return y

    def displayOverview(self):
        # Display overview of model
        print("\nActor network model summary:")
        self.model.summary()
        print("############################\n")

class CriticNet(keras.Model):
    """
    Neural network architecture for the critic.
    """
    def __init__(self, modelParam):
        super(CriticNet, self).__init__()
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



class GradAscentTrainer(keras.models.Model):
    """
    Gradient ascent training algorithm.
    https://spinningup.openai.com/en/latest/algorithms/vpg.html
    """

    def __init__(self, actor, critic, training_param):
        super(GradAscentTrainer, self).__init__()
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
        print("addExperience")
        if self.training:
            critic = critic[0,0]
            self.buffer.append(np.array([s0, a0, r, s1, critic]))
            self.action_hist.append(s1)
            self.critic_hist.append(critic)
            self.reward_hist.append(r)

    def trainStep(self):
        print("trainStep")
        eps = np.finfo(np.float32).eps.item()  # Smallest number such that 1.0 + eps != 1.0
        if self.training:
            gamma = training_param["gamma"]
            returns = []
            discounted_sum = 0
            for r in self.reward_hist[::-1]:
                discounted_sum = r + gamma*discounted_sum
                returns.insert(0, discounted_sum)
            # Normalise
            returns = np.array(returns)
            returns = (returns - np.mean(returns))/(np.std(returns) + eps)
            returns = returns.tolist()

            # Calculate loss values:
            history = zip(self.action_hist[0], self.action_hist[1],
                          self.critic_hist, returns)
            actor_losses = []
            critic_losses = []
            for action1_val, action2_val, critic_val, ret in history:
                diff = ret - critic_val
                # if diff neg weaken, if diff pos strengthen connections
                actor_losses.append(-action1_val)
        # Fetch experience sample from buffer, calculate critic loss, update critic,
        # calculate actor loss, update actor, update target networks
        print()

