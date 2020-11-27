import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

class ActorCriticNetDiscrete(keras.Model):
    """
    Neural network architecture for the actor.
    """
    def __init__(self, modelParam):
        super(ActorCriticNetDiscrete, self).__init__()
        # TODO Add variability in depth.
        self.inputLayer = layers.Input(shape=(modelParam["n_inputs"],))

        self.denseActorLayer1 = layers.Dense(modelParam["n_nodes"][0], activation=tf.nn.relu)(self.inputLayer)
        self.denseActorLayer2 = layers.Dense(modelParam["n_nodes"][1], activation=tf.nn.relu)(self.denseActorLayer1)
        self.outputLayerVel = layers.Dense(3, activation=tf.nn.softmax)(self.denseActorLayer2)
        self.outputLayerOff = layers.Dense(3, activation=tf.nn.softmax)(self.denseActorLayer2)

        self.denseCriticLayer1 = layers.Dense(modelParam["n_nodes"][0], activation=tf.nn.relu)(self.inputLayer)
        self.outputLayerCritic = layers.Dense(1, activation=tf.nn.softmax)(self.denseCriticLayer1)

        self.model = keras.Model(inputs=self.inputLayer, outputs=[self.outputLayerVel, self.outputLayerOff, self.outputLayerCritic])

    def call(self, inputs: tf.Tensor):
        y = self.model(inputs)
        return y

    def displayOverview(self):
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
        self.buffer = Buffer()  # Buffer class defined below
        self.training = True
        self.training_param = training_param

    def addExperience(self, s0, a0, r, s1, critic):
        if self.training:
            self.buffer.addToBuffer(s0, a0, r, s1, c)

    def trainStep(self):
        eps = np.finfo(np.float32).eps.item()  # Smallest number such that 1.0 + eps != 1.0
        if self.training:
            returns = []
            discounted_sum = 0

            # Return = SUM_t=0^inf (gamma*reward_t)
            for r in self.buffer.reward_hist[::-1]:
                discounted_sum = r + self.training_param["gamma"] * discounted_sum
                returns.insert(0, discounted_sum)

            # Normalise returns
            returns = np.array(returns)
            returns = (returns - np.mean(returns)) / (np.std(returns) + eps)
            returns = returns.tolist()

            # Fetch experience from buffer and calculate loss values:
            history = zip(self.buffer.action_vel_hist, self.buffer.action_off_hist,
                          self.buffer.critic_hist, returns)
            actor_losses_vel = []
            actor_losses_off = []
            critic_losses = []
            for actor_log_prob_vel, actor_log_prob_off, critic_val, ret in history:
                diff = ret - critic_val
                # if diff neg weaken, if diff pos strengthen connections
                actor_losses_vel.append(-actor_log_prob_vel * diff)  # actor velocity loss
                actor_losses_off.append(-actor_log_prob_off * diff)  # actor off loss
                critic_losses.append(
                    self.training_param["loss_function"](tf.expand_dims(critic_val, 0), tf.expand_dims(ret, 0)))

            # Backpropogation
            loss_value_actor = sum(actor_losses_vel) + sum(actor_losses_off)
            gradients_actor = self.tape_actor.gradient(loss_value_actor, self.actor_net.trainable_variables)
            self.training_param["optimiser"].apply_gradients(zip(gradients_actor, self.actor_net.trainable_variables))

            loss_value_critic = sum(critic_losses)
            gradients_critic = self.tape_critic.gradient(loss_value_critic, self.critic_net.trainable_variables)
            self.training_param["optimiser"].apply_gradients(zip(gradients_critic, self.critic_net.trainable_variables))

            # Clear loss values and reward history
            self.action_hist.clear()
            self.critic_hist.clear()
            self.reward_hist.clear()


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

