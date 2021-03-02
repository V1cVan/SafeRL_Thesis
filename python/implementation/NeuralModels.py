import tensorflow as tf
from tensorflow import keras
<<<<<<< HEAD
from tensorflow.keras import layers, Sequential
=======
from tensorflow.keras import layers
>>>>>>> 3e009d148339af9b90fd5bc3e6092d76d2fc34b4
import numpy as np
from HelperClasses import EpisodeBuffer, DataLogger


<<<<<<< HEAD
class ActorCriticNetDiscrete(keras.Model):
    """
    Neural network architecture for the actor and critic.
    Here actions DO feed into critic network.
    Actor and critic networks are separate.
    """

    def __init__(self, modelParam):
        super(ActorCriticNetDiscrete, self).__init__()
        tf.random.set_seed(modelParam["seed"])
        np.random.seed(modelParam["seed"])
        act_func = modelParam["activation_function"]
        n_units = modelParam["n_units"]
        n_inputs = modelParam["n_inputs"]
        n_actions = modelParam["n_actions"]

        self.he = tf.keras.initializers.HeUniform()
        glorot = tf.keras.initializers.GlorotUniform()
        normal = tf.keras.initializers.RandomUniform(minval=-0.03, maxval=0.03)
        var_scale = tf.keras.initializers.VarianceScaling(scale=2.0, mode='fan_in', distribution='truncated_normal')

        input_layer = layers.Input(shape=(n_inputs,),
                                   name="inputStateLayer")

        # Actor net:
        dense_actor_layer1 = self.dense_layer(num_units=n_units[0], act_func=act_func)(input_layer)
        dense_actor_layer2 = self.dense_layer(num_units=n_units[1], act_func=act_func)(dense_actor_layer1)
        output_layer_vel = layers.Dense(3,
                                        name="OutputLayerVelocity",
                                        activation=tf.nn.softmax,
                                        kernel_initializer=glorot,
                                        bias_initializer=tf.keras.initializers.Constant(0))(dense_actor_layer2)
        output_layer_steer = layers.Dense(3,
                                          name="OutputLayerSteering",
                                          activation=tf.nn.softmax,
                                          kernel_initializer=glorot,
                                          bias_initializer=tf.keras.initializers.Constant(0))(dense_actor_layer2)

        # Critic net:
        dense_critic_layer1 = self.dense_layer(num_units=n_units[0], act_func=act_func)(
            layers.concatenate([output_layer_vel, output_layer_steer, input_layer]))
        dense_critic_layer2 = self.dense_layer(num_units=n_units[1], act_func=act_func)(dense_critic_layer1)
        output_layer_critic = layers.Dense(1, name="OutputLayerCritic")(dense_critic_layer2)

        self.model = keras.Model(inputs=input_layer,
                                 outputs=[output_layer_vel, output_layer_steer, output_layer_critic],
                                 name="ActorCriticNetwork")

    def dense_layer(self, num_units, act_func):
        return layers.Dense(
            num_units,
            activation=act_func,
            kernel_initializer=self.he)

    def call(self, inputs: tf.Tensor):
        """ Returns the output of the model given an input. """
        return self.model(inputs)
=======
class ActorCriticNetDiscrete_1(keras.Model):
    """
    Neural network architecture for the actor and critic.
    Here actions do NOT feed into critic network.
    Actor and critic networks are separate.
    """
    def __init__(self, modelParam):
        super(ActorCriticNetDiscrete_1, self).__init__()
        tf.random.set_seed(modelParam["seed"])
        np.random.seed(modelParam["seed"])
        act_func = modelParam["activation_function"]

        initializer_relu = tf.keras.initializers.HeUniform()
        initializer_softmax = tf.keras.initializers.GlorotUniform()

        # TODO Add variability in depth.
        # Actor net:
        self.inputLayer = layers.Input(shape=(modelParam["n_inputs"],),
                                       name="inputStateLayer")

        self.denseActorLayer1 = layers.Dense(modelParam["n_nodes"][0],
                                             activation=act_func,
                                             kernel_initializer=initializer_relu,
                                             name="denseActorLayer1")(self.inputLayer)
        # Critic net:
        self.denseCriticLayer1 = layers.Dense(modelParam["n_nodes"][0],
                                              activation=act_func,
                                              kernel_initializer=initializer_relu,
                                              name="denseCriticLayer1")(self.inputLayer)

        if modelParam["n_nodes"][1] == 0:  # if no depth in network:
            self.outputLayerVel = layers.Dense(3, activation=tf.nn.softmax,
                                               kernel_initializer=initializer_softmax,
                                               name="outputActorLayerVel")(self.denseActorLayer1)
            self.outputLayerOff = layers.Dense(3, activation=tf.nn.softmax,
                                               kernel_initializer=initializer_softmax,
                                               name="outputActorLayerOff")(self.denseActorLayer1)
            self.outputLayerCritic = layers.Dense(1,
                                                  name="outputCriticLayer")(self.denseCriticLayer1)
        else:  # if depth in network exists
            self.denseActorLayer2 = layers.Dense(modelParam["n_nodes"][1],
                                                 activation=act_func,
                                                 kernel_initializer=initializer_relu,
                                                 name="denseActorLayer2")(self.denseActorLayer1)
            self.outputLayerVel = layers.Dense(3, activation=tf.nn.softmax,
                                               kernel_initializer=initializer_softmax,
                                               name="outputActorLayerVel")(self.denseActorLayer2)
            self.outputLayerOff = layers.Dense(3, activation=tf.nn.softmax,
                                               kernel_initializer=initializer_softmax,
                                               name="outputActorLayerOff")(self.denseActorLayer2)

            self.denseCriticLayer2 = layers.Dense(modelParam["n_nodes"][1],
                                                  activation=act_func,
                                                  kernel_initializer=initializer_relu,
                                                  name="denseCriticLayer2")(self.denseCriticLayer1)
            self.outputLayerCritic = layers.Dense(1,
                                                  name="outputCriticLayer")(self.denseCriticLayer2)



        # TODO look at batch normalisation



        self.model = keras.Model(inputs=self.inputLayer,
                                 outputs=[self.outputLayerVel, self.outputLayerOff, self.outputLayerCritic],
                                 name="ActorCriticNetwork_basic")

    def call(self, inputs: tf.Tensor):
        """ Returns the output of the model given an input. """
        y = self.model(inputs)
        return y

    def display_overview(self):
        """ Displays an overview of the model. """
        self.model.summary()
        keras.utils.plot_model(self.model, show_shapes=True, show_layer_names=True)


class ActorCriticNetDiscrete_2(keras.Model):
    """
    Neural network architecture for the actor and critic.
    Here actions DO feed into critic network.
    Actor and critic networks are separate.
    """
    def __init__(self, modelParam):
        super(ActorCriticNetDiscrete_2, self).__init__()
        tf.random.set_seed(modelParam["seed"])
        np.random.seed(modelParam["seed"])
        act_func = modelParam["activation_function"]

        initializer_relu = tf.keras.initializers.HeUniform()
        initializer_softmax = tf.keras.initializers.GlorotUniform()

        if modelParam["n_nodes"][1] == 0:  # if no depth in network:
            print("Warning: Model not correctly implemented without a depth in the network!")

        # TODO Add variability in depth.
        # Actor net:
        inputLayer = layers.Input(shape=(modelParam["n_inputs"],),
                                  name="inputStateLayer")

        denseActorLayer1 = layers.Dense(modelParam["n_nodes"][0],
                                        activation=act_func,
                                        kernel_initializer=initializer_relu,
                                        name="denseActorLayer1")(inputLayer)
        denseActorLayer2 = layers.Dense(modelParam["n_nodes"][1],
                                        activation=act_func,
                                        kernel_initializer=initializer_relu,
                                        name="denseActorLayer2")(denseActorLayer1)
        outputLayerVel = layers.Dense(3, activation=tf.nn.softmax,
                                      kernel_initializer=initializer_softmax,
                                      name="outputActorLayerVel")(denseActorLayer2)
        outputLayerOff = layers.Dense(3, activation=tf.nn.softmax,
                                      kernel_initializer=initializer_softmax,
                                      name="outputActorLayerOff")(denseActorLayer2)


        # Critic net:
        denseCriticLayer1 = layers.Dense(modelParam["n_nodes"][0],
                                         activation=act_func,
                                         kernel_initializer=initializer_relu,
                                         name="denseCriticLayer1")(layers.concatenate([outputLayerVel,
                                                                                       outputLayerOff,
                                                                                       inputLayer]))
        denseCriticLayer2 = layers.Dense(modelParam["n_nodes"][1],
                                         activation=act_func,
                                         kernel_initializer=initializer_relu,
                                         name="denseCriticLayer2")(denseCriticLayer1)
        outputLayerCritic = layers.Dense(1,
                                         name="outputCriticLayer")(denseCriticLayer2)



        # TODO look at batch normalisation



        self.model = keras.Model(inputs=inputLayer,
                                 outputs=[outputLayerVel, outputLayerOff, outputLayerCritic],
                                 name="ActorCriticNetwork_advanced")

    def call(self, inputs: tf.Tensor):
        """ Returns the output of the model given an input. """
        y = self.model(inputs)
        return y
>>>>>>> 3e009d148339af9b90fd5bc3e6092d76d2fc34b4

    def display_overview(self):
        """ Displays an overview of the model. """
        self.model.summary()
        keras.utils.plot_model(self.model, show_shapes=True, show_layer_names=True)


<<<<<<< HEAD
# class ActorCriticNetDiscrete(keras.Model):
#     """
#     Neural network architecture for the actor and critic.
#     Here actions DO feed into critic network.
#     Actor and critic networks are separate.
#     """
#
#     def __init__(self, modelParam):
#         super(ActorCriticNetDiscrete, self).__init__()
#         tf.random.set_seed(modelParam["seed"])
#         np.random.seed(modelParam["seed"])
#         act_func = modelParam["activation_function"]
#         n_units = modelParam["n_units"]
#         n_inputs = modelParam["n_inputs"]
#         n_actions = modelParam["n_actions"]
#
#         # initializer_relu = tf.keras.initializers.HeUniform()
#         # initializer_softmax = tf.keras.initializers.GlorotUniform()
#
#         input_layer = layers.Input(shape=(n_inputs,),
#                                    name="inputStateLayer")
#
#         # Actor net:
#         dense_actor_layer1 = self.dense_layer(num_units=n_units[0], act_func=act_func)(input_layer)
#         dense_actor_layer2 = self.dense_layer(num_units=n_units[1], act_func=act_func)(dense_actor_layer1)
#         output_layer_vel = layers.Dense(3,
#                                         name="OutputLayerVelocity",
#                                         activation=tf.nn.softmax,
#                                         kernel_initializer=tf.keras.initializers.RandomUniform(
#                                             minval=-0.03, maxval=0.03),
#                                         bias_initializer=tf.keras.initializers.Constant(0))(dense_actor_layer2)
#         output_layer_steer = layers.Dense(3,
#                                           name="OutputLayerSteering",
#                                           activation=tf.nn.softmax,
#                                           kernel_initializer=tf.keras.initializers.RandomUniform(
#                                               minval=-0.03, maxval=0.03),
#                                           bias_initializer=tf.keras.initializers.Constant(0))(dense_actor_layer2)
#
#         # Critic net:
#         dense_critic_layer1 = self.dense_layer(num_units=n_units[0], act_func=act_func)(
#             layers.concatenate([output_layer_vel, output_layer_steer, input_layer]))
#         dense_critic_layer2 = self.dense_layer(num_units=n_units[1], act_func=act_func)(dense_critic_layer1)
#         output_layer_critic = layers.Dense(1,
#                                            name="OutputLayerCritic",
#                                            kernel_initializer=tf.keras.initializers.RandomUniform(
#                                                minval=-0.03, maxval=0.03),
#                                            bias_initializer=tf.keras.initializers.Constant(0))(dense_critic_layer2)
#
#         self.model = keras.Model(inputs=input_layer,
#                                  outputs=[output_layer_vel, output_layer_steer, output_layer_critic],
#                                  name="ActorCriticNetwork")
#
#     def dense_layer(self, num_units, act_func):
#         return layers.Dense(
#             num_units,
#             activation=act_func,
#             kernel_initializer=tf.keras.initializers.VarianceScaling(
#                 scale=2.0, mode='fan_in', distribution='truncated_normal'))
#
#     def call(self, inputs: tf.Tensor):
#         """ Returns the output of the model given an input. """
#         return self.model(inputs)
#
#     def display_overview(self):
#         """ Displays an overview of the model. """
#         self.model.summary()
#         keras.utils.plot_model(self.model, show_shapes=True, show_layer_names=True)


=======
>>>>>>> 3e009d148339af9b90fd5bc3e6092d76d2fc34b4
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
<<<<<<< HEAD
=======
        self.temperature = 1
>>>>>>> 3e009d148339af9b90fd5bc3e6092d76d2fc34b4

    def set_neg_collision_reward(self, timestep, punishment):
        """ Sets a negative reward if a collision occurs. """
        self.buffer.alter_reward_at_timestep(timestep, punishment)

    def get_action_choice(self, action_probs):
        """ Randomly choose from the available actions."""
        action_vel_probs, action_off_probs = action_probs

<<<<<<< HEAD
        if np.isnan(action_vel_probs).any() or np.isnan(action_vel_probs).any():
            x = 10

        # np.random.choice accepts probabilities
        # vel_action_choice = 1
        # off_action_choice = 1
        vel_action_choice = np.random.choice(3, p=np.squeeze(action_vel_probs))
        off_action_choice = np.random.choice(3, p=np.squeeze(action_vel_probs))
        return vel_action_choice, off_action_choice

    # @tf.function
=======
        action_vel_log_probs = np.log(action_vel_probs) / self.temperature
        action_off_log_probs = np.log(action_off_probs) / self.temperature

        action_vel = np.exp(action_vel_log_probs) / np.sum(np.exp(action_vel_log_probs))
        action_off = np.exp(action_off_log_probs) / np.sum(np.exp(action_off_log_probs))

        if np.isnan(action_vel).any() or np.isnan(action_off).any():
            x = 10

        # np.random.choice accepts probabilities
        vel_action_choice = np.random.choice(3, p=np.squeeze(action_vel))
        off_action_choice = np.random.choice(3, p=np.squeeze(action_off))
        return vel_action_choice, off_action_choice

    #@tf.function
>>>>>>> 3e009d148339af9b90fd5bc3e6092d76d2fc34b4
    def get_expected_returns(self, rewards: tf.Tensor) -> tf.Tensor:
        """
        Computes expected returns per timestep.
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

<<<<<<< HEAD
        returns = ((returns - tf.math.reduce_mean(returns)) / (tf.math.reduce_std(returns) + eps))
        return returns

    # @tf.function
=======
        returns = ((returns - tf.math.reduce_mean(returns))/(tf.math.reduce_std(returns) + eps))
        return returns

    #@tf.function
>>>>>>> 3e009d148339af9b90fd5bc3e6092d76d2fc34b4
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
        actor_vel_loss = tf.math.reduce_sum(tf.math.multiply(-action_vel_log_probs, advantage))
        actor_off_loss = tf.math.reduce_sum(tf.math.multiply(-action_off_log_probs, advantage))

        critic_loss = self.training_param["huber_loss"](critic_values, returns)

        loss = critic_loss + actor_vel_loss + actor_off_loss

        # TODO look at a critic receiving actions directly in the model structure?

        return loss, advantage

<<<<<<< HEAD
    # @tf.function
=======

    #@tf.function
>>>>>>> 3e009d148339af9b90fd5bc3e6092d76d2fc34b4
    def train_step(self):
        """ Performs a training step. """
        if self.training:
            with tf.GradientTape() as tape:

                # Gather and convert data from the buffer (data from simulation):
                timesteps, sim_states, rewards, sim_action_vel_choices, sim_action_off_choices \
                    = self.buffer.experience

                action_vel = tf.TensorArray(dtype=tf.float32, size=tf.size(timesteps))
                action_off = tf.TensorArray(dtype=tf.float32, size=tf.size(timesteps))

                # TODO tf.print to check values for nans in weights
                # tf.print(self.actor_critic_net.weights)

                # Forward Pass - (Re)Calculation of actions that caused saved states
                # TODO log the values from ACPolicy class to ensure actions+critic correspond to calculations done here (indices etc.)
                action_vel_probs, action_off_probs, critic_values = self.actor_critic_net(sim_states)
                critic_values = tf.squeeze(critic_values)
                # tf.print("states: ")
                # tf.print(sim_states)
                # tf.print("action_vel_probs: ")
                # tf.print(action_vel_probs)
                # tf.print("action_off_probs: ")
                # tf.print(action_off_probs)
                # tf.print("critic_values: ")
                # tf.print(critic_values)

                # Choose actions based on what was previously (randomly) sampled during simulation
                for t in range(0, len(timesteps)):
<<<<<<< HEAD
                    vel_choice = sim_action_vel_choices[t, 0]
                    off_choice = sim_action_off_choices[t, 0]
=======
                    vel_choice = sim_action_vel_choices[t,0]
                    off_choice = sim_action_off_choices[t,0]
>>>>>>> 3e009d148339af9b90fd5bc3e6092d76d2fc34b4
                    action_vel.write(t, action_vel_probs[t, vel_choice])
                    action_off.write(t, action_off_probs[t, off_choice])

                action_vel = action_vel.stack()
                action_off = action_off.stack()

                # Calculate expected returns
                returns = self.get_expected_returns(rewards)

                # Calculating loss values to update our network
                loss, advantage = self.compute_loss(action_vel, action_off, critic_values, returns)

<<<<<<< HEAD
=======

>>>>>>> 3e009d148339af9b90fd5bc3e6092d76d2fc34b4
            for x in self.actor_critic_net.weights:
                if tf.reduce_any(tf.math.is_nan(x)):
                    print("NAN detected in network weight")

<<<<<<< HEAD
            # Compute the gradients from the loss
            grads = tape.gradient(loss, self.actor_critic_net.trainable_variables)

            # # Clip gradients
            # if self.training_param["clip_gradients"]:
            #     norm = self.training_param["clip_norm"]
            #     grads = [tf.clip_by_norm(g, norm)
            #              for g in grads]
=======

            # Compute the gradients from the loss
            grads = tape.gradient(loss, self.actor_critic_net.trainable_variables)


>>>>>>> 3e009d148339af9b90fd5bc3e6092d76d2fc34b4

            # Apply the gradients to the model's parameters
            self.training_param["adam_optimiser"].apply_gradients(
                zip(grads, self.actor_critic_net.trainable_variables))

            episode_reward = tf.math.reduce_sum(rewards)

<<<<<<< HEAD
            self.buffer.set_training_variables(self.episode, loss, advantage, returns, np.squeeze(grads),
                                               self.actor_critic_net.weights)

            return episode_reward
=======
            self.buffer.set_training_variables(self.episode, loss, advantage, returns, np.squeeze(grads), self.actor_critic_net.weights)

            return episode_reward




>>>>>>> 3e009d148339af9b90fd5bc3e6092d76d2fc34b4
