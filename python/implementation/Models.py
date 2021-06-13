import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Sequential
import numpy as np
from HelperClasses import EpisodeBuffer, DataLogger


class CNN(keras.Model):
    """
    Builds a convolutional neural network to handle the dynamic part of the state vector.
    """
    def __init__(self, model_param):
        super(CNN, self).__init__()
        self.model_param = model_param
        tf.random.set_seed(model_param["seed"])
        np.random.seed(model_param["seed"])
        act_func = model_param["activation_function"]
        n_units = model_param["n_units"]
        n_inputs = model_param["n_inputs"]
        n_actions = model_param["n_actions"]
        # TODO add number of conv layers to model_params for easier tuning
        # TODO add n_filters, conv_width(kernel_size) to model_params
        n_filters = 6   # Dimensionality of output space
        kernel_size = (3,)  # Convolution width
        n_inputs_static = 7
        n_vehicles = 12  # Defined by DMax default
        n_inputs_dynamic = 4  # lat. and long. vel. and pos. rel to ego vehicle


        # input shape = (batch_size, 12 [n_vehicles], 4 [rel pos and vel])
        self.static_input_layer = layers.Input(shape=(n_inputs_static), name="StaticStateInput")
        self.dynamic_input_layer = layers.Input(shape=tf.TensorShape([n_inputs_dynamic, n_vehicles]), name="DynamicStateInput")

        self.conv_layer_1 = layers.Conv1D(filters=n_filters,
                                          kernel_size=kernel_size,
                                          activation=act_func,
                                          padding='same',
                                          name="ConvolutionalLayer1")(self.dynamic_input_layer)

        self.flatten_layer = layers.Flatten(name="FlattenLayer")(self.conv_layer_1)

        self.concat_layer = layers.Concatenate(name="ConcatenationLayer")([self.flatten_layer, self.static_input_layer])

        self.Q_layer_1 = layers.Dense(n_units[1], activation=act_func, name="QLayer1")(self.concat_layer)
        self.Q_layer_2 = layers.Dense(n_units[2], activation=act_func, name="QLayer2")(self.Q_layer_1)

        self.output_layer = layers.Dense(n_actions)(self.Q_layer_2)

        self.model = keras.Model(inputs=[self.dynamic_input_layer, self.static_input_layer],
                                         outputs=self.output_layer,
                                         name="CNN_DQN")

        self.display_overview()

    @tf.function
    def call(self, inputs: tf.Tensor):
        """ Returns the output of the model given an input. """
        return self.model(inputs)

    def display_overview(self):
        """ Displays an overview of the model. """
        self.model.summary()
        keras.utils.plot_model(self.model,
                               show_shapes=True,
                               show_layer_names=True,
                               to_file='./models/Convolutional_DQN.png')

class DeepSetQNetwork(keras.Model):
    """
    Builds a deep Q-network using DeepSetQ approach incorporating permutation invariance.
    """
    def __init__(self, model_param):
        super(DeepSetQNetwork, self).__init__()
        self.model_param = model_param
        tf.random.set_seed(model_param["seed"])
        np.random.seed(model_param["seed"])
        act_func = model_param["activation_function"]
        # TODO Add n_units for deepset parts of network to the main script to change network size easily
        n_units = model_param["n_units"]
        n_units_phi = [16, 32]
        n_units_rho = [32, 16]
        n_inputs_static = 7
        n_inputs_dynamic = 4
        n_vehicles = 12  # Defined by default D_max size
        n_actions = model_param["n_actions"]

        he = tf.keras.initializers.HeUniform()

        self.static_input_layer = layers.Input(shape=(n_inputs_static), name="StaticStateInput")
        self.dynamic_input_layer = layers.Input(shape=tf.TensorShape([n_vehicles, n_inputs_dynamic]), name="DynamicStateInput")

        self.phi_layer_1 = layers.Dense(n_units_phi[0], activation=act_func, kernel_initializer=he, name="PhiLayer1")(
            self.dynamic_input_layer)
        self.phi_layer_2 = layers.Dense(n_units_phi[1], activation=act_func, kernel_initializer=he, name="PhiLayer2")(
            self.phi_layer_1)

        self.sum_layer = layers.Add(name="Summation_layer")([self.phi_layer_2[:,0,:],
                                       self.phi_layer_2[:,1,:],
                                       self.phi_layer_2[:,2,:],
                                       self.phi_layer_2[:,3,:],
                                       self.phi_layer_2[:,4,:],
                                       self.phi_layer_2[:,5,:],
                                       self.phi_layer_2[:,6,:],
                                       self.phi_layer_2[:,7,:],
                                       self.phi_layer_2[:,8,:],
                                       self.phi_layer_2[:,9,:],
                                       self.phi_layer_2[:,10,:],
                                       self.phi_layer_2[:,11,:]])

        self.rho_layer_1 = layers.Dense(n_units_rho[0], activation=act_func, kernel_initializer=he, name="rhoLayer1")(self.sum_layer)
        self.rho_layer_2 = layers.Dense(n_units_rho[1], activation=act_func, kernel_initializer=he, name="rhoLayer2")(self.rho_layer_1)

        self.concat_layer = layers.Concatenate(name="ConcatenationLayer")([self.rho_layer_2, self.static_input_layer])

        self.Q_layer_1 = layers.Dense(n_units[0], activation=act_func, kernel_initializer=he, name="QLayer1")(self.concat_layer)
        self.Q_layer_2 = layers.Dense(n_units[1], activation=act_func, kernel_initializer=he, name="QLayer2")(self.Q_layer_1)

        self.output_layer = layers.Dense(n_actions)(self.Q_layer_2)

        self.model = keras.Model(inputs=[self.dynamic_input_layer, self.static_input_layer],
                                 outputs=self.output_layer,
                                 name="Deepset_DDQN")

        self.display_overview()

    @tf.function
    def call(self, inputs: tf.Tensor):
        """ Returns the output of the model given an input. """
        return self.model(inputs)

    def display_overview(self):
        """ Displays an overview of the model. """
        self.model.summary()
        keras.utils.plot_model(self.model,
                               show_shapes=True,
                               show_layer_names=True,
                               to_file='./models/Deepset_Q_network.png')



class DeepQNetwork(keras.Model):
    """
    Double Deep Q-network
    """
    def __init__(self, model_param):
        super(DeepQNetwork, self).__init__()
        self.model_param = model_param
        tf.random.set_seed(model_param["seed"])
        np.random.seed(model_param["seed"])
        act_func = model_param["activation_function"]
        n_units = model_param["n_units"]
        n_inputs = model_param["n_inputs"]
        n_actions = model_param["n_actions"]

        he = tf.keras.initializers.HeUniform()
        glorot = tf.keras.initializers.GlorotUniform()
        normal = tf.keras.initializers.RandomUniform(minval=-0.01, maxval=0.01)
        var_scale = tf.keras.initializers.VarianceScaling(scale=2.0, mode='fan_in', distribution='truncated_normal')

        input_layer = layers.Input(shape=(n_inputs,),
                                   name="inputState")

        dense_layer1 = self.dense_layer(num_units=n_units[0],
                                        initialiser=he,
                                        act_func=act_func,
                                        name="dense1")(input_layer)
        dense_layer2 = self.dense_layer(num_units=n_units[1],
                                        initialiser=he,
                                        act_func=act_func,
                                        name="dense2")(dense_layer1)
        dense_layer3 = self.dense_layer(num_units=n_units[2],
                                        initialiser=he,
                                        act_func=act_func,
                                        name="dense3")(dense_layer2)
        output_layer = layers.Dense(n_actions,
                                    name="Output",
                                    kernel_initializer=var_scale,
                                    bias_initializer=tf.keras.initializers.Constant(0))(dense_layer3)

        self.model = keras.Model(inputs=input_layer,
                                 outputs=output_layer,
                                 name="DDQN")

        self.display_overview()

    def dense_layer(self, num_units, act_func, initialiser, name):
        return layers.Dense(
            num_units,
            activation=act_func,
            kernel_initializer=initialiser,
            name=name)

    @tf.function
    def call(self, inputs: tf.Tensor):
        """ Returns the output of the model given an input. """
        return self.model(inputs)

    def display_overview(self):
        """ Displays an overview of the model. """
        self.model.summary()
        keras.utils.plot_model(self.model,
                               show_shapes=True,
                               show_layer_names=True,
                               to_file='./models/DeepQNetwork.png')


class DuellingDqnNetwork(keras.Model):
    """
    Builds the Q-network as a keras model.
    """
    def __init__(self, model_param):
        super(DuellingDqnNetwork, self).__init__()
        self.model_param = model_param
        tf.random.set_seed(model_param["seed"])
        np.random.seed(model_param["seed"])
        act_func = model_param["activation_function"]
        n_units = model_param["n_units"]
        n_inputs = model_param["n_inputs"]
        n_actions = model_param["n_actions"]

        he = tf.keras.initializers.HeUniform()
        glorot = tf.keras.initializers.GlorotUniform()
        normal = tf.keras.initializers.RandomUniform(minval=-0.01, maxval=0.01)
        var_scale = tf.keras.initializers.VarianceScaling(scale=2.0, mode='fan_in', distribution='truncated_normal')

        input_layer = layers.Input(shape=(n_inputs,),
                                   name="inputStateLayer")

        dense_layer1 = self.dense_layer(num_units=n_units[0],
                                        initialiser=he,
                                        act_func=act_func)(input_layer)
        dense_layer2 = self.dense_layer(num_units=n_units[1],
                                        initialiser=he,
                                        act_func=act_func)(dense_layer1)
        dense_layer3 = self.dense_layer(num_units=n_units[2],
                                        initialiser=he,
                                        act_func=act_func)(dense_layer2)

        value_layer, advantage_layer = layers.Lambda(lambda w: tf.split(w, 2, 1))(dense_layer3)

        value_layer = layers.Dense(1)(value_layer)
        advantage_layer = layers.Dense(n_actions)(advantage_layer)

        reduce_mean_layer = layers.Lambda(lambda w: tf.reduce_mean(w, axis=1, keepdims=True))

        output_layer = layers.Add()(
            [value_layer, layers.Subtract()([advantage_layer, reduce_mean_layer(advantage_layer)])])

        self.model = keras.Model(inputs=input_layer,
                                 outputs=output_layer,
                                 name="DuellingDQN")

    def dense_layer(self, num_units, act_func, initialiser):
        return layers.Dense(
            num_units,
            activation=act_func,
            kernel_initializer=initialiser)

    def display_overview(self):
        """ Displays an overview of the model. """
        self.model.summary()
        keras.utils.plot_model(self.model,
                               show_shapes=True,
                               show_layer_names=True,
                               to_file='./models/DuellingDeepQNetwork.png')

    @tf.function
    def call(self, inputs: tf.Tensor):
        """ Returns the output of the model given an input. """
        y = self.model(inputs)
        return y


class AcNetworkSingleAction(keras.Model):
    """
    Neural network architecture for the actor and critic.
    Actor and critic networks are separate.
    Actions feed into critic network.
    Single action output for combinations of steering and velocity commands.
    """
    def __init__(self, model_param):
        super(AcNetworkSingleAction, self).__init__()
        self.model_param = model_param
        tf.random.set_seed(model_param["seed"])
        np.random.seed(model_param["seed"])
        act_func = model_param["activation_function"]
        n_units = model_param["n_units"]
        n_inputs = model_param["n_inputs"]
        n_actions = model_param["n_actions"]

        he = tf.keras.initializers.HeUniform()
        glorot = tf.keras.initializers.GlorotUniform()
        normal = tf.keras.initializers.RandomUniform(minval=-0.01, maxval=0.01)
        var_scale = tf.keras.initializers.VarianceScaling(scale=2.0, mode='fan_in', distribution='truncated_normal')

        input_layer = layers.Input(shape=(n_inputs,),
                                   name="inputStateLayer")

        # Actor net:
        dense_actor_layer1 = self.dense_layer(num_units=n_units[0], init=he, act_func=act_func)(input_layer)
        dense_actor_layer2 = self.dense_layer(num_units=n_units[1], init=he, act_func=act_func)(dense_actor_layer1)
        output_action_layer = layers.Dense(n_actions,
                                           name="OutputActionLayer",
                                           activation=tf.nn.softmax,
                                           kernel_initializer=var_scale,
                                           bias_initializer=tf.keras.initializers.Constant(0))(dense_actor_layer2)

        # Critic net:
        dense_critic_layer1 = self.dense_layer(num_units=n_units[0], init=he, act_func=act_func)(
            layers.concatenate([output_action_layer, input_layer]))
        dense_critic_layer2 = self.dense_layer(num_units=n_units[1], init=he, act_func=act_func)(dense_critic_layer1)
        output_layer_critic = layers.Dense(1, name="OutputLayerCritic")(dense_critic_layer2)

        self.model = keras.Model(inputs=input_layer,
                                 outputs=[output_action_layer, output_layer_critic],
                                 name="ActorCriticNetworkSingleAction")

        self.display_overview()

    def dense_layer(self, num_units, act_func, init):
        return layers.Dense(
            num_units,
            activation=act_func,
            kernel_initializer=init)

    @tf.function
    def call(self, inputs: tf.Tensor):
        """ Returns the output of the model given an input. """
        return self.model(inputs)

    def display_overview(self):
        """ Displays an overview of the model. """
        self.model.summary()
        keras.utils.plot_model(self.model,
                               show_shapes=True,
                               show_layer_names=True,
                               to_file='./models/ActorCriticNetworkSingleAction.png')


class AcNetworkDoubleAction(keras.Model):
    """
    Neural network architecture for the actor and critic.
    Actor and critic networks are separate.
    Actions feed into critic network.
    Actor network has two actions, one for steering and one for velocity.
    """
    def __init__(self, model_param):
        super(AcNetworkDoubleAction, self).__init__()
        self.model_param = model_param
        tf.random.set_seed(model_param["seed"])
        np.random.seed(model_param["seed"])
        act_func = model_param["activation_function"]
        n_units = model_param["n_units"]
        n_inputs = model_param["n_inputs"]
        n_actions = model_param["n_actions"]

        he = tf.keras.initializers.HeUniform()
        glorot = tf.keras.initializers.GlorotUniform()
        normal = tf.keras.initializers.RandomUniform(minval=-0.01, maxval=0.01)
        var_scale = tf.keras.initializers.VarianceScaling(scale=2.0, mode='fan_in', distribution='truncated_normal')

        input_layer = layers.Input(shape=(n_inputs,),
                                   name="inputStateLayer")

        # Actor net:
        dense_actor_layer1 = self.dense_layer(num_units=n_units[0], init=he, act_func=act_func)(input_layer)
        dense_actor_layer2 = self.dense_layer(num_units=n_units[1], init=he, act_func=act_func)(dense_actor_layer1)
        output_velocity_layer = layers.Dense(3,
                                             name="OutputVelocityLayer",
                                             activation=tf.nn.softmax,
                                             kernel_initializer=var_scale,
                                             bias_initializer=tf.keras.initializers.Constant(0))(dense_actor_layer2)
        output_steering_layer = layers.Dense(3,
                                             name="OutputSteeringLayer",
                                             activation=tf.nn.softmax,
                                             kernel_initializer=var_scale,
                                             bias_initializer=tf.keras.initializers.Constant(0))(dense_actor_layer2)

        # Critic net:
        dense_critic_layer1 = self.dense_layer(num_units=n_units[0], init=he, act_func=act_func)(input_layer)
        dense_critic_layer2 = self.dense_layer(num_units=n_units[1], init=he, act_func=act_func)(
            layers.concatenate([output_velocity_layer, output_steering_layer, dense_critic_layer1]))
        output_critic_layer = layers.Dense(1, name="OutputLayerCritic")(dense_critic_layer2)

        self.model = keras.Model(inputs=input_layer,
                                 outputs=[output_velocity_layer, output_steering_layer, output_critic_layer],
                                 name="ActorCriticNetworkDoubleActions")

        self.display_overview()

    def dense_layer(self, num_units, act_func, init):
        return layers.Dense(
            num_units,
            activation=act_func,
            kernel_initializer=init)

    @tf.function
    def call(self, inputs: tf.Tensor):
        """ Returns the output of the model given an input. """
        return self.model(inputs)

    def display_overview(self):
        """ Displays an overview of the model. """
        self.model.summary()
        keras.utils.plot_model(self.model,
                               show_shapes=True,
                               show_layer_names=True,
                               to_file='./models/ActorCriticNetworkDoubleActions.png')



# class ActorCriticVelocity(keras.Model):
#     """
#     Neural network architecture for the actor and critic.
#     Here actions DO feed into critic network.
#     Actor and critic networks are separate.
#     """
#     def __init__(self, model_param):
#         super(ActorCriticVelocity, self).__init__()
#         tf.random.set_seed(model_param["seed"])
#         np.random.seed(model_param["seed"])
#         act_func = model_param["activation_function"]
#         n_units = model_param["n_units"]
#         n_inputs = model_param["n_inputs"]
#         n_actions = model_param["n_actions"]
#
#         self.he = tf.keras.initializers.HeNormal()
#         glorot = tf.keras.initializers.GlorotNormal()
#         normal = tf.keras.initializers.RandomUniform(minval=-0.03, maxval=0.03)
#         var_scale = tf.keras.initializers.VarianceScaling(scale=2.0, mode='fan_in', distribution='truncated_normal')
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
#                                         kernel_initializer=glorot,
#                                         bias_initializer=tf.keras.initializers.Constant(0))(dense_actor_layer2)
#
#         # Critic net:
#         dense_critic_layer1 = self.dense_layer(num_units=n_units[0], act_func=act_func)(
#             layers.concatenate([output_layer_vel, input_layer]))
#         dense_critic_layer2 = self.dense_layer(num_units=n_units[1], act_func=act_func)(dense_critic_layer1)
#         output_layer_critic_vel = layers.Dense(1, name="OutputLayerCritic")(dense_critic_layer2)
#
#         self.model = keras.Model(inputs=input_layer,
#                                  outputs=[output_layer_vel, output_layer_critic_vel],
#                                  name="AC-Velocity")
#
#     def dense_layer(self, num_units, act_func):
#         return layers.Dense(
#             num_units,
#             activation=act_func,
#             kernel_initializer=self.he)
#
#     @tf.function
#     def call(self, inputs: tf.Tensor):
#         """ Returns the output of the model given an input. """
#         return self.model(inputs)
#
#     def display_overview(self):
#         """ Displays an overview of the model. """
#         self.model.summary()
#         keras.utils.plot_model(self.model, show_shapes=True, show_layer_names=True)
#
#
# class ActorCriticSteering(keras.Model):
#     """
#     Neural network architecture for the actor and critic.
#     Here actions DO feed into critic network.
#     Actor and critic networks are separate.
#     """
#
#     def __init__(self, modelParam):
#         super(ActorCriticSteering, self).__init__()
#         tf.random.set_seed(modelParam["seed"])
#         np.random.seed(modelParam["seed"])
#         act_func = modelParam["activation_function"]
#         n_units = modelParam["n_units"]
#         n_inputs = modelParam["n_inputs"]
#         n_actions = modelParam["n_actions"]
#
#         self.he = tf.keras.initializers.HeNormal()
#         glorot = tf.keras.initializers.GlorotNormal()
#         normal = tf.keras.initializers.RandomUniform(minval=-0.03, maxval=0.03)
#         var_scale = tf.keras.initializers.VarianceScaling(scale=2.0, mode='fan_in', distribution='truncated_normal')
#
#         input_layer = layers.Input(shape=(n_inputs,),
#                                    name="inputStateLayer")
#
#         # Actor net for steering:
#         dense_actor_layer1 = self.dense_layer(num_units=n_units[0], act_func=act_func)(input_layer)
#         dense_actor_layer2 = self.dense_layer(num_units=n_units[1], act_func=act_func)(dense_actor_layer1)
#
#         output_layer_steer = layers.Dense(3,
#                                           name="OutputLayerSteering",
#                                           activation=tf.nn.softmax,
#                                           kernel_initializer=glorot,
#                                           bias_initializer=tf.keras.initializers.Constant(0))(dense_actor_layer2)
#
#         # Critic net:
#         dense_critic_layer1 = self.dense_layer(num_units=n_units[0], act_func=act_func)(
#             layers.concatenate([output_layer_steer, input_layer]))
#         dense_critic_layer2 = self.dense_layer(num_units=n_units[1], act_func=act_func)(dense_critic_layer1)
#         output_layer_critic_steer = layers.Dense(1, name="OutputLayerCritic")(dense_critic_layer2)
#
#         self.model = keras.Model(inputs=input_layer,
#                                  outputs=[output_layer_steer, output_layer_critic_steer],
#                                  name="AC-Steering")
#
#     def dense_layer(self, num_units, act_func):
#         return layers.Dense(
#             num_units,
#             activation=act_func,
#             kernel_initializer=self.he)
#
#     @tf.function
#     def call(self, inputs: tf.Tensor):
#         """ Returns the output of the model given an input. """
#         return self.model(inputs)
#
#     def display_overview(self):
#         """ Displays an overview of the model. """
#         self.model.summary()
#         keras.utils.plot_model(self.model, show_shapes=True, show_layer_names=True)
