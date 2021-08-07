import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Sequential
import numpy as np
from HelperClasses import EpisodeBuffer, DataLogger


class LSTM_DRQN(keras.Model):
    """
    Builds a LSTM, recurrent model to learn temporal relationships between states.
    """
    # TODO batch norm layer?
    # TODO parameters for tuning the lstm network easily
    def __init__(self, model_param):
        super(LSTM_DRQN, self).__init__()
        self.model_param = model_param
        tf.random.set_seed(model_param["seed"])
        np.random.seed(model_param["seed"])
        act_func = model_param["activation_function"]
        n_units = model_param["n_units"]
        n_inputs = model_param["n_inputs"]  # Number of items in state matrix
        n_actions = model_param["n_actions"]
        n_timesteps = 4  # Number of stacked measurements considered
        # TODO LSTM with attention mechanism vs lstm without attention mechanism !!!
        # TODO LSTM without the need for frame stacking? I dont think so?
        # https://levelup.gitconnected.com/building-seq2seq-lstm-with-luong-attention-in-keras-for-time-series-forecasting-1ee00958decb
        input_layer = layers.Input(shape=(n_timesteps, n_inputs,), name="inputState")
        LSTM_layer = layers.LSTM(units=n_units[0], name="LSTM")(input_layer)
        dense_layer1 = layers.Dense(units=n_units[1], activation=act_func, name="dense1")(LSTM_layer)
        dense_layer2 = layers.Dense(units=n_units[2], activation=act_func, name="dense3")(dense_layer1)
        output_layer = layers.Dense(n_actions, name="Output")(dense_layer2)
        self.model = keras.Model(inputs=input_layer,
                                 outputs=output_layer,
                                 trainable=self.model_param["trainable"],
                                 name="LSTM_DRQN")
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
                               to_file='./models/LSTM_DRQN.png')


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

        # TODO mean pooling layers + parameters for tuning easily
        # TODO add number of conv layers to model_params for easier tuning
        # TODO add n_filters, conv_width(kernel_size) to model_params

        self.cnn_config = model_param["cnn_param"]["config"]
        n_inputs_static = 7
        n_vehicles = 12  # Defined by DMax default
        n_inputs_dynamic = 4  # lat. and long. vel. and pos. rel to ego vehicle

        self.static_input_layer = layers.Input(shape=(n_inputs_static), name="StaticStateInput")
        # TODO Check 2DConvolutions For comparisons!
        # TODO investigate multiple CNN layers and parameters
        if self.cnn_config == 0:     # 0=1D conv. on vehicle dim.,
            n_filters = model_param["cnn_param"]["n_filters_0"]  # Dimensionality of output space
            kernel_size = model_param["cnn_param"]["kernel_size_0"]  # Convolution width
            input_matrix = tf.TensorShape([n_inputs_dynamic, n_vehicles])

            # input shape = (batch_size, 4 [rel pos and vel], 12 [n_vehicles])
            self.dynamic_input_layer = layers.Input(shape=input_matrix, name="DynamicStateInput")
            self.conv_layer_1 = layers.Conv1D(filters=n_filters,
                                              kernel_size=kernel_size,
                                              activation=act_func,
                                              padding='same',
                                              name="ConvolutionalLayer1")(self.dynamic_input_layer)
            # Output_shape = (batch_size, 4 [rel pos and vel], 6 n_filters)

        elif self.cnn_config == 1:       # 1=1D conv. on measurements dim.,
            n_filters = model_param["cnn_param"]["n_filters_1"]  # Dimensionality of output space
            kernel_size = model_param["cnn_param"]["kernel_size_1"]  # Convolution width
            input_matrix = tf.TensorShape([n_vehicles, n_inputs_dynamic])
            # input shape = (batch_size, 12 [n_vehicles], 4 [rel pos and vel])
            self.dynamic_input_layer = layers.Input(shape=input_matrix, name="DynamicStateInput")
            self.conv_layer_1 = layers.Conv1D(filters=n_filters,
                                              kernel_size=kernel_size,
                                              activation=act_func,
                                              padding='same',
                                              name="ConvolutionalLayer1")(self.dynamic_input_layer)
            # Output_shape = (batch_size, 12 n_vehicles, n_filters)

        elif self.cnn_config == 2:       # 2=2D conv. on vehicle and measurements dimensions,
            n_filters = model_param["cnn_param"]["n_filters_2"]  # Dimensionality of output space
            kernel_size = model_param["cnn_param"]["kernel_size_2"]  # Convolution width
            n_timesteps = 1  # Number of stacked measurements considered
            input_matrix = tf.TensorShape([n_vehicles, n_inputs_dynamic, n_timesteps])
            # input shape = (batch_size, 12 [n_vehicles], 4 [rel pos and vel])
            self.dynamic_input_layer = layers.Input(shape=input_matrix, name="DynamicStateInput")
            self.conv_layer_1 = layers.Conv2D(filters=n_filters,
                                              kernel_size=kernel_size,
                                              activation=act_func,
                                              padding='same',
                                              name="ConvolutionalLayer1")(self.dynamic_input_layer)
            # Output shape = (batch_size, n_vehicles, n_measurements, n_filters)
            # TODO consider pooling functionality

        elif self.cnn_config == 3:       # 3=3D conv. on vehicle and measurement dimensions through time
            n_filters = model_param["cnn_param"]["n_filters_3"]  # Dimensionality of output space
            kernel_size = model_param["cnn_param"]["kernel_size_3"]  # Convolution width
            n_timesteps = 4  # Number of stacked measurements considered
            input_matrix = tf.TensorShape([n_timesteps,n_vehicles, n_inputs_dynamic, 1])
            # input shape = (batch_size, 12 [n_vehicles], 4 [rel pos and vel], 4 [t_0, t_1, ..])
            self.dynamic_input_layer = layers.Input(shape=input_matrix, name="DynamicStateInput")
            self.conv_layer_1 = layers.Conv3D(filters=n_filters,
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
                                 trainable=self.model_param["trainable"],
                                 name="CNN_DQN")

        self.display_overview()

    @tf.function
    def call(self, inputs: tf.Tensor):
        if self.cnn_config == 0:
            dynamic_input = inputs[0]
            static_input = inputs[1]
            batch_size, x1, x2 = dynamic_input.shape
            # TODO check this reshape, maybe transpose would do !
            dynamic_input = tf.reshape(dynamic_input, [batch_size, x2, x1])
            inputs = (dynamic_input, static_input)
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
        act_func_phi = model_param['deepset_param']['act_func_phi']
        act_func_rho = model_param['deepset_param']['act_func_rho']
        n_units = model_param["n_units"]
        n_units_phi = model_param["deepset_param"]["n_units_phi"]
        n_units_rho = model_param["deepset_param"]["n_units_rho"]
        n_inputs_static = 7
        n_inputs_dynamic = 4
        n_vehicles = 12  # Defined by default D_max size
        n_actions = model_param["n_actions"]

        self.static_input_layer = layers.Input(shape=(n_inputs_static), name="StaticStateInput")
        self.dynamic_input_layer = layers.Input(shape=tf.TensorShape([n_vehicles, n_inputs_dynamic]), name="DynamicStateInput")

        if len(n_units_phi) == 3:
            self.phi_layer_1 = layers.Dense(n_units_phi[0], activation=act_func_phi, name="PhiLayer1")(
                self.dynamic_input_layer)
            self.phi_layer_2 = layers.Dense(n_units_phi[1], activation=act_func_phi, name="PhiLayer2")(
                self.phi_layer_1)
            self.phi_layer_3 = layers.Dense(n_units_phi[2], activation=act_func_phi, name="PhiLayer3")(
                self.phi_layer_2)
            self.layer_list =[self.phi_layer_3[:, 0, :],
                              self.phi_layer_3[:, 1, :],
                              self.phi_layer_3[:, 2, :],
                              self.phi_layer_3[:, 3, :],
                              self.phi_layer_3[:, 4, :],
                              self.phi_layer_3[:, 5, :],
                              self.phi_layer_3[:, 6, :],
                              self.phi_layer_3[:, 7, :],
                              self.phi_layer_3[:, 8, :],
                              self.phi_layer_3[:, 9, :],
                              self.phi_layer_3[:, 10, :],
                              self.phi_layer_3[:, 11, :]]

            self.sum_layer = layers.Add(name="Summation_layer")(self.layer_list)
        else:
            self.phi_layer_1 = layers.Dense(n_units_phi[0], activation=act_func_phi, name="PhiLayer1")(
                self.dynamic_input_layer)
            self.phi_layer_2 = layers.Dense(n_units_phi[1], activation=act_func_phi, name="PhiLayer2")(
                self.phi_layer_1)
            self.layer_list = [self.phi_layer_2[:, 0, :],
                          self.phi_layer_2[:, 1, :],
                          self.phi_layer_2[:, 2, :],
                          self.phi_layer_2[:, 3, :],
                          self.phi_layer_2[:, 4, :],
                          self.phi_layer_2[:, 5, :],
                          self.phi_layer_2[:, 6, :],
                          self.phi_layer_2[:, 7, :],
                          self.phi_layer_2[:, 8, :],
                          self.phi_layer_2[:, 9, :],
                          self.phi_layer_2[:, 10, :],
                          self.phi_layer_2[:, 11, :]]
            self.sum_layer = layers.Add(name="Summation_layer")(self.layer_list)

        if len(n_units_rho) == 3:
            self.rho_layer_1 = layers.Dense(n_units_rho[0], activation=act_func_rho, name="rhoLayer1")(self.sum_layer)
            self.rho_layer_2 = layers.Dense(n_units_rho[1], activation=act_func_rho, name="rhoLayer2")(self.rho_layer_1)
            self.rho_layer_3 = layers.Dense(n_units_rho[2], activation=act_func_rho, name="rhoLayer3")(self.rho_layer_2)
            if model_param["batch_normalisation"] == True:
                self.batch_norm_layer = layers.BatchNormalization(name="batch_norm")(self.rho_layer_3)
                self.concat_layer = layers.Concatenate(name="ConcatenationLayer")(
                    [self.batch_norm_layer, self.static_input_layer])
            else:
                self.concat_layer = layers.Concatenate(name="ConcatenationLayer")(
                    [self.rho_layer_3, self.static_input_layer])
        else:
            self.rho_layer_1 = layers.Dense(n_units_rho[0], activation=act_func_rho, name="rhoLayer1")(self.sum_layer)
            self.rho_layer_2 = layers.Dense(n_units_rho[1], activation=act_func_rho, name="rhoLayer2")(self.rho_layer_1)
            if model_param["batch_normalisation"]==True:
                self.batch_norm_layer = layers.BatchNormalization(name="batch_norm")(self.rho_layer_2)
                self.concat_layer = layers.Concatenate(name="ConcatenationLayer")(
                    [self.batch_norm_layer, self.static_input_layer])
            else:
                self.concat_layer = layers.Concatenate(name="ConcatenationLayer")([self.rho_layer_2, self.static_input_layer])

        self.Q_layer_1 = layers.Dense(n_units[0], activation=act_func, name="QLayer1")(self.concat_layer)
        self.Q_layer_2 = layers.Dense(n_units[1], activation=act_func, name="QLayer2")(self.Q_layer_1)

        self.output_layer = layers.Dense(n_actions)(self.Q_layer_2)

        self.model = keras.Model(inputs=[self.dynamic_input_layer, self.static_input_layer],
                                 trainable=self.model_param["trainable"],
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

        input_layer = layers.Input(shape=(n_inputs,),
                                   name="inputState")
        if len(n_units) == 2:
            dense_layer1 = layers.Dense(units=n_units[0],
                                        activation=act_func,
                                        name='dense1')(input_layer)
            dense_layer2 = layers.Dense(units=n_units[1],
                                        activation=act_func,
                                        name='dense2')(dense_layer1)
            if model_param["batch_normalisation"] == True:
                normalisation_layer = layers.BatchNormalization()(dense_layer2)
                output_layer = layers.Dense(n_actions,
                                            name="Output")(normalisation_layer)
            else:
                output_layer = layers.Dense(n_actions,
                                            name="Output")(dense_layer2)
        else:
            dense_layer1 = layers.Dense(units=n_units[0],
                                        activation=act_func,
                                        name='dense1')(input_layer)
            dense_layer2 = layers.Dense(units=n_units[1],
                                        activation=act_func,
                                        name='dense2')(dense_layer1)
            dense_layer3 = layers.Dense(units=n_units[2],
                                        activation=act_func,
                                        name='dense3')(dense_layer2)
            if model_param["batch_normalisation"] == True:
                normalisation_layer = layers.BatchNormalization(name="batch_norm")(dense_layer3)
                output_layer = layers.Dense(n_actions,
                                            name="Output")(normalisation_layer)
            else:
                output_layer = layers.Dense(n_actions,
                                            name="Output")(dense_layer3)

        self.model = keras.Model(inputs=input_layer,
                                 outputs=output_layer,
                                 trainable=self.model_param["trainable"],
                                 name="DDQN")
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


        input_layer = layers.Input(shape=(n_inputs,),
                                   name="inputStateLayer")

        if len(n_units) == 2:
            dense_layer1 = layers.Dense(n_units[0], activation=act_func)(input_layer)
            dense_layer2 = layers.Dense(n_units[1], activation=act_func)(dense_layer1)
            value_layer, advantage_layer = layers.Lambda(lambda w: tf.split(w, 2, 1))(dense_layer2)
        else:
            dense_layer1 = layers.Dense(n_units[0], activation=act_func)(input_layer)
            dense_layer2 = layers.Dense(n_units[1], activation=act_func)(dense_layer1)
            dense_layer3 = layers.Dense(n_units[2], activation=act_func)(dense_layer2)
            value_layer, advantage_layer = layers.Lambda(lambda w: tf.split(w, 2, 1))(dense_layer3)

        value_layer = layers.Dense(1)(value_layer)
        advantage_layer = layers.Dense(n_actions)(advantage_layer)

        reduce_mean_layer = layers.Lambda(lambda w: tf.reduce_mean(w, axis=1, keepdims=True))

        output_layer = layers.Add()(
            [value_layer, layers.Subtract()([advantage_layer, reduce_mean_layer(advantage_layer)])])

        self.model = keras.Model(inputs=input_layer,
                                 outputs=output_layer,
                                 trainable=self.model_param["trainable"],
                                 name="DuellingDQN")


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


# class AcNetworkSingleAction(keras.Model):
#     """
#     Neural network architecture for the actor and critic.
#     Actor and critic networks are separate.
#     Actions feed into critic network.
#     Single action output for combinations of steering and velocity commands.
#     """
#     def __init__(self, model_param):
#         super(AcNetworkSingleAction, self).__init__()
#         self.model_param = model_param
#         tf.random.set_seed(model_param["seed"])
#         np.random.seed(model_param["seed"])
#         act_func = model_param["activation_function"]
#         n_units = model_param["n_units"]
#         n_inputs = model_param["n_inputs"]
#         n_actions = model_param["n_actions"]
#
#         he = tf.keras.initializers.HeUniform()
#         glorot = tf.keras.initializers.GlorotUniform()
#         normal = tf.keras.initializers.RandomUniform(minval=-0.01, maxval=0.01)
#         var_scale = tf.keras.initializers.VarianceScaling(scale=2.0, mode='fan_in', distribution='truncated_normal')
#
#         input_layer = layers.Input(shape=(n_inputs,),
#                                    name="inputStateLayer")
#
#         # Actor net:
#         dense_actor_layer1 = self.dense_layer(num_units=n_units[0], init=he, act_func=act_func)(input_layer)
#         dense_actor_layer2 = self.dense_layer(num_units=n_units[1], init=he, act_func=act_func)(dense_actor_layer1)
#         output_action_layer = layers.Dense(n_actions,
#                                            name="OutputActionLayer",
#                                            activation=tf.nn.softmax,
#                                            kernel_initializer=var_scale,
#                                            bias_initializer=tf.keras.initializers.Constant(0))(dense_actor_layer2)
#
#         # Critic net:
#         dense_critic_layer1 = self.dense_layer(num_units=n_units[0], init=he, act_func=act_func)(
#             layers.concatenate([output_action_layer, input_layer]))
#         dense_critic_layer2 = self.dense_layer(num_units=n_units[1], init=he, act_func=act_func)(dense_critic_layer1)
#         output_layer_critic = layers.Dense(1, name="OutputLayerCritic")(dense_critic_layer2)
#
#         self.model = keras.Model(inputs=input_layer,
#                                  outputs=[output_action_layer, output_layer_critic],
#                                  name="ActorCriticNetworkSingleAction")
#
#         self.display_overview()
#
#     def dense_layer(self, num_units, act_func, init):
#         return layers.Dense(
#             num_units,
#             activation=act_func,
#             kernel_initializer=init)
#
#     @tf.function
#     def call(self, inputs: tf.Tensor):
#         """ Returns the output of the model given an input. """
#         return self.model(inputs)
#
#     def display_overview(self):
#         """ Displays an overview of the model. """
#         self.model.summary()
#         keras.utils.plot_model(self.model,
#                                show_shapes=True,
#                                show_layer_names=True,
#                                to_file='./models/ActorCriticNetworkSingleAction.png')
#
#
# class AcNetworkDoubleAction(keras.Model):
#     """
#     Neural network architecture for the actor and critic.
#     Actor and critic networks are separate.
#     Actions feed into critic network.
#     Actor network has two actions, one for steering and one for velocity.
#     """
#     def __init__(self, model_param):
#         super(AcNetworkDoubleAction, self).__init__()
#         self.model_param = model_param
#         tf.random.set_seed(model_param["seed"])
#         np.random.seed(model_param["seed"])
#         act_func = model_param["activation_function"]
#         n_units = model_param["n_units"]
#         n_inputs = model_param["n_inputs"]
#         n_actions = model_param["n_actions"]
#
#         he = tf.keras.initializers.HeUniform()
#         glorot = tf.keras.initializers.GlorotUniform()
#         normal = tf.keras.initializers.RandomUniform(minval=-0.01, maxval=0.01)
#         var_scale = tf.keras.initializers.VarianceScaling(scale=2.0, mode='fan_in', distribution='truncated_normal')
#
#         input_layer = layers.Input(shape=(n_inputs,),
#                                    name="inputStateLayer")
#
#         # Actor net:
#         dense_actor_layer1 = self.dense_layer(num_units=n_units[0], init=he, act_func=act_func)(input_layer)
#         dense_actor_layer2 = self.dense_layer(num_units=n_units[1], init=he, act_func=act_func)(dense_actor_layer1)
#         output_velocity_layer = layers.Dense(3,
#                                              name="OutputVelocityLayer",
#                                              activation=tf.nn.softmax,
#                                              kernel_initializer=var_scale,
#                                              bias_initializer=tf.keras.initializers.Constant(0))(dense_actor_layer2)
#         output_steering_layer = layers.Dense(3,
#                                              name="OutputSteeringLayer",
#                                              activation=tf.nn.softmax,
#                                              kernel_initializer=var_scale,
#                                              bias_initializer=tf.keras.initializers.Constant(0))(dense_actor_layer2)
#
#         # Critic net:
#         dense_critic_layer1 = self.dense_layer(num_units=n_units[0], init=he, act_func=act_func)(input_layer)
#         dense_critic_layer2 = self.dense_layer(num_units=n_units[1], init=he, act_func=act_func)(
#             layers.concatenate([output_velocity_layer, output_steering_layer, dense_critic_layer1]))
#         output_critic_layer = layers.Dense(1, name="OutputLayerCritic")(dense_critic_layer2)
#
#         self.model = keras.Model(inputs=input_layer,
#                                  outputs=[output_velocity_layer, output_steering_layer, output_critic_layer],
#                                  name="ActorCriticNetworkDoubleActions")
#
#         self.display_overview()
#
#     def dense_layer(self, num_units, act_func, init):
#         return layers.Dense(
#             num_units,
#             activation=act_func,
#             kernel_initializer=init)
#
#     @tf.function
#     def call(self, inputs: tf.Tensor):
#         """ Returns the output of the model given an input. """
#         return self.model(inputs)
#
#     def display_overview(self):
#         """ Displays an overview of the model. """
#         self.model.summary()
#         keras.utils.plot_model(self.model,
#                                show_shapes=True,
#                                show_layer_names=True,
#                                to_file='./models/ActorCriticNetworkDoubleActions.png')



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
