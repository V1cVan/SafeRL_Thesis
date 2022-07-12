import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Sequential
import numpy as np
from HelperClasses import EpisodeBuffer, DataLogger
import sys

class LSTM(keras.Model):
    """
    Builds a LSTM, recurrent model to learn temporal relationships between states.
    """
    # TODO batch norm layer?
    # TODO parameters for tuning the lstm network easily
    def __init__(self, model_param):
        super(LSTM, self).__init__()
        self.model_param = model_param
        tf.random.set_seed(model_param["seed"])
        np.random.seed(model_param["seed"])
        act_func_q = model_param["activation_function"]
        n_units_q = model_param["n_units"]
        n_units_lstm = model_param["LSTM_param"]["n_units"]
        n_inputs = model_param["n_inputs"]  # Number of items in state matrix
        n_actions = model_param["n_actions"]
        n_timesteps = 4  # Number of stacked measurements considered

        input_layer = layers.Input(shape=(n_timesteps,n_inputs), name="inputState")
        # Many-to-one LSTM layer

        LSTM_layer = layers.LSTM(units=n_units_lstm,
                                 name="LSTM")(input_layer)

        dense_layer1 = layers.Dense(units=n_units_q[0], activation=act_func_q, name="dense1")(LSTM_layer)
        # dense_layer2 = layers.Dense(units=n_units_q[1], activation=act_func_q, name="dense2")(dense_layer1)

        output_layer = layers.Dense(n_actions, name="Output")(dense_layer1)

        model_name = "LSTM"
        self.model = keras.Model(inputs=input_layer,
                                 outputs=output_layer,
                                 trainable=self.model_param["trainable"],
                                 name=model_name)
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


class TemporalCNN(keras.Model):
    """
    Builds a temporal CNN of a 2D or 3D shape.
    """
    def __init__(self, model_param):
        super(TemporalCNN, self).__init__()
        self.model_param = model_param
        tf.random.set_seed(model_param['seed'])
        np.random.seed(model_param['seed'])
        remove_state_velocity = model_param["remove_velocity"]
        if remove_state_velocity:
            n_dynamic = 2  # Number of dynamic measurement elements
        else:
            n_dynamic = 4  # Number of dynamic measurement elements
        # CNN parameters:
        cnn_type = model_param['cnn_param']['temporal_CNN_type']  # 2D or 3D
        kernel = model_param['cnn_param']['kernel']
        filters = model_param['cnn_param']['filters']
        strides = model_param['cnn_param']['strides']
        n_layers = len(filters)

        # Q-network parameters:
        act_func = model_param['activation_function']
        n_units = model_param['n_units']
        n_actions = model_param['n_actions']
        n_vehicles = 12

        n_static = 7  # Number of static measurement elements
        n_measurements = 4

        # Dynamic vector input shape = (n_measurements x n_vehicles x n_dynamic)
        # Static vector input shape = (n_measurements x n_static)
        self.dynamic_input = layers.Input(shape=(n_vehicles, n_dynamic, n_measurements), name="dynamic_input_layer")
        self.static_input = layers.Input(shape=(n_static), name="static_input_layer")

        if cnn_type == '2D':
            self.conv_layer_1 = layers.Conv2D(filters=filters[0],
                                              kernel_size=kernel[0],
                                              activation=act_func,
                                              data_format="channels_last",
                                              padding='same',
                                              strides=strides[0],
                                              name="conv_layer_1")(self.dynamic_input)
            if n_layers == 1:
                self.flatten_layer = layers.Flatten(name="flatten_layer")(self.conv_layer_1)
            else:
                self.conv_layer_2 = layers.Conv2D(filters=filters[1],
                                                  kernel_size=kernel[1],
                                                  activation=act_func,
                                                  strides=strides[1],
                                                  data_format="channels_last",
                                                  name="conv_layer_2")(self.conv_layer_1)
                pooling = layers.MaxPooling2D(pool_size=(2, 1))(self.conv_layer_2)
                self.flatten_layer = layers.Flatten(name="flatten_layer")(pooling)

        else:  # cnn_type == '3D'
            self.conv_layer_1 = layers.Conv3D(filters=filters[0],
                                              kernel_size=kernel[0],
                                              activation=act_func,
                                              strides=strides[0],
                                              data_format="channels_last",
                                              name="conv_layer_1")(self.dynamic_input)
            if n_layers == 1:
                self.flatten_layer = layers.Flatten(name="flatten_layer")(self.conv_layer_1)
            else:
                self.conv_layer_2 = layers.Conv3D(filters=filters[1],
                                                  kernel_size=kernel[1],
                                                  activation=act_func,
                                                  strides=strides[1],
                                                  data_format="channels_last",
                                                  name="conv_layer_2")(self.conv_layer_1)
                pooling = layers.MaxPooling2D(pool_size=(2, 1))(self.conv_layer_2)
                self.flatten_layer = layers.Flatten(name="flatten_layer")(pooling)

        self.concat_layer = layers.Concatenate(name="ConcatenationLayer")(
            [self.flatten_layer, self.static_input])

        self.Q_layer_1 = layers.Dense(n_units[0], activation=act_func, name="Q_layer_1")(self.concat_layer)
        self.Q_layer_2 = layers.Dense(n_units[1], activation=act_func, name="Q_layer_2")(self.Q_layer_1)

        self.output_layer = layers.Dense(n_actions)(self.Q_layer_2)

        model_name = "Temporal_CNN"+'_'+cnn_type
        self.model = keras.Model(inputs=[self.dynamic_input, self.static_input],
                                 outputs=self.output_layer,
                                 trainable=self.model_param["trainable"],
                                 name=model_name)
        self.display_overview(model_name)

    @tf.function
    def call(self, inputs: tf.Tensor):
        """ Returns the output of the model given an input. """
        dynamic_input = tf.transpose(inputs[0], perm=[0,2,3,1])
        static_input = inputs[1]
        return self.model([dynamic_input, static_input])

    def display_overview(self, model_name):
        """ Displays an overview of the model. """
        path = './models/' + model_name + '.png'
        self.model.summary()
        keras.utils.plot_model(self.model,
                               show_shapes=True,
                               show_layer_names=True,
                               to_file=path)


class CNN(keras.Model):
    """
    Builds a convolutional neural network to handle the dynamic part of the state vector.
    """
    def __init__(self, model_param):
        super(CNN, self).__init__()
        self.model_param = model_param
        tf.random.set_seed(model_param['seed'])
        np.random.seed(model_param['seed'])

        # CNN parameters:
        self.cnn_type = model_param['cnn_param']['normal_CNN_type']
        # 0=1D conv. on vehicle dim.,
        # 1=1D conv. on measurements dim.,
        # 2=2D conv. on vehicle and measurements dimensions.
        kernel = model_param['cnn_param']['kernel']
        filters = model_param['cnn_param']['filters']
        strides = model_param['cnn_param']['strides']
        use_pooling = model_param['cnn_param']['use_pooling']
        pool_size = model_param["cnn_param"]['pool_size']
        self.n_layers = len(filters)

        # Q-network parameters:
        act_func = model_param['activation_function']
        n_units = model_param['n_units']
        n_actions = model_param['n_actions']
        n_vehicles = 12
        n_dynamic = 4  # Number of dynamic measurement elements
        n_static = 7  # Number of static measurement elements
        n_measurements = 1


        # Static vector input shape = (n_measurements x n_static)
        self.static_input = layers.Input(shape=(n_static), name="static_input_layer")
        # input_matrix = tf.TensorShape([n_inputs_dynamic, n_vehicles])

        if self.cnn_type == 0:  # 0=1D conv. on vehicle dim.,
            # Dynamic vector input shape (type0) = (n_dynamic x n_vehicles)
            self.dynamic_input = layers.Input(shape=(n_dynamic, n_vehicles), name="dynamic_input_layer")
            self.conv_layer_1 = layers.Conv1D(filters=filters[0],
                                              kernel_size=kernel[0],
                                              activation=act_func,
                                              strides=(1,),
                                              padding="same",
                                              name="conv_layer_1")(self.dynamic_input)
            # Output_shape = (batch_size, 4 [rel pos and vel], n_filters) (with padding = 'same')
            if self.n_layers == 1:
                self.flatten_layer = layers.Flatten(name="FlattenLayer")(self.conv_layer_1)
            elif self.n_layers == 2:
                self.conv_layer_2 = layers.Conv1D(filters=filters[1],
                                                  kernel_size=kernel[1],
                                                  activation=act_func,
                                                  strides=(1,),
                                                  padding="same",
                                                  name="conv_layer_2")(self.conv_layer_1)
                self.flatten_layer = layers.Flatten(name="FlattenLayer")(self.conv_layer_2)
        elif self.cnn_type == 1:  # 1=1D conv. on measurements dim.,
            # Dynamic vector input shape (type1) = ( n_vehicles x n_dynamic )
            self.dynamic_input = layers.Input(shape=(n_vehicles, n_dynamic), name="dynamic_input_layer")
            # input shape = (batch_size, 12 [n_vehicles], 4 [rel pos and vel])
            self.conv_layer_1 = layers.Conv1D(filters=filters[0],
                                              kernel_size=kernel[0],
                                              activation=act_func,
                                              strides=(1,),
                                              padding="same",
                                              name="conv_layer_1")(self.dynamic_input)
            # Output_shape = (batch_size, 12 n_vehicles, n_filters) (with padding = 'same')
            if self.n_layers == 1:
                if use_pooling:
                    self.pooling_layer = layers.Lambda(lambda input: tf.reduce_sum(input, axis=1), name="SumPoolLayer")(
                        self.conv_layer_1)
                    self.flatten_layer = layers.Flatten(name="FlattenLayer")(self.pooling_layer)
                else:
                    self.flatten_layer = layers.Flatten(name="FlattenLayer")(self.conv_layer_1)
            elif self.n_layers == 2:
                self.conv_layer_2 = layers.Conv1D(filters=filters[1],
                                                  kernel_size=kernel[1],
                                                  activation=act_func,
                                                  strides=(1,),
                                                  padding="same",
                                                  name="conv_layer_2")(self.conv_layer_1)
                if use_pooling:
                    self.pooling_layer = layers.Lambda(lambda input: tf.reduce_sum(input, axis=1), name="SumPoolLayer")(
                        self.conv_layer_2)
                    self.flatten_layer = layers.Flatten(name="FlattenLayer")(self.pooling_layer)
                else:
                    self.flatten_layer = layers.Flatten(name="FlattenLayer")(self.conv_layer_2)
            else:
                print("Number of layers in CNN cannot be anything other than 1 or 2")
                sys.exit()


        elif self.cnn_type == 2:  # 2=2D conv. on vehicle and measurements dimensions,
            # Dynamic vector input shape (type2) = (n_measurements(1) x n_vehicles x n_dynamic)
            self.dynamic_input = layers.Input(shape=(n_vehicles, n_dynamic, 1), name="dynamic_input_layer")
            # input shape = (batch_size, 12 [n_vehicles], 4 [rel pos and vel])
            self.conv_layer_1 = layers.Conv2D(filters=filters[0],
                                              kernel_size=kernel[0],
                                              activation=act_func,
                                              padding="same",
                                              strides=strides[0],
                                              name="conv_layer_1")(self.dynamic_input)
            self.conv_layer_2 = layers.Conv2D(filters=filters[1],
                                              kernel_size=kernel[1],
                                              padding="same",
                                              activation=act_func,
                                              strides=strides[1],
                                              name="conv_layer_2")(self.conv_layer_1)
            if use_pooling:
                self.pooling_layer = layers.AvgPool2D(pool_size=pool_size,
                                                      name="avg_pool",
                                                      padding="same")(self.conv_layer_2)
                self.flatten_layer = layers.Flatten(name="FlattenLayer")(self.pooling_layer)
            else:
                self.flatten_layer = layers.Flatten(name="FlattenLayer")(self.conv_layer_2)

        self.concat_layer = layers.Concatenate(name="ConcatenationLayer")([self.flatten_layer, self.static_input])

        self.Q_layer_1 = layers.Dense(n_units[0], activation=act_func, name="QLayer1")(self.concat_layer)
        self.Q_layer_2 = layers.Dense(n_units[1], activation=act_func, name="QLayer2")(self.Q_layer_1)

        self.output_layer = layers.Dense(n_actions)(self.Q_layer_2)

        model_name = "Normal_CNN" + '_' + str(self.cnn_type)
        self.model = keras.Model(inputs=[self.dynamic_input, self.static_input],
                                 outputs=self.output_layer,
                                 trainable=self.model_param["trainable"],
                                 name=model_name)

        self.display_overview(model_name)

    @tf.function
    def call(self, inputs: tf.Tensor):
        if self.cnn_type == 0:
            dynamic_input = tf.transpose(inputs[0], perm=[0,2,1])
            static_input = inputs[1]
            inputs = [dynamic_input, static_input]
        """ Returns the output of the model given an input. """
        if self.cnn_type == 2:
            dynamic_input = tf.expand_dims(inputs[0],axis=[-1])
            static_input = inputs[1]
            inputs = [dynamic_input, static_input]
        return self.model(inputs)

    def display_overview(self, model_name):
        """ Displays an overview of the model. """
        # self.model.summary()
        model_path = './models/'+model_name+'.png'
        keras.utils.plot_model(self.model,
                               show_shapes=True,
                               show_layer_names=True,
                               to_file=model_path)

class PhiNetwork(layers.Layer):
    """
    Creates a phi network as a layer for the deepset network
    """
    def __init__(self, n_units_phi, activation, name, model_param):
        super(PhiNetwork, self).__init__()
        self.number_layers = len(n_units_phi)
        tf.random.set_seed(model_param["seed"])
        np.random.seed(model_param["seed"])
        if self.number_layers == 3:
            self.phi_layer_1 = layers.Dense(n_units_phi[0], activation=activation, name=name[0])
            self.phi_layer_2 = layers.Dense(n_units_phi[1], activation=activation, name=name[1])
            self.phi_layer_3 = layers.Dense(n_units_phi[2], activation=activation, name=name[2])
        else:
            self.phi_layer_1 = layers.Dense(n_units_phi[0], activation=activation, name=name[0])
            self.phi_layer_2 = layers.Dense(n_units_phi[1], activation=activation, name=name[1])

    @tf.function
    def call(self, inputs):
        if self.number_layers == 3:
            phi1 = self.phi_layer_1(inputs, input_shape=((1,None,4)), batch_size=1)
            phi2 = self.phi_layer_2(phi1)
            phi3 = self.phi_layer_3(phi2)
            return phi3
        else:
            phi1 = self.phi_layer_1(inputs)
            phi2 = self.phi_layer_2(phi1)
            return phi2

class PhiNetwork2(layers.Layer):
    """
    Creates a phi network as a layer for the deepset network
    """
    def __init__(self, n_units_phi, activation, name, model_param):
        super(PhiNetwork2, self).__init__()
        tf.random.set_seed(model_param["seed"])
        np.random.seed(model_param["seed"])
        self.number_layers = len(n_units_phi)
        if self.number_layers == 3:
            self.phi_layer_1 = layers.Dense(n_units_phi[0], activation=activation, name=name[0])
            self.phi_layer_2 = layers.Dense(n_units_phi[1], activation=activation, name=name[1])
            self.phi_layer_3 = layers.Dense(n_units_phi[2], activation=activation, name=name[2])
        else:
            self.phi_layer_1 = layers.Dense(n_units_phi[0], activation=activation, name=name[0])
            self.phi_layer_2 = layers.Dense(n_units_phi[1], activation=activation, name=name[1])

    @tf.function
    def call(self, inputs, batch_size):
        if self.number_layers == 3:
            phi1 = self.phi_layer_1(inputs, input_shape=((batch_size,12,4)), batch_size=(batch_size,12))
            phi2 = self.phi_layer_2(phi1)
            phi3 = self.phi_layer_3(phi2)
            return phi3
        else:
            phi1 = self.phi_layer_1(inputs)
            phi2 = self.phi_layer_2(phi1)
            return phi2

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
        self.n_layers_rho = len(n_units_rho)
        self.n_layers_phi = len(n_units_phi)
        if self.n_layers_phi == 3:
            self.phi_feature_size = self.model_param["deepset_param"]["n_units_phi"][-1]
        else:
            self.phi_feature_size = self.model_param["deepset_param"]["n_units_phi"][1]

        self.is_batch_norm = model_param["batch_normalisation"]
        n_inputs_static = 7
        n_inputs_dynamic = 4
        n_vehicles = 12  # Defined by default D_max size
        n_actions = model_param["n_actions"]

        self.phi_network = PhiNetwork(n_units_phi, act_func_phi, ("PhiLayer1", "PhiLayer2", "PhiLayer3"), model_param)
        self.phi_network2 = PhiNetwork2(n_units_phi, act_func_phi, ("PhiLayer1", "PhiLayer2", "PhiLayer3"), model_param)

        self.sum_layer = layers.Add(name="Summation_layer")
        # self.sum_layer = layers.Lambda(lambda phi_out: tf.expand_dims(tf.reduce_sum(phi_out, axis=0), axis=0))

        if self.n_layers_rho == 3:
            self.rho_layer_1 = layers.Dense(n_units_rho[0], activation=act_func_rho, name="rhoLayer1")
            self.rho_layer_2 = layers.Dense(n_units_rho[1], activation=act_func_rho, name="rhoLayer2")
            self.rho_layer_3 = layers.Dense(n_units_rho[2], activation=act_func_rho, name="rhoLayer3")
            if self.is_batch_norm:
                self.batch_norm_layer = layers.BatchNormalization(name="batch_norm")
                self.concat_layer = layers.Concatenate(name="ConcatenationLayer")
            else:
                self.concat_layer = layers.Concatenate(name="ConcatenationLayer")
        else:  # 2 rho layers
            self.rho_layer_1 = layers.Dense(n_units_rho[0], activation=act_func_rho, name="rhoLayer1")
            self.rho_layer_2 = layers.Dense(n_units_rho[1], activation=act_func_rho, name="rhoLayer2")
            if self.is_batch_norm:
                self.batch_norm_layer = layers.BatchNormalization(name="batch_norm")
                self.concat_layer = layers.Concatenate(name="ConcatenationLayer")
            else:
                self.concat_layer = layers.Concatenate(name="ConcatenationLayer")

        self.Q_layer_1 = layers.Dense(n_units[0], activation=act_func, name="QLayer1")
        self.Q_layer_2 = layers.Dense(n_units[1], activation=act_func, name="QLayer2")

        self.output_layer = layers.Dense(n_actions)


    @tf.function
    def call(self, inputs: tf.Tensor):
        """ Returns the output of the model given an input. """
        dynamic_input = inputs[0]
        static_input = inputs[1]


        # For debugging:
        # x = np.array([
        #     [0, 0, 0, 0],[0, 0, 0, 0],[0, 0, 0, 0],
        #     [0, 0, 0, 0],[0, 0, 0, 0], [0, 0, 0, 0],
        #     [0, 0, 0, 0],[1, 1, 1, 1], [1, 1, 1, 1],
        #     [1, 1, 1, 1],[1, 1, 1, 1], [1, 1, 1, 1]
        # ])
        # dynamic_input = tf.squeeze(tf.convert_to_tensor(np.array([[inputs[0][0,:,:], tf.ones(tf.shape(inputs[0][0,:,:])), tf.zeros(tf.shape(inputs[0][0,:,:])), inputs[0][0,:,:]*x]])))
        batch_size = tf.shape(dynamic_input)[0]

        # non_zero_indices = tf.where(tf.not_equal(tf.reduce_sum(tf.abs(dynamic_input), axis=2), 0))
        # # (batchnumber, vehiclenumber)
        # zero_indices = tf.where(tf.equal(tf.reduce_sum(tf.abs(dynamic_input), axis=2), 0))
        # non_zero_batch_size = tf.size(tf.unique(non_zero_indices[:,0])[0])
        #
        # # Wasused:
        # non_zero_vehicles = tf.gather_nd(dynamic_input, non_zero_indices, batch_dims=0)
        # # number_non_zero_vehicles = tf.cast(tf.shape(non_zero_indices)[0]/non_zero_batch_size, dtype=tf.int32)
        # #Notused:
        # # non_zero_vehicles = tf.reshape(non_zero_vehicles, shape=[non_zero_batch_size, number_non_zero_vehicles, 4])
        # # non_zero_vehicles=tf.gather(dynamic_input, tf.where(tf.not_equal(tf.reduce_sum(tf.abs(dynamic_input), axis=2), 0))[:, 1], axis=1)
        #
        #
        #
        # y_nz, idx_nz, count_nz = tf.unique_with_counts(non_zero_indices[:, 0])
        # y_z, idx_z, count_z = tf.unique_with_counts(zero_indices[:, 0])
        # # all_zero_index = tf.cast(count_z == 12, tf.int32)
        # all_zero_batch_index = tf.cast(tf.boolean_mask(y_z, tf.cast(count_z == 12,tf.int32)),tf.int32)
        # non_zero_batch_index = tf.cast(tf.boolean_mask(y_nz,tf.cast(count_nz >= 1, tf.int32)),tf.int32)
        # batch_phi_out = []
        #
        # for b in range(batch_size):
        #     if tf.reduce_any(b == non_zero_batch_index):
        #         phi_out = []
        #         temp = tf.where(tf.not_equal(tf.reduce_sum(tf.abs(dynamic_input[b, :, :]), axis=1), 0))[:, 0]
        #         non_zero_vehicles_in_batch = tf.gather(dynamic_input[b, :, :], temp, axis=0)
        #         # non_zero_vehicles_in_batch = tf.squeeze(non_zero_vehicles_in_batch)
        #         # Wasused
        #         n_veh = tf.shape(non_zero_vehicles_in_batch)[0]
        #         # for v in tf.range(n_veh):
        #         #     phi_out.append(self.phi_network(tf.expand_dims(non_zero_vehicles_in_batch[v, :], axis=0)))
        #         # phi_sum = self.sum_layer(phi_out)
        #         phi_out = self.phi_network(non_zero_vehicles_in_batch)
        #         # phi_sum = self.sum_layer(phi_out)
        #         phi_sum = tf.reduce_sum(phi_out, axis=0)
        #         batch_phi_out.append(phi_sum)
        #     elif tf.reduce_any(b == all_zero_batch_index):
        #         phi_sum = tf.zeros((self.phi_feature_size))
        #         batch_phi_out.append(phi_sum)
        #     else:
        #         print("error")
        #         sys.exit()
        #
        # if batch_size == 1:
        #     summation = tf.expand_dims(tf.squeeze(batch_phi_out), axis=0)
        # else:
        #     summation = tf.squeeze(batch_phi_out)
        #         # if tf.size(non_zero_vehicles) == 0:
        #         #     summation = tf.zeros((batch_size, self.phi_feature_size))
        #         # else:
        #         #     number_non_zero_veh = tf.shape(non_zero_vehicles)[1]
        #         #     phi_out = []
        #         #     for veh_ind in tf.range(number_non_zero_veh):
        #         #         phi_out.append(self.phi_network(non_zero_vehicles[:, veh_ind, :]))
        #         #     summation = self.sum_layer(phi_out)
        summation = tf.reduce_sum(
            tf.where(tf.expand_dims(tf.not_equal(tf.reduce_sum(tf.abs(dynamic_input), axis=2), 0), axis=[-1]),
                     self.phi_network2(dynamic_input, batch_size), 0), axis=1)
        if self.n_layers_rho == 3:
            rho_1_out = self.rho_layer_1(summation)
            rho_2_out = self.rho_layer_2(rho_1_out)
            rho_3_out = self.rho_layer_3(rho_2_out)
            if self.is_batch_norm:
                batch_norm_out = self.batch_norm_layer(rho_3_out)
                concat = [batch_norm_out, static_input]
                concat_out = self.concat_layer(concat)
            else:
                concat = [rho_3_out, static_input]
                concat_out = self.concat_layer(concat)
        else:
            rho_1_out = self.rho_layer_1(summation)
            rho_2_out = self.rho_layer_2(rho_1_out)
            if self.is_batch_norm:
                batch_norm_out = self.batch_norm_layer(rho_2_out)
                concat = [batch_norm_out, static_input]
                concat_out = self.concat_layer(concat)
            else:
                concat = [rho_2_out, static_input]
                concat_out = self.concat_layer(concat)

        q_1_out = self.Q_layer_1(concat_out)
        q_2_out = self.Q_layer_2(q_1_out)
        output = self.output_layer(q_2_out)
        return output


    def display_overview(self):
        """ Displays an overview of the model. """
        # self.model.summary()
        keras.utils.plot_model(self.model,
                               show_shapes=True,
                               show_layer_names=True,
                               to_file='./models/Deepset_Q_network.png')

class LstmDeepSetNetwork(keras.Model):
    """
    Builds a deep Q-network using DeepSet permutation invariant model with temporal LSTM layer.
    # TODO Check simplification of Deepset network above and test permuting inputs
    """
    def __init__(self, model_param):
        super(LstmDeepSetNetwork, self).__init__()
        self.model_param = model_param
        tf.random.set_seed(model_param["seed"])
        np.random.seed(model_param["seed"])
        act_func = model_param["activation_function"]
        act_func_phi = model_param['deepset_param']['act_func_phi']
        act_func_rho = model_param['deepset_param']['act_func_rho']
        self.n_units_lstm = model_param["LSTM_param"]["n_units"]
        n_units = model_param["n_units"]
        n_units_phi = model_param["deepset_param"]["n_units_phi"]
        n_units_rho = model_param["deepset_param"]["n_units_rho"]
        self.n_layers_rho = len(n_units_rho)
        self.n_layers_phi = len(n_units_phi)
        if self.n_layers_phi == 3:
            self.phi_feature_size = self.model_param["deepset_param"]["n_units_phi"][-1]
        else:
            self.phi_feature_size = self.model_param["deepset_param"]["n_units_phi"][1]

        self.is_batch_norm = model_param["batch_normalisation"]
        n_inputs_static = 7
        n_inputs_dynamic = 4
        n_vehicles = 12  # Defined by default D_max size
        n_actions = model_param["n_actions"]

        self.phi_network = PhiNetwork(n_units_phi, act_func_phi, ("PhiLayer1", "PhiLayer2", "PhiLayer3"), model_param)
        self.phi_network2 = PhiNetwork2(n_units_phi, act_func_phi, ("PhiLayer1", "PhiLayer2", "PhiLayer3"), model_param)

        self.sum_layer = layers.Add(name="Summation_layer")
        # self.sum_layer = layers.Lambda(lambda phi_out: tf.expand_dims(tf.reduce_sum(phi_out, axis=0), axis=0))

        if self.n_layers_rho == 3:
            self.rho_layer_1 = layers.Dense(n_units_rho[0], activation=act_func_rho, name="rhoLayer1")
            self.rho_layer_2 = layers.Dense(n_units_rho[1], activation=act_func_rho, name="rhoLayer2")
            self.rho_layer_3 = layers.Dense(n_units_rho[2], activation=act_func_rho, name="rhoLayer3")
            if self.is_batch_norm:
                self.batch_norm_layer = layers.BatchNormalization(name="batch_norm")
                self.concat_layer = layers.Concatenate(name="ConcatenationLayer")
            else:
                self.concat_layer = layers.Concatenate(name="ConcatenationLayer")
        else:  # 2 rho layers
            self.rho_layer_1 = layers.Dense(n_units_rho[0], activation=act_func_rho, name="rhoLayer1")
            self.rho_layer_2 = layers.Dense(n_units_rho[1], activation=act_func_rho, name="rhoLayer2")
            if self.is_batch_norm:
                self.batch_norm_layer = layers.BatchNormalization(name="batch_norm")
                self.concat_layer = layers.Concatenate(name="ConcatenationLayer")
            else:
                self.concat_layer = layers.Concatenate(name="ConcatenationLayer")

        self.LSTM = layers.LSTM(units=self.n_units_lstm,
                                 name="LSTM")
        self.Q_layer_1 = layers.Dense(n_units[0], activation=act_func, name="QLayer1")
        self.Q_layer_2 = layers.Dense(n_units[1], activation=act_func, name="QLayer2")

        self.output_layer = layers.Dense(n_actions)

    #@tf.function
    def call(self, inputs: tf.Tensor):
        """ Returns the output of the model given an input. """
        dynamic_input = inputs[0]
        static_input = inputs[1]

        batch_size = tf.shape(dynamic_input)[0]

        summation = tf.reduce_sum(
            tf.where(tf.expand_dims(tf.not_equal(tf.reduce_sum(tf.abs(dynamic_input), axis=2), 0), axis=[-1]),
                     self.phi_network2(dynamic_input, batch_size), 0), axis=1)
        if self.n_layers_rho == 3:
            rho_1_out = self.rho_layer_1(summation)
            rho_2_out = self.rho_layer_2(rho_1_out)
            rho_3_out = self.rho_layer_3(rho_2_out)
            if self.is_batch_norm:
                batch_norm_out = self.batch_norm_layer(rho_3_out)
                concat = [batch_norm_out, static_input]
                concat_out = self.concat_layer(concat)
            else:
                concat = [rho_3_out, static_input]
                concat_out = self.concat_layer(concat)
        else:
            rho_1_out = self.rho_layer_1(summation)
            rho_2_out = self.rho_layer_2(rho_1_out)
            if self.is_batch_norm:
                batch_norm_out = self.batch_norm_layer(rho_2_out)
                concat = [batch_norm_out, static_input]
                concat_out = self.concat_layer(concat)
            else:
                concat = [rho_2_out, static_input]
                concat_out = self.concat_layer(concat)


        q_1_out = self.LSTM(concat_out)
        q_2_out = self.Q_layer_2(q_1_out)
        output = self.output_layer(q_2_out)
        return output


    def display_overview(self):
        """ Displays an overview of the model. """
        # self.model.summary()
        keras.utils.plot_model(self.model,
                               show_shapes=True,
                               show_layer_names=True,
                               to_file='./models/LstmDeepSetNetwork.png')


class OldDeepSetQNetwork(keras.Model):
    """
    Builds a deep Q-network using DeepSetQ approach incorporating permutation invariance.
    """
    def __init__(self, model_param):
        super(OldDeepSetQNetwork, self).__init__()
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
                                 name="Old_Deepset_DDQN")

        self.display_overview()

    @tf.function
    def call(self, inputs: tf.Tensor):
        """ Returns the output of the model given an input. """
        return self.model(inputs)

    def display_overview(self):
        """ Displays an overview of the model. """
        # self.model.summary()
        keras.utils.plot_model(self.model,
                               show_shapes=True,
                               show_layer_names=True,
                               to_file='./models/Old_Deepset_Q_network.png')



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

        input_layer0 = layers.Input(shape=(n_inputs,),
                                   name="inputState")
        input_layer = layers.Flatten()(input_layer0)
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

        self.model = keras.Model(inputs=input_layer0,
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
        # self.model.summary()
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
        # self.model.summary()
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
