"""Create the HTR deep learning model using tf.keras"""

import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Lambda
from tensorflow.keras.layers import Activation, BatchNormalization, Reshape
from tensorflow.keras.backend import ctc_batch_cost
from tensorflow.keras.optimizers import RMSprop

# import os
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

INPUT_SIZE = (800, 64, 1)


class HTR():

    def __init__(self, data_gen):
        self.max_line_length = data_gen.max_line_length
        self.rnn_output = data_gen.output_size

        self.__build_model()
        data_gen.downsample_factor = self.downsample_factor

    def __build_model(self):
        input_data = Input(name="the_inputs", shape=INPUT_SIZE, dtype="float32")

        cnn_out = self.__setup_cnn(input_data)
        rnn_out = self.__setup_rnn(cnn_out)

        labels = Input(name='the_labels', shape=[self.max_line_length], dtype='float32')
        input_length = Input(name='input_length', shape=[1], dtype='int64')
        label_length = Input(name='label_length', shape=[1], dtype='int64')

        args = [rnn_out, labels, input_length, label_length]
        loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')(args)

        args[0] = input_data
        self.model = Model(name="HTR", inputs=args, outputs=loss_out)

        opt = RMSprop(learning_rate=1e-4)
        self.model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=opt)

    def __setup_cnn(self, input_data):
        """CNN model"""

        kernels = [5, 5, 3, 3, 3]
        pool_sizes = strides = [(2,2), (2,2), (1,2), (1,2), (1,2)]

        self.downsample_factor = sum([p[0] for p in pool_sizes if p[0] > 1])
        self.filters = [64, 128, 256, 256, 512]

        cnn = input_data
        init = tf.random_normal_initializer(stddev=0.1)

        for i in range(len(strides)):
            conv = Conv2D(filters=self.filters[i], kernel_size=kernels[i], padding="same", activation="relu", kernel_initializer=init)(cnn)
            batch_norm = BatchNormalization()(conv)
            cnn = MaxPooling2D(pool_size=pool_sizes[i], strides=strides[i], padding="valid")(batch_norm)

        return cnn

    def __setup_rnn(self, input_data):
        """RNN model"""

        conv_to_rnn_dims = (INPUT_SIZE[:2][0] // self.downsample_factor,
                            (INPUT_SIZE[:2][1] // self.downsample_factor) * self.filters[0])
        inner = Reshape(target_shape=conv_to_rnn_dims)(input_data)

        num_units = self.filters[-1]
        lstm = tf.keras.layers.LSTM(units=num_units, return_sequences=True, kernel_initializer="he_normal")
        bilstm = tf.keras.layers.Bidirectional(layer=lstm)(inner)

        bilstm = Dense(units=self.rnn_output, kernel_initializer="he_normal")(bilstm)
        bilstm = Activation(name="softmax", activation="softmax")(bilstm)

        return bilstm


def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    # the 2 is critical here since the first couple outputs of the RNN tend to be garbage:
    y_pred = y_pred[:, 2:, :]
    return ctc_batch_cost(labels, y_pred, input_length, label_length)
