"""Networks to the Handwritten Text Recognition Model"""

from tensorflow.keras.layers import Input, Conv2D, Bidirectional, LSTM, Dense, Multiply
from tensorflow.keras.layers import Dropout, BatchNormalization, MaxPooling2D, Reshape
from tensorflow.keras.layers import TimeDistributed, Activation, LeakyReLU, ReLU
from tensorflow.keras.experimental import CosineDecayRestarts
from tensorflow.keras.optimizers import RMSprop


def puigcerver(env):
    """
    Convolucional Recurrent Neural Network by Puigcerver et al.
        Reference:
            Puigcerver, J.: Are multidimensional recurrent layers really
            necessary for handwritten text recognition? In: Document
            Analysis and Recognition (ICDAR), 2017 14th
            IAPR International Conference on, vol. 1, pp. 67–72. IEEE (2017)
    """

    input_data = Input(name="input", shape=env.input_size)

    cnn = Conv2D(filters=16, kernel_size=(3,3), strides=(1,1), padding="same")(input_data)
    cnn = BatchNormalization(epsilon=0.001)(cnn)
    cnn = LeakyReLU()(cnn)
    cnn = MaxPooling2D(pool_size=(2,2), strides=(2,2), padding="valid")(cnn)

    cnn = Conv2D(filters=32, kernel_size=(3,3), strides=(1,1), padding="same")(cnn)
    cnn = BatchNormalization(epsilon=0.001)(cnn)
    cnn = LeakyReLU()(cnn)
    cnn = MaxPooling2D(pool_size=(2,2), strides=(2,2), padding="valid")(cnn)

    cnn = Dropout(rate=0.2)(cnn)
    cnn = Conv2D(filters=48, kernel_size=(3,3), strides=(1,1), padding="same")(cnn)
    cnn = BatchNormalization(epsilon=0.001)(cnn)
    cnn = LeakyReLU()(cnn)
    cnn = MaxPooling2D(pool_size=(2,2), strides=(2,2), padding="valid")(cnn)

    cnn = Dropout(rate=0.2)(cnn)
    cnn = Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding="same")(cnn)
    cnn = BatchNormalization(epsilon=0.001)(cnn)
    cnn = LeakyReLU()(cnn)

    cnn = Dropout(rate=0.2)(cnn)
    cnn = Conv2D(filters=80, kernel_size=(3,3), strides=(1,1), padding="same")(cnn)
    cnn = BatchNormalization(epsilon=0.001)(cnn)
    cnn = LeakyReLU()(cnn)

    shape = cnn.get_shape()
    blstm = Reshape((shape[1], shape[2] * shape[3]))(cnn)

    blstm = Dropout(rate=0.5)(blstm)
    blstm = Bidirectional(LSTM(units=256, return_sequences=True))(blstm)

    blstm = Dropout(rate=0.5)(blstm)
    blstm = Bidirectional(LSTM(units=256, return_sequences=True))(blstm)

    blstm = Dropout(rate=0.5)(blstm)
    blstm = Bidirectional(LSTM(units=256, return_sequences=True))(blstm)

    blstm = Dropout(rate=0.5)(blstm)
    blstm = Bidirectional(LSTM(units=256, return_sequences=True))(blstm)

    blstm = Dropout(rate=0.5)(blstm)
    blstm = Bidirectional(LSTM(units=256, return_sequences=True))(blstm)

    blstm = Dropout(rate=0.5)(blstm)
    blstm = TimeDistributed(Dense(units=(len(env.charset) + 1)))(blstm)

    outrnn = Activation(activation="softmax")(blstm)

    decay_lr = CosineDecayRestarts(initial_learning_rate=3e-4, first_decay_steps=int(64000 / env.batch_size))
    opt = RMSprop(learning_rate=decay_lr)

    return (input_data, outrnn, opt)


def bluche(env):
    """
    Gated Convolucional Recurrent Neural Network by Bluche et al.
        Reference:
            Bluche, T., Messina, R.: Gated convolutional recurrent
            neural networks for multilingual handwriting recognition.
            In: Document Analysis and Recognition (ICDAR), 2017
            14th IAPR International Conference on, vol. 1, pp. 646–651, 2017.
    """

    input_data = Input(name="input", shape=env.input_size)

    cnn = Conv2D(filters=8, kernel_size=(3,3), strides=(1,1), padding="same")(input_data)
    cnn = BatchNormalization(epsilon=0.001)(cnn)
    cnn = ReLU()(cnn)

    cnn = Conv2D(filters=16, kernel_size=(2,4), strides=(2,4), padding="same")(cnn)
    cnn = BatchNormalization(epsilon=0.001)(cnn)
    cnn = ReLU()(cnn)

    cnn = GatedConv(nb_filters=16, kernel_size=(1,3), strides=(1,1))(cnn)
    cnn = BatchNormalization(epsilon=0.001)(cnn)

    cnn = Conv2D(filters=32, kernel_size=(3,3), strides=(1,1), padding="same")(cnn)
    cnn = BatchNormalization(epsilon=0.001)(cnn)
    cnn = ReLU()(cnn)

    cnn = GatedConv(nb_filters=32, kernel_size=(1,3), strides=(1,1))(cnn)
    cnn = BatchNormalization(epsilon=0.001)(cnn)

    cnn = Conv2D(filters=64, kernel_size=(2,4), strides=(2,4), padding="same")(cnn)
    cnn = BatchNormalization(epsilon=0.001)(cnn)
    cnn = ReLU()(cnn)

    cnn = GatedConv(nb_filters=64, kernel_size=(1,3), strides=(1,1))(cnn)
    cnn = BatchNormalization(epsilon=0.001)(cnn)

    cnn = Conv2D(filters=128, kernel_size=(3,3), strides=(1,1), padding="same")(cnn)
    cnn = BatchNormalization(epsilon=0.001)(cnn)
    cnn = ReLU()(cnn)
    cnn = MaxPooling2D(pool_size=(2,2), strides=(2,2), padding="valid")(cnn)

    shape = cnn.get_shape()
    blstm = Reshape((shape[1], shape[2] * shape[3]))(cnn)

    blstm = Dropout(rate=0.5)(blstm)
    blstm = Bidirectional(LSTM(units=128, return_sequences=True))(blstm)
    blstm = Dense(units=128)(blstm)

    blstm = Dropout(rate=0.5)(blstm)
    blstm = Bidirectional(LSTM(units=128, return_sequences=True))(blstm)
    blstm = TimeDistributed(Dense(units=(len(env.charset) + 1)))(blstm)

    outrnn = Activation(activation="softmax")(blstm)

    decay_lr = CosineDecayRestarts(initial_learning_rate=4e-4, first_decay_steps=int(64000 / env.batch_size))
    opt = RMSprop(learning_rate=decay_lr)

    return (input_data, outrnn, opt)


"""
A Tensorflow Keras layer implementing gated convolutions by Dauphin et al.
    Args:
        nb_filters (int): Number of output filters.
        kernel_size (int or tuple): Size of convolution kernel.
        strides (int or tuple): Strides of the convolution.
        padding (str): One of ``'valid'`` or ``'same'``.
        kwargs: Other layer keyword arguments.
    Reference:
        Y. N. Dauphin, A. Fan, M. Auli, and D. Grangier,
        Language modeling with gated convolutional networks, in
        Proc. 34th Int. Conf. Mach. Learn. (ICML), vol. 70,
        Sydney, Australia, pp. 933–941, 2017.
"""


class GatedConv(Conv2D):
    """Gated Convolutional Class"""

    def __init__(self, nb_filters=64, kernel_size=(3, 3), **kwargs):
        super(GatedConv, self).__init__(filters=nb_filters * 2, kernel_size=kernel_size, **kwargs)
        self.nb_filters = nb_filters

    def call(self, inputs):
        """Apply gated convolution"""

        output = super(GatedConv, self).call(inputs)
        nb_filters = self.nb_filters
        linear = Activation("linear")(output[:, :, :, :nb_filters])
        sigmoid = Activation("sigmoid")(output[:, :, :, nb_filters:])

        return Multiply()([linear, sigmoid])

    def compute_output_shape(self, input_shape):
        """Compute shape of layer output"""

        output_shape = super(GatedConv, self).compute_output_shape(input_shape)
        return tuple(output_shape[:3]) + (self.nb_filters,)

    def get_config(self):
        """Return the config of the layer"""

        config = super(GatedConv, self).get_config()
        config['nb_filters'] = self.nb_filters
        del config['filters']
        return config
