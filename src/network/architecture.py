"""Networks to the Handwritten Text Recognition Model"""

from tensorflow.keras.layers import Input, Conv2D, Bidirectional, LSTM, Dense, Multiply
from tensorflow.keras.layers import Dropout, BatchNormalization, MaxPooling2D, Reshape
from tensorflow.keras.layers import Activation, LeakyReLU, PReLU
from tensorflow.keras.optimizers import RMSprop


def bluche(input_size, output_size):
    """
    Gated Convolucional Recurrent Neural Network by Bluche et al.
        Reference:
            Bluche, T., Messina, R.: Gated convolutional recurrent
            neural networks for multilingual handwriting recognition.
            In: Document Analysis and Recognition (ICDAR), 2017
            14th IAPR International Conference on, vol. 1, pp. 646–651, 2017.
    """

    input_data = Input(name="input", shape=input_size)

    cnn = Conv2D(filters=8, kernel_size=(3,3), strides=(2,2), padding="same")(input_data)
    cnn = Activation(activation="tanh")(cnn)

    cnn = Conv2D(filters=16, kernel_size=(2,4), strides=(2,4), padding="same")(cnn)
    cnn = Activation(activation="tanh")(cnn)
    cnn = GatedConv(nb_filters=16, kernel_size=(3,3), padding="same")(cnn)

    cnn = Conv2D(filters=32, kernel_size=(3,3), strides=(1,1), padding="same")(cnn)
    cnn = Activation(activation="tanh")(cnn)
    cnn = GatedConv(nb_filters=32, kernel_size=(3,3), padding="same")(cnn)

    cnn = Conv2D(filters=64, kernel_size=(2,4), strides=(2,4), padding="same")(cnn)
    cnn = Activation(activation="tanh")(cnn)
    cnn = GatedConv(nb_filters=64, kernel_size=(3,3), padding="same")(cnn)

    cnn = Conv2D(filters=128, kernel_size=(3,3), strides=(1,1), padding="same")(cnn)
    cnn = Activation(activation="tanh")(cnn)

    cnn = MaxPooling2D(pool_size=(1,4), strides=(1,4), padding="valid")(cnn)

    shape = cnn.get_shape()
    blstm = Reshape((shape[1], shape[2] * shape[3]))(cnn)

    blstm = Bidirectional(LSTM(units=128, return_sequences=True))(blstm)
    blstm = Dense(units=128)(blstm)
    blstm = Activation(activation="tanh")(blstm)

    blstm = Bidirectional(LSTM(units=128, return_sequences=True))(blstm)
    blstm = Dense(units=(output_size + 1))(blstm)
    outrnn = Activation(activation="softmax")(blstm)

    optimizer = RMSprop(learning_rate=4e-4)

    return (input_data, outrnn, optimizer)


def puigcerver(input_size, output_size):
    """
    Convolucional Recurrent Neural Network by Puigcerver et al.
        Reference:
            Puigcerver, J.: Are multidimensional recurrent layers really
            necessary for handwritten text recognition? In: Document
            Analysis and Recognition (ICDAR), 2017 14th
            IAPR International Conference on, vol. 1, pp. 67–72. IEEE (2017)
    """

    input_data = Input(name="input", shape=input_size)

    cnn = Conv2D(filters=16, kernel_size=(3,3), strides=(1,1), padding="same")(input_data)
    cnn = BatchNormalization()(cnn)
    cnn = LeakyReLU(alpha=0.01)(cnn)
    cnn = MaxPooling2D(pool_size=(2,2), strides=(2,2), padding="valid")(cnn)

    cnn = Conv2D(filters=32, kernel_size=(3,3), strides=(1,1), padding="same")(cnn)
    cnn = BatchNormalization()(cnn)
    cnn = LeakyReLU(alpha=0.01)(cnn)
    cnn = MaxPooling2D(pool_size=(2,2), strides=(2,2), padding="valid")(cnn)

    cnn = Dropout(rate=0.2)(cnn)
    cnn = Conv2D(filters=48, kernel_size=(3,3), strides=(1,1), padding="same")(cnn)
    cnn = BatchNormalization()(cnn)
    cnn = LeakyReLU(alpha=0.01)(cnn)
    cnn = MaxPooling2D(pool_size=(2,2), strides=(2,2), padding="valid")(cnn)

    cnn = Dropout(rate=0.2)(cnn)
    cnn = Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding="same")(cnn)
    cnn = BatchNormalization()(cnn)
    cnn = LeakyReLU(alpha=0.01)(cnn)

    cnn = Dropout(rate=0.2)(cnn)
    cnn = Conv2D(filters=80, kernel_size=(3,3), strides=(1,1), padding="same")(cnn)
    cnn = BatchNormalization()(cnn)
    cnn = LeakyReLU(alpha=0.01)(cnn)

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
    blstm = Dense(units=(output_size + 1))(blstm)
    outrnn = Activation(activation="softmax")(blstm)

    optimizer = RMSprop(learning_rate=3e-4)

    return (input_data, outrnn, optimizer)


def flor(input_size, output_size):
    input_data = Input(name="input", shape=input_size)

    cnn = Conv2D(filters=16, kernel_size=(3,3), strides=(2,1), padding="same", kernel_initializer="he_uniform")(input_data)
    cnn = PReLU(shared_axes=[1,2])(cnn)
    # cnn = BatchNormalization(renorm=True)(cnn)

    cnn = Conv2D(filters=32, kernel_size=(2,4), strides=(2,2), padding="same", kernel_initializer="he_uniform")(cnn)
    cnn = PReLU(shared_axes=[1,2])(cnn)
    # cnn = BatchNormalization(renorm=True)(cnn)
    cnn = GatedConv(nb_filters=32, kernel_size=(3,3), padding="same")(cnn)

    cnn = Conv2D(filters=48, kernel_size=(3,3), strides=(2,4), padding="same", kernel_initializer="he_uniform")(cnn)
    cnn = PReLU(shared_axes=[1,2])(cnn)
    # cnn = BatchNormalization(renorm=True)(cnn)

    cnn = Conv2D(filters=64, kernel_size=(2,4), strides=(1,2), padding="same", kernel_initializer="he_uniform")(cnn)
    cnn = PReLU(shared_axes=[1,2])(cnn)
    # cnn = BatchNormalization(renorm=True)(cnn)
    cnn = GatedConv(nb_filters=64, kernel_size=(3,3), padding="same")(cnn)

    cnn = Conv2D(filters=128, kernel_size=(3,3), strides=(1,2), padding="same", kernel_initializer="he_uniform")(cnn)
    cnn = PReLU(shared_axes=[1,2])(cnn)
    # cnn = BatchNormalization(renorm=True)(cnn)

    cnn = MaxPooling2D(pool_size=(1,4), strides=(1,4), padding="valid")(cnn)

    shape = cnn.get_shape()
    blstm = Reshape((shape[1], shape[2] * shape[3]))(cnn)

    # blstm = Dropout(rate=0.5)(blstm)
    blstm = Bidirectional(LSTM(units=128, return_sequences=True))(blstm)
    blstm = Dense(units=128)(blstm)

    # blstm = Dropout(rate=0.5)(blstm)
    blstm = Bidirectional(LSTM(units=128, return_sequences=True))(blstm)
    blstm = Dense(units=(output_size + 1))(blstm)
    outrnn = Activation(activation="softmax")(blstm)

    optimizer = RMSprop(learning_rate=4e-4)

    return (input_data, outrnn, optimizer)


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

    def __init__(self, nb_filters, **kwargs):
        super(GatedConv, self).__init__(filters=nb_filters, **kwargs)
        self.nb_filters = nb_filters

    def call(self, inputs):
        """Apply gated convolution"""

        output = super(GatedConv, self).call(inputs)
        linear = Activation("linear")(output)
        sigmoid = Activation("sigmoid")(output)

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
