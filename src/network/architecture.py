"""Networks to the Handwritten Text Recognition Model"""

from network.layers import FullGatedConv2D, GatedConv2D
from tensorflow.keras.constraints import MaxNorm
from tensorflow.keras.layers import Conv2D, Bidirectional, LSTM, GRU, Dense
from tensorflow.keras.layers import Dropout, BatchNormalization, LeakyReLU, PReLU
from tensorflow.keras.layers import Input, MaxPooling2D, Reshape, TimeDistributed
from tensorflow.keras.optimizers import RMSprop


def bluche(input_size, output_size, learning_rate=4e-4):
    """
    Gated Convolucional Recurrent Neural Network by Bluche et al.
        Reference:
            Bluche, T., Messina, R.:
            Gated convolutional recurrent neural networks for multilingual handwriting recognition.
            In: Document Analysis and Recognition (ICDAR), 2017
            14th IAPR International Conference on, vol. 1, pp. 646–651, 2017.
            URL: https://ieeexplore.ieee.org/document/8270042

            Moysset, B. and Messina, R.:
            Are 2D-LSTM really dead for offline text recognition?
            In: International Journal on Document Analysis and Recognition (IJDAR)
            Springer Science and Business Media LLC
            URL: http://dx.doi.org/10.1007/s10032-019-00325-0
    """

    input_data = Input(name="input", shape=input_size)
    cnn = Reshape((input_size[0] // 2, input_size[1] // 2, input_size[2] * 4))(input_data)

    cnn = Conv2D(filters=8, kernel_size=(3,3), strides=(1,1), padding="same", activation="tanh")(cnn)
    cnn = Dropout(rate=0.5)(cnn)

    cnn = Conv2D(filters=16, kernel_size=(2,4), strides=(2,4), padding="same", activation="tanh")(cnn)
    cnn = Dropout(rate=0.5)(cnn)

    cnn = GatedConv2D(filters=16, kernel_size=(3,3), strides=(1,1), padding="same")(cnn)

    cnn = Conv2D(filters=32, kernel_size=(3,3), strides=(1,1), padding="same", activation="tanh")(cnn)
    cnn = Dropout(rate=0.5)(cnn)

    cnn = GatedConv2D(filters=32, kernel_size=(3,3), strides=(1,1), padding="same")(cnn)

    cnn = Conv2D(filters=64, kernel_size=(2,4), strides=(2,4), padding="same", activation="tanh")(cnn)
    cnn = Dropout(rate=0.5)(cnn)

    cnn = GatedConv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding="same")(cnn)

    cnn = Conv2D(filters=128, kernel_size=(3,3), strides=(1,1), padding="same", activation="tanh")(cnn)
    cnn = Dropout(rate=0.5)(cnn)

    cnn = MaxPooling2D(pool_size=(1,4), strides=(1,4), padding="valid")(cnn)

    shape = cnn.get_shape()
    blstm = Reshape((shape[1], shape[2] * shape[3]))(cnn)

    blstm = Bidirectional(LSTM(units=128, return_sequences=True, dropout=0.5))(blstm)
    blstm = Dense(units=128, activation="tanh")(blstm)

    blstm = Bidirectional(LSTM(units=128, return_sequences=True, dropout=0.5))(blstm)
    output_data = Dense(units=output_size, activation="softmax")(blstm)

    optimizer = RMSprop(learning_rate=learning_rate)

    return (input_data, output_data, optimizer)


def puigcerver(input_size, output_size, learning_rate=3e-4):
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

    blstm = Bidirectional(LSTM(units=256, return_sequences=True, dropout=0.5))(blstm)
    blstm = Bidirectional(LSTM(units=256, return_sequences=True, dropout=0.5))(blstm)
    blstm = Bidirectional(LSTM(units=256, return_sequences=True, dropout=0.5))(blstm)
    blstm = Bidirectional(LSTM(units=256, return_sequences=True, dropout=0.5))(blstm)
    blstm = Bidirectional(LSTM(units=256, return_sequences=True, dropout=0.5))(blstm)

    blstm = Dropout(rate=0.5)(blstm)
    output_data = Dense(units=output_size, activation="softmax")(blstm)

    optimizer = RMSprop(learning_rate=learning_rate)

    return (input_data, output_data, optimizer)


def flor(input_size, output_size, learning_rate=5e-4):
    """Gated Convolucional Recurrent Neural Network by Flor."""

    input_data = Input(name="input", shape=input_size)

    cnn = Conv2D(filters=16, kernel_size=(3,3), strides=(2,2), padding="same")(input_data)
    cnn = PReLU(shared_axes=[1,2])(cnn)
    cnn = BatchNormalization(renorm=True)(cnn)

    cnn = FullGatedConv2D(filters=16, kernel_size=(3,3), padding="same")(cnn)

    cnn = Conv2D(filters=32, kernel_size=(3,3), strides=(1,2), padding="same")(cnn)
    cnn = PReLU(shared_axes=[1,2])(cnn)
    cnn = BatchNormalization(renorm=True)(cnn)

    cnn = FullGatedConv2D(filters=32, kernel_size=(3,3), padding="same")(cnn)

    cnn = Conv2D(filters=40, kernel_size=(2,4), strides=(2,2), padding="same")(cnn)
    cnn = PReLU(shared_axes=[1,2])(cnn)
    cnn = BatchNormalization(renorm=True)(cnn)

    cnn = FullGatedConv2D(filters=40, kernel_size=(3,3), padding="same", kernel_constraint=MaxNorm(4, [0,1,2]))(cnn)
    cnn = Dropout(rate=0.2)(cnn)

    cnn = Conv2D(filters=48, kernel_size=(3,3), strides=(1,2), padding="same")(cnn)
    cnn = PReLU(shared_axes=[1,2])(cnn)
    cnn = BatchNormalization(renorm=True)(cnn)

    cnn = FullGatedConv2D(filters=48, kernel_size=(3,3), padding="same", kernel_constraint=MaxNorm(4, [0,1,2]))(cnn)
    cnn = Dropout(rate=0.2)(cnn)

    cnn = Conv2D(filters=56, kernel_size=(2,4), strides=(2,2), padding="same")(cnn)
    cnn = PReLU(shared_axes=[1,2])(cnn)
    cnn = BatchNormalization(renorm=True)(cnn)

    cnn = FullGatedConv2D(filters=56, kernel_size=(3,3), padding="same", kernel_constraint=MaxNorm(4, [0,1,2]))(cnn)
    cnn = Dropout(rate=0.2)(cnn)

    cnn = Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding="same")(cnn)
    cnn = PReLU(shared_axes=[1,2])(cnn)
    cnn = BatchNormalization(renorm=True)(cnn)

    cnn = MaxPooling2D(pool_size=(1,2), strides=(1,2), padding="valid")(cnn)

    shape = cnn.get_shape()
    bgru = Reshape((shape[1], shape[2] * shape[3]))(cnn)

    bgru = Bidirectional(GRU(units=128, return_sequences=True, dropout=0.5))(bgru)
    bgru = TimeDistributed(Dense(units=128))(bgru)

    bgru = Bidirectional(GRU(units=128, return_sequences=True, dropout=0.5))(bgru)
    output_data = TimeDistributed(Dense(units=output_size, activation="softmax"))(bgru)

    optimizer = RMSprop(learning_rate=learning_rate)

    return (input_data, output_data, optimizer)
