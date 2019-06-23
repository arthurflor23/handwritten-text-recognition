"""Networks to the Handwritten Text Recognition Model"""

from tensorflow.keras.layers import Input, Conv2D, Bidirectional, LSTM, Dense
from tensorflow.keras.layers import Dropout, BatchNormalization, MaxPooling2D
from tensorflow.keras.layers import Reshape, Activation, LeakyReLU, PReLU
from tensorflow.keras.optimizers import RMSprop
from network.gated import Gated, GatedConv


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
    cnn = Reshape((input_size[0] // 2, input_size[1] // 2, input_size[2] * 4))(input_data)

    cnn = Conv2D(filters=8, kernel_size=(3,3), strides=(1,1), padding="same")(cnn)
    cnn = Activation(activation="tanh")(cnn)

    cnn = Conv2D(filters=16, kernel_size=(2,4), strides=(2,4), padding="same")(cnn)
    cnn = Activation(activation="tanh")(cnn)

    cnn = GatedConv(filters=16, kernel_size=(3,3), strides=(1,1), padding="same")(cnn)

    cnn = Conv2D(filters=32, kernel_size=(3,3), strides=(1,1), padding="same")(cnn)
    cnn = Activation(activation="tanh")(cnn)

    cnn = GatedConv(filters=32, kernel_size=(3,3), strides=(1,1), padding="same")(cnn)

    cnn = Conv2D(filters=64, kernel_size=(2,4), strides=(2,4), padding="same")(cnn)
    cnn = Activation(activation="tanh")(cnn)

    cnn = GatedConv(filters=64, kernel_size=(3,3), strides=(1,1), padding="same")(cnn)

    cnn = Conv2D(filters=128, kernel_size=(3,3), strides=(1,1), padding="same")(cnn)
    cnn = Activation(activation="tanh")(cnn)

    cnn = MaxPooling2D(pool_size=(1,4), strides=(1,4), padding="valid")(cnn)

    shape = cnn.get_shape()
    blstm = Reshape((shape[1], shape[2] * shape[3]))(cnn)

    blstm = Bidirectional(LSTM(units=128, return_sequences=True))(blstm)
    blstm = Dense(units=128)(blstm)
    blstm = Activation(activation="tanh")(blstm)

    blstm = Bidirectional(LSTM(units=128, return_sequences=True))(blstm)
    blstm = Dense(units=output_size)(blstm)

    output_data = Activation(activation="softmax")(blstm)
    optimizer = RMSprop(learning_rate=4e-4)

    return (input_data, output_data, optimizer)


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
    cnn = BatchNormalization(fused=True)(cnn)
    cnn = LeakyReLU(alpha=0.01)(cnn)
    cnn = MaxPooling2D(pool_size=(2,2), strides=(2,2), padding="valid")(cnn)

    cnn = Conv2D(filters=32, kernel_size=(3,3), strides=(1,1), padding="same")(cnn)
    cnn = BatchNormalization(fused=True)(cnn)
    cnn = LeakyReLU(alpha=0.01)(cnn)
    cnn = MaxPooling2D(pool_size=(2,2), strides=(2,2), padding="valid")(cnn)

    cnn = Dropout(rate=0.2)(cnn)
    cnn = Conv2D(filters=48, kernel_size=(3,3), strides=(1,1), padding="same")(cnn)
    cnn = BatchNormalization(fused=True)(cnn)
    cnn = LeakyReLU(alpha=0.01)(cnn)
    cnn = MaxPooling2D(pool_size=(2,2), strides=(2,2), padding="valid")(cnn)

    cnn = Dropout(rate=0.2)(cnn)
    cnn = Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding="same")(cnn)
    cnn = BatchNormalization(fused=True)(cnn)
    cnn = LeakyReLU(alpha=0.01)(cnn)

    cnn = Dropout(rate=0.2)(cnn)
    cnn = Conv2D(filters=80, kernel_size=(3,3), strides=(1,1), padding="same")(cnn)
    cnn = BatchNormalization(fused=True)(cnn)
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
    blstm = Dense(units=output_size)(blstm)

    output_data = Activation(activation="softmax")(blstm)
    optimizer = RMSprop(learning_rate=3e-4)

    return (input_data, output_data, optimizer)


def flor(input_size, output_size):
    input_data = Input(name="input", shape=input_size)

    cnn = Conv2D(filters=16, kernel_size=(3,3), strides=(2,2), padding="same")(input_data)
    cnn = PReLU(shared_axes=[1,2])(cnn)
    cnn = BatchNormalization(renorm=True)(cnn)

    cnn = Conv2D(filters=32, kernel_size=(3,3), strides=(1,1), padding="same")(cnn)
    cnn = PReLU(shared_axes=[1,2])(cnn)
    cnn = BatchNormalization(renorm=True)(cnn)

    cnn = Gated(filters=32, kernel_size=(3,3), padding="same")(cnn)

    cnn = Conv2D(filters=40, kernel_size=(2,4), strides=(2,4), padding="same")(cnn)
    cnn = PReLU(shared_axes=[1,2])(cnn)
    cnn = BatchNormalization(renorm=True)(cnn)

    cnn = Gated(filters=40, kernel_size=(3,3), padding="same")(cnn)
    cnn = Dropout(rate=0.2)(cnn)

    cnn = Conv2D(filters=48, kernel_size=(3,3), strides=(1,1), padding="same")(cnn)
    cnn = PReLU(shared_axes=[1,2])(cnn)
    cnn = BatchNormalization(renorm=True)(cnn)

    cnn = Gated(filters=48, kernel_size=(3,3), padding="same")(cnn)
    cnn = Dropout(rate=0.2)(cnn)

    cnn = Conv2D(filters=56, kernel_size=(2,4), strides=(2,4), padding="same")(cnn)
    cnn = PReLU(shared_axes=[1,2])(cnn)
    cnn = BatchNormalization(renorm=True)(cnn)

    cnn = Gated(filters=56, kernel_size=(3,3), padding="same")(cnn)
    cnn = Dropout(rate=0.2)(cnn)

    cnn = Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding="same")(cnn)
    cnn = PReLU(shared_axes=[1,2])(cnn)
    cnn = BatchNormalization(renorm=True)(cnn)

    cnn = MaxPooling2D(pool_size=(1,2), strides=(1,2), padding="valid")(cnn)

    shape = cnn.get_shape()
    blstm = Reshape((shape[1], shape[2] * shape[3]))(cnn)

    blstm = Bidirectional(LSTM(units=128, return_sequences=True, dropout=0.5))(blstm)
    blstm = Dense(units=128)(blstm)

    blstm = Bidirectional(LSTM(units=128, return_sequences=True, dropout=0.5))(blstm)
    blstm = Dense(units=output_size)(blstm)

    output_data = Activation(activation="softmax")(blstm)

    optimizer = RMSprop(learning_rate=5e-4)

    return (input_data, output_data, optimizer)
