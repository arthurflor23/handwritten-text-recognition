import tensorflow as tf

from sarah.models.components.base import BaseRecognitionModel
from sarah.models.components.layers import OctConv2D


class RecognitionModel(BaseRecognitionModel):
    """
    TensorFlow model for multilingual handwriting recognition using OCNN and BiLSTMs.
    It's based on traditional deep learning methods for offline handwriting recognition (CRNN).

    References
    ----------
    Are multidimensional recurrent layers really necessary for handwritten text recognition?
        https://ieeexplore.ieee.org/document/8269951

    Drop an Octave: Reducing Spatial Redundancy in Convolutional Neural Networks with Octave Convolution
        https://arxiv.org/abs/1904.05049
    """

    def compile(self, learning_rate=None):
        """
        Compiles neural network model.

        Parameters
        ----------
        learning_rate : float, optional
            The learning rate for the optimizer.
        """

        super().compile(run_eagerly=False)

        if learning_rate is None:
            learning_rate = 3e-4

        self.optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate, rho=0.9, epsilon=1e-8)

    def build_model(self):
        """
        Builds the neural network model.

        This method sets up the architecture of the model by defining layers, their connections,
            and configurations. It is typically called in the constructor to create the model structure.
        """

        # encoder model
        encoder_input = tf.keras.Input(shape=self.image_shape)

        inputs = [encoder_input, tf.keras.layers.AveragePooling2D(pool_size=2)(encoder_input)]
        high, low = OctConv2D(alpha=0.25, filters=16)(inputs)

        high = tf.keras.layers.BatchNormalization()(high)
        low = tf.keras.layers.BatchNormalization()(low)
        high = tf.keras.layers.LeakyReLU(negative_slope=0.01)(high)
        low = tf.keras.layers.LeakyReLU(negative_slope=0.01)(low)
        high = tf.keras.layers.MaxPooling2D(pool_size=(2, 1), strides=(2, 1), padding='valid')(high)
        low = tf.keras.layers.MaxPooling2D(pool_size=(2, 1), strides=(2, 1), padding='valid')(low)

        high, low = OctConv2D(alpha=0.25, filters=32)([high, low])

        high = tf.keras.layers.BatchNormalization()(high)
        low = tf.keras.layers.BatchNormalization()(low)
        high = tf.keras.layers.LeakyReLU(negative_slope=0.01)(high)
        low = tf.keras.layers.LeakyReLU(negative_slope=0.01)(low)
        high = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(high)
        low = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(low)

        high = tf.keras.layers.Dropout(rate=0.2)(high)
        low = tf.keras.layers.Dropout(rate=0.2)(low)

        high = tf.keras.layers.Conv2D(filters=48, kernel_size=(3, 3), strides=(1, 1), padding='same')(high)
        low = tf.keras.layers.Conv2D(filters=48, kernel_size=(3, 3), strides=(1, 1), padding='same')(low)

        high = tf.keras.layers.BatchNormalization()(high)
        low = tf.keras.layers.BatchNormalization()(low)
        high = tf.keras.layers.LeakyReLU(negative_slope=0.01)(high)
        low = tf.keras.layers.LeakyReLU(negative_slope=0.01)(low)
        high = tf.keras.layers.MaxPooling2D(pool_size=(1, 2), strides=(1, 2), padding='valid')(high)
        low = tf.keras.layers.MaxPooling2D(pool_size=(1, 2), strides=(1, 2), padding='valid')(low)

        high = tf.keras.layers.Dropout(rate=0.2)(high)
        low = tf.keras.layers.Dropout(rate=0.2)(low)

        high = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same')(high)
        low = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same')(low)

        high = tf.keras.layers.BatchNormalization()(high)
        low = tf.keras.layers.BatchNormalization()(low)
        high = tf.keras.layers.LeakyReLU(negative_slope=0.01)(high)
        low = tf.keras.layers.LeakyReLU(negative_slope=0.01)(low)

        high = tf.keras.layers.Dropout(rate=0.2)(high)
        low = tf.keras.layers.Dropout(rate=0.2)(low)

        high = tf.keras.layers.Conv2D(filters=80, kernel_size=(3, 3), strides=(1, 1), padding='same')(high)
        low = tf.keras.layers.Conv2D(filters=80, kernel_size=(3, 3), strides=(1, 1), padding='same')(low)

        high = tf.keras.layers.BatchNormalization()(high)
        low = tf.keras.layers.BatchNormalization()(low)
        high = tf.keras.layers.LeakyReLU(negative_slope=0.01)(high)
        low = tf.keras.layers.LeakyReLU(negative_slope=0.01)(low)

        high, low = OctConv2D(alpha=0.25, filters=80)([high, low])

        high = tf.keras.layers.BatchNormalization()(high)
        high = tf.keras.layers.Activation('relu')(high)

        low = tf.keras.layers.BatchNormalization()(low)
        low = tf.keras.layers.Activation('relu')(low)

        high_to_high = tf.keras.layers.Conv2D(80, 3, padding='same')(high)
        low_to_high = tf.keras.layers.Conv2D(80, 3, padding='same')(low)

        low_to_high = tf.keras.layers.Lambda(lambda x: tf.tile(x, [1, 2, 2, 1]), name='tile')(low_to_high)

        encoder = tf.keras.layers.Add()([high_to_high, low_to_high])
        encoder = tf.keras.layers.BatchNormalization()(encoder)
        encoder = tf.keras.layers.Activation('relu')(encoder)

        self.encoder = tf.keras.Model(name='encoder', inputs=encoder_input, outputs=encoder)

        # decoder model
        decoder_input = tf.keras.Input(shape=encoder.shape[1:])

        decoder = tf.keras.layers.Reshape(target_shape=(encoder.shape[1], -1))(decoder_input)

        decoder = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(units=256, dropout=0.5, return_sequences=True))(decoder)

        decoder = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(units=256, dropout=0.5, return_sequences=True))(decoder)

        decoder = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(units=256, dropout=0.5, return_sequences=True))(decoder)

        decoder = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(units=256, dropout=0.5, return_sequences=True))(decoder)

        decoder = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(units=256, dropout=0.5, return_sequences=True))(decoder)

        decoder = tf.keras.layers.Dropout(rate=0.5)(decoder)

        decoder = tf.keras.layers.Dense(units=self.lexical_shape[-1], activation='softmax')(decoder)
        decoder = tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=-2), name='expand_dims')(decoder)

        self.decoder = tf.keras.Model(name='decoder', inputs=decoder_input, outputs=decoder)

        # recognition model
        self.recognition = tf.keras.Model(name=self.name,
                                          inputs=self.encoder.input,
                                          outputs=self.decoder(self.encoder.output))
