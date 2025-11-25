import tensorflow as tf

from sarah.models.components.base import BaseRecognitionModel
from sarah.models.components.layers import OctaveConv2D


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
        Compiles the model.

        Parameters
        ----------
        learning_rate : float, optional
            Optimizer learning rate.
        """

        super().compile(run_eagerly=False, jit_compile=False)

        if learning_rate is None:
            learning_rate = 3e-4

        self.optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate, rho=0.9)

    def build_model(self):
        """
        Builds the model architecture.
        """

        # encoder model
        encoder_input = tf.keras.Input(shape=self.image_shape)
        encoder = tf.keras.layers.Lambda(lambda x: tf.transpose(x, perm=(0, 2, 1, 3)), name='perm1')(encoder_input)

        encoder = [encoder, tf.keras.layers.AveragePooling2D(pool_size=2)(encoder)]
        high, low = OctaveConv2D(alpha=0.25, filters=16)(encoder)

        high = tf.keras.layers.BatchNormalization()(high)
        low = tf.keras.layers.BatchNormalization()(low)
        high = tf.keras.layers.LeakyReLU(negative_slope=0.01)(high)
        low = tf.keras.layers.LeakyReLU(negative_slope=0.01)(low)
        high = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(high)
        low = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(low)

        high, low = OctaveConv2D(alpha=0.25, filters=32)([high, low])

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

        high, low = OctaveConv2D(alpha=0.25, filters=80)([high, low])

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

        encoder = tf.keras.layers.Reshape(target_shape=(encoder.shape[1], encoder.shape[2] // 16, -1))(encoder)

        self.encoder = tf.keras.Model(name='recognition_encoder', inputs=encoder_input, outputs=encoder)

        # decoder model
        decoder_input = tf.keras.Input(shape=encoder.shape[1:])
        decoder = tf.keras.layers.Reshape(target_shape=(-1, encoder.shape[-1]))(decoder_input)

        for _ in range(5):
            forwards = tf.keras.layers.Dropout(rate=0.5)(decoder)
            backwards = tf.keras.layers.Dropout(rate=0.5)(decoder)
            forwards = tf.keras.layers.LSTM(units=256, return_sequences=True, go_backwards=False)(forwards)
            backwards = tf.keras.layers.LSTM(units=256, return_sequences=True, go_backwards=True)(backwards)
            decoder = tf.keras.layers.Concatenate(axis=-1)([forwards, tf.keras.ops.flip(backwards, axis=1)])

        decoder = tf.keras.layers.Dropout(rate=0.5)(decoder)
        decoder = tf.keras.layers.Dense(units=self.lexical_shape[-1])(decoder)

        decoder = tf.keras.layers.Reshape(target_shape=encoder.shape[1:-1] + self.lexical_shape[-1:])(decoder)
        decoder = tf.keras.layers.Lambda(lambda x: tf.transpose(x, perm=(0, 2, 1, 3)), name='perm2')(decoder)

        self.decoder = tf.keras.Model(name='recognition_decoder', inputs=decoder_input, outputs=decoder)

        # recognition model
        self.recognition = tf.keras.Model(name=self.name,
                                          inputs=self.encoder.input,
                                          outputs=self.decoder(self.encoder(encoder_input)))
