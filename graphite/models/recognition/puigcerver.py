import tensorflow as tf

from graphite.models.components.common import BaseRecognitionModel


class RecognitionModel(BaseRecognitionModel):
    """
    TensorFlow model for multilingual handwriting recognition using CNNs and BiLSTMs.
    It's based on traditional deep learning methods for offline handwriting recognition (CRNN).

    References
    ----------
    Are multidimensional recurrent layers really necessary for handwritten text recognition?
        https://ieeexplore.ieee.org/document/8269951
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

        self.optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)

    def build_model(self):
        """
        Builds the neural network model.

        This method sets up the architecture of the model by defining layers, their connections,
            and configurations. It is typically called in the constructor to create the model structure.
        """

        # encoder model
        encoder_input = tf.keras.Input(shape=self.image_shape)

        encoder = tf.keras.layers.Conv2D(filters=16,
                                         kernel_size=(3, 3),
                                         strides=(1, 1),
                                         padding='same')(encoder_input)

        encoder = tf.keras.layers.BatchNormalization()(encoder)
        encoder = tf.keras.layers.LeakyReLU(alpha=0.01)(encoder)
        encoder = tf.keras.layers.MaxPooling2D(pool_size=(2, 1), strides=(2, 1), padding='valid')(encoder)

        encoder = tf.keras.layers.Conv2D(filters=32,
                                         kernel_size=(3, 3),
                                         strides=(1, 1),
                                         padding='same')(encoder)

        encoder = tf.keras.layers.BatchNormalization()(encoder)
        encoder = tf.keras.layers.LeakyReLU(alpha=0.01)(encoder)
        encoder = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(encoder)
        encoder = tf.keras.layers.Dropout(rate=0.2)(encoder)

        encoder = tf.keras.layers.Conv2D(filters=48,
                                         kernel_size=(3, 3),
                                         strides=(1, 1),
                                         padding='same')(encoder)

        encoder = tf.keras.layers.BatchNormalization()(encoder)
        encoder = tf.keras.layers.LeakyReLU(alpha=0.01)(encoder)
        encoder = tf.keras.layers.MaxPooling2D(pool_size=(1, 2), strides=(1, 2), padding='valid')(encoder)
        encoder = tf.keras.layers.Dropout(rate=0.2)(encoder)

        encoder = tf.keras.layers.Conv2D(filters=64,
                                         kernel_size=(3, 3),
                                         strides=(1, 1),
                                         padding='same')(encoder)

        encoder = tf.keras.layers.BatchNormalization()(encoder)
        encoder = tf.keras.layers.LeakyReLU(alpha=0.01)(encoder)
        encoder = tf.keras.layers.Dropout(rate=0.2)(encoder)

        encoder = tf.keras.layers.Conv2D(filters=80,
                                         kernel_size=(3, 3),
                                         strides=(1, 1),
                                         padding='same')(encoder)

        encoder = tf.keras.layers.BatchNormalization()(encoder)
        encoder = tf.keras.layers.LeakyReLU(alpha=0.01)(encoder)

        encoder = tf.keras.layers.Reshape(target_shape=(encoder.shape[1], -1))(encoder)

        self.encoder = tf.keras.Model(inputs=encoder_input, outputs=encoder, name='encoder')

        # decoder model
        decoder_input = tf.keras.Input(shape=encoder.shape[1:])

        decoder = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(256, return_sequences=True, dropout=0.5))(decoder_input)

        decoder = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(256, return_sequences=True, dropout=0.5))(decoder)

        decoder = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(256, return_sequences=True, dropout=0.5))(decoder)

        decoder = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(256, return_sequences=True, dropout=0.5))(decoder)

        decoder = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(256, return_sequences=True, dropout=0.5))(decoder)

        decoder = tf.keras.layers.Dropout(rate=0.5)(decoder)
        decoder = tf.keras.layers.Dense(units=self.lexical_shape[-1], activation='softmax')(decoder)

        decoder = tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=-2), name='expand_dims')(decoder)

        self.decoder = tf.keras.Model(inputs=decoder_input, outputs=decoder, name='decoder')

        # recognition model
        decoder_output = self.decoder(self.encoder.output)
        self.recognition = tf.keras.Model(inputs=encoder_input, outputs=decoder_output, name=self.name)
