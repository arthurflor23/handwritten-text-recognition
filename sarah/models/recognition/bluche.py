import tensorflow as tf

from sarah.models.components.base import BaseRecognitionModel
from sarah.models.components.layers import GatedConv2D


class RecognitionModel(BaseRecognitionModel):
    """
    TensorFlow model for multilingual handwriting recognition using CNNs and BiLSTMs.
    Features gated convolutional layers for enhanced feature extraction.

    References
    ----------
    Gated convolutional recurrent neural networks for multilingual handwriting recognition
        https://ieeexplore.ieee.org/document/8270042
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
            learning_rate = 4e-4

        self.optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate, rho=0.9, epsilon=1e-8)

    def build_model(self):
        """
        Builds the neural network model.

        This method sets up the architecture of the model by defining layers, their connections,
            and configurations. It is typically called in the constructor to create the model structure.
        """

        # encoder model
        encoder_input = tf.keras.Input(shape=self.image_shape)

        encoder = tf.keras.layers.Reshape(target_shape=(512, 64, -1))(encoder_input)

        encoder = tf.keras.layers.Conv2D(filters=8,
                                         kernel_size=(3, 3),
                                         strides=(1, 1),
                                         padding='same',
                                         activation='tanh')(encoder)

        encoder = tf.keras.layers.Conv2D(filters=16,
                                         kernel_size=(2, 4),
                                         strides=(2, 4),
                                         padding='same',
                                         activation='tanh')(encoder)

        encoder = GatedConv2D()(encoder)

        encoder = tf.keras.layers.Conv2D(filters=32,
                                         kernel_size=(3, 3),
                                         strides=(1, 1),
                                         padding='same',
                                         activation='tanh')(encoder)

        encoder = GatedConv2D()(encoder)

        encoder = tf.keras.layers.Conv2D(filters=64,
                                         kernel_size=(2, 4),
                                         strides=(1, 4),
                                         padding='same',
                                         activation='tanh')(encoder)

        encoder = GatedConv2D()(encoder)

        encoder = tf.keras.layers.Conv2D(filters=128,
                                         kernel_size=(3, 3),
                                         strides=(1, 1),
                                         padding='same',
                                         activation='tanh')(encoder)

        encoder = tf.keras.layers.MaxPooling2D(pool_size=(1, 4), strides=(1, 4), padding='valid')(encoder)

        self.encoder = tf.keras.Model(name='encoder', inputs=encoder_input, outputs=encoder)

        # decoder model
        decoder_input = tf.keras.Input(shape=encoder.shape[1:])

        decoder = tf.keras.layers.Reshape(target_shape=(encoder.shape[1], -1))(decoder_input)

        decoder = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=128, return_sequences=True))(decoder)

        decoder = tf.keras.layers.Dense(units=128, activation='tanh')(decoder)

        decoder = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=128, return_sequences=True))(decoder)

        decoder = tf.keras.layers.Dense(units=self.lexical_shape[-1], activation='softmax')(decoder)
        decoder = tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=-2), name='expand_dims')(decoder)

        self.decoder = tf.keras.Model(name='decoder', inputs=decoder_input, outputs=decoder)

        # recognition model
        self.recognition = tf.keras.Model(name=self.name,
                                          inputs=self.encoder.input,
                                          outputs=self.decoder(self.encoder.output))
