import tensorflow as tf

from sarah.models.components.base import BaseRecognitionModel
from sarah.models.components.layers import GatedConv2DDual


class RecognitionModel(BaseRecognitionModel):
    """
    TensorFlow model for multilingual handwriting recognition using CNNs and GRUs.
    Features gated convolutional layers for enhanced feature extraction.

    References
    ----------
    HTR-Flor: A Deep Learning System for Offline Handwritten Text Recognition
        https://ieeexplore.ieee.org/document/9266005

    A Robust Handwritten Recognition System for Learning on Different Data Restriction Scenarios
        https://www.sciencedirect.com/science/article/abs/pii/S0167865522001052
    """

    def compile(self, learning_rate=None):
        """
        Compiles the model.

        Parameters
        ----------
        learning_rate : float, optional
            Optimizer learning rate.
        """

        super().compile(run_eagerly=False)

        if learning_rate is None:
            learning_rate = 1e-3

        self.optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate, rho=0.9)

    def build_model(self):
        """
        Builds the model architecture.
        """

        # encoder model
        encoder_input = tf.keras.Input(shape=self.image_shape)
        encoder = tf.keras.layers.Lambda(lambda x: tf.transpose(x, perm=(0, 2, 1, 3)), name='perm1')(encoder_input)

        encoder = tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3), strides=(2, 2), padding='same')(encoder)
        encoder = tf.keras.layers.PReLU(shared_axes=[1, 2])(encoder)
        encoder = tf.keras.layers.BatchNormalization()(encoder)

        encoder = GatedConv2DDual()(encoder)

        encoder = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same')(encoder)
        encoder = tf.keras.layers.PReLU(shared_axes=[1, 2])(encoder)
        encoder = tf.keras.layers.BatchNormalization()(encoder)

        encoder = GatedConv2DDual()(encoder)

        encoder = tf.keras.layers.Conv2D(filters=40, kernel_size=(2, 4), strides=(2, 4), padding='same')(encoder)
        encoder = tf.keras.layers.PReLU(shared_axes=[1, 2])(encoder)
        encoder = tf.keras.layers.BatchNormalization()(encoder)

        encoder = GatedConv2DDual()(encoder)
        encoder = tf.keras.layers.Dropout(rate=0.2)(encoder)

        encoder = tf.keras.layers.Conv2D(filters=48, kernel_size=(3, 3), strides=(1, 1), padding='same')(encoder)
        encoder = tf.keras.layers.PReLU(shared_axes=[1, 2])(encoder)
        encoder = tf.keras.layers.BatchNormalization()(encoder)

        encoder = GatedConv2DDual()(encoder)
        encoder = tf.keras.layers.Dropout(rate=0.2)(encoder)

        encoder = tf.keras.layers.Conv2D(filters=56, kernel_size=(2, 4), strides=(1, 2), padding='same')(encoder)
        encoder = tf.keras.layers.PReLU(shared_axes=[1, 2])(encoder)
        encoder = tf.keras.layers.BatchNormalization()(encoder)

        encoder = GatedConv2DDual()(encoder)
        encoder = tf.keras.layers.Dropout(rate=0.2)(encoder)

        encoder = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same')(encoder)
        encoder = tf.keras.layers.PReLU(shared_axes=[1, 2])(encoder)
        encoder = tf.keras.layers.BatchNormalization()(encoder)

        encoder = tf.keras.layers.MaxPooling2D(pool_size=(1, 2), strides=(1, 2), padding='valid')(encoder)
        encoder = tf.keras.layers.Reshape(target_shape=(encoder.shape[1], encoder.shape[2] // 2, -1))(encoder)

        self.encoder = tf.keras.Model(name='recognition_encoder', inputs=encoder_input, outputs=encoder)

        # decoder model
        decoder_input = tf.keras.Input(shape=encoder.shape[1:])
        decoder = tf.keras.layers.Reshape(target_shape=(-1, encoder.shape[-1]))(decoder_input)

        forwards = tf.keras.layers.Dropout(rate=0.5)(decoder)
        backwards = tf.keras.layers.Dropout(rate=0.5)(decoder)
        forwards = tf.keras.layers.GRU(units=128, return_sequences=True, go_backwards=False)(forwards)
        backwards = tf.keras.layers.GRU(units=128, return_sequences=True, go_backwards=True)(backwards)
        decoder = tf.keras.layers.Concatenate(axis=-1)([forwards, tf.keras.ops.flip(backwards, axis=1)])

        decoder = tf.keras.layers.Dense(units=256)(decoder)

        forwards = tf.keras.layers.Dropout(rate=0.5)(decoder)
        backwards = tf.keras.layers.Dropout(rate=0.5)(decoder)
        forwards = tf.keras.layers.GRU(units=128, return_sequences=True, go_backwards=False)(forwards)
        backwards = tf.keras.layers.GRU(units=128, return_sequences=True, go_backwards=True)(backwards)
        decoder = tf.keras.layers.Concatenate(axis=-1)([forwards, tf.keras.ops.flip(backwards, axis=1)])

        decoder = tf.keras.layers.Dense(units=self.lexical_shape[-1])(decoder)

        decoder = tf.keras.layers.Reshape(target_shape=encoder.shape[1:-1] + self.lexical_shape[-1:])(decoder)
        decoder = tf.keras.layers.Lambda(lambda x: tf.transpose(x, perm=(0, 2, 1, 3)), name='perm2')(decoder)

        self.decoder = tf.keras.Model(name='recognition_decoder', inputs=decoder_input, outputs=decoder)

        # recognition model
        self.recognition = tf.keras.Model(name=self.name,
                                          inputs=self.encoder.input,
                                          outputs=self.decoder(self.encoder(encoder_input)))
