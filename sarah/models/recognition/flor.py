import tensorflow as tf

from sarah.models.components.base import BaseRecognitionModel
from sarah.models.components.layers import GatedResidualConv2D
from sarah.models.components.layers import SelfAttentionDense


class RecognitionModel(BaseRecognitionModel):
    """
    References
    ----------
    A Robust Handwritten Recognition System for Learning on Different Data Restriction Scenarios
        https://www.sciencedirect.com/science/article/abs/pii/S0167865522001052

    HTR-Flor: A Deep Learning System for Offline Handwritten Text Recognition
        https://ieeexplore.ieee.org/document/9266005
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
            learning_rate = 1e-3

        self.optimizer = tf.keras.optimizers.AdamW(
            learning_rate=learning_rate, beta_1=0.5, beta_2=0.99, weight_decay=0.01, epsilon=1e-7)

    def build_model(self):
        """
        Builds the model architecture.
        """

        # encoder model
        encoder_input = tf.keras.Input(shape=self.image_shape)

        encoder = tf.keras.layers.Conv2D(filters=8, kernel_size=3, padding='same')(encoder_input)
        encoder = tf.keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-3)(encoder)
        encoder = tf.keras.layers.Activation(activation='swish')(encoder)

        encoder = GatedResidualConv2D()(encoder)

        encoder = tf.keras.layers.Conv2D(filters=16, kernel_size=3, padding='same')(encoder)
        encoder = tf.keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-3)(encoder)
        encoder = tf.keras.layers.Activation(activation='swish')(encoder)
        encoder = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(encoder)

        encoder = GatedResidualConv2D(dropout=0.1)(encoder)
        encoder = tf.keras.layers.Dropout(rate=0.1)(encoder)

        encoder = tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding='same')(encoder)
        encoder = tf.keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-3)(encoder)
        encoder = tf.keras.layers.Activation(activation='swish')(encoder)
        encoder = tf.keras.layers.MaxPooling2D(pool_size=(2, 1), strides=(2, 1))(encoder)

        encoder = GatedResidualConv2D(dropout=0.1)(encoder)
        encoder = tf.keras.layers.Dropout(rate=0.1)(encoder)

        encoder = tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding='same')(encoder)
        encoder = tf.keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-3)(encoder)
        encoder = tf.keras.layers.Activation(activation='swish')(encoder)
        encoder = tf.keras.layers.MaxPooling2D(pool_size=(2, 1), strides=(2, 1))(encoder)

        encoder = GatedResidualConv2D(dropout=0.1)(encoder)
        encoder = tf.keras.layers.Dropout(rate=0.1)(encoder)

        encoder = tf.keras.layers.Conv2D(filters=80, kernel_size=3, padding='same')(encoder)
        encoder = tf.keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-3)(encoder)
        encoder = tf.keras.layers.Activation(activation='swish')(encoder)
        encoder = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(encoder)

        encoder = GatedResidualConv2D(dropout=0.1)(encoder)
        encoder = tf.keras.layers.Dropout(rate=0.1)(encoder)

        encoder = tf.keras.layers.Conv2D(filters=112, kernel_size=3, padding='same')(encoder)
        encoder = tf.keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-3)(encoder)
        encoder = tf.keras.layers.Activation(activation='swish')(encoder)
        encoder = tf.keras.layers.MaxPooling2D(pool_size=(2, 1), strides=(2, 1))(encoder)

        encoder = GatedResidualConv2D(dropout=0.1)(encoder)
        encoder = tf.keras.layers.Dropout(rate=0.1)(encoder)

        encoder = tf.keras.layers.Conv2D(filters=128, kernel_size=3, padding='same')(encoder)
        encoder = tf.keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-3)(encoder)
        encoder = tf.keras.layers.Activation(activation='swish')(encoder)
        encoder = tf.keras.layers.MaxPooling2D(pool_size=(2, 1), strides=(2, 1))(encoder)

        self.encoder = tf.keras.Model(name='recognition_encoder', inputs=encoder_input, outputs=encoder)

        # decoder model
        decoder_input = tf.keras.Input(shape=encoder.shape[1:])

        decoder = SelfAttentionDense(k=1/8, h=1/2, pooling=True)(decoder_input)

        decoder = tf.keras.layers.Reshape(target_shape=(-1, decoder.shape[-1]))(decoder)

        for rate in [0.4, 0.6, 0.6]:
            forwards = tf.keras.layers.Dropout(rate=rate)(decoder)
            forwards = tf.keras.layers.LSTM(units=128, return_sequences=True, go_backwards=False)(forwards)

            backwards = tf.keras.layers.Dropout(rate=rate)(decoder)
            backwards = tf.keras.layers.LSTM(units=128, return_sequences=True, go_backwards=True)(backwards)

            decoder = tf.keras.layers.Concatenate(axis=-1)([forwards, tf.keras.ops.flip(backwards, axis=1)])

        decoder = tf.keras.layers.Reshape(target_shape=(encoder.shape[1], encoder.shape[2], -1))(decoder)

        decoder = tf.keras.layers.LayerNormalization(epsilon=1e-3)(decoder)
        decoder = tf.keras.layers.Dropout(rate=0.6)(decoder)
        decoder = tf.keras.layers.Dense(units=self.lexical_shape[-1])(decoder)

        self.decoder = tf.keras.Model(name='recognition_decoder', inputs=decoder_input, outputs=decoder)

        # recognition model
        self.recognition = tf.keras.Model(name=self.name,
                                          inputs=self.encoder.input,
                                          outputs=self.decoder(self.encoder(encoder_input)))
