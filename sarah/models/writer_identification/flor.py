import tensorflow as tf

from sarah.models.components.base import BaseWriterIdentificationModel
from sarah.models.components.layers import GatedResidualConv2D


class WriterIdentificationModel(BaseWriterIdentificationModel):
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

        super().compile(run_eagerly=False)

        if learning_rate is None:
            learning_rate = 1e-4

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, beta_1=0.5, beta_2=0.999)

    def build_model(self):
        """
        Builds the model architecture.
        """

        feats = []

        # encoder model
        encoder_input = tf.keras.Input(shape=self.image_shape)

        encoder = tf.keras.layers.Conv2D(filters=8, kernel_size=3, padding='same')(encoder_input)
        encoder = tf.keras.layers.GroupNormalization(groups=-1)(encoder)
        encoder = tf.keras.layers.Activation(activation='swish')(encoder)

        encoder = GatedResidualConv2D()(encoder)

        encoder = tf.keras.layers.Conv2D(filters=16, kernel_size=3, padding='same')(encoder)
        encoder = tf.keras.layers.GroupNormalization(groups=-1)(encoder)
        encoder = tf.keras.layers.Activation(activation='swish')(encoder)
        encoder = tf.keras.layers.MaxPooling2D(pool_size=(2, 4), strides=(2, 4))(encoder)

        encoder = GatedResidualConv2D()(encoder)

        encoder = tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding='same')(encoder)
        encoder = tf.keras.layers.GroupNormalization(groups=-1)(encoder)
        encoder = tf.keras.layers.Activation(activation='swish')(encoder)
        encoder = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(encoder)

        encoder = GatedResidualConv2D(dropout=0.1)(encoder)
        encoder = tf.keras.layers.Dropout(rate=0.1)(encoder)

        encoder = tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding='same')(encoder)
        encoder = tf.keras.layers.GroupNormalization(groups=-1)(encoder)
        encoder = tf.keras.layers.Activation(activation='swish')(encoder)
        encoder = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(encoder)

        encoder = GatedResidualConv2D(dropout=0.1)(encoder)
        encoder = tf.keras.layers.Dropout(rate=0.1)(encoder)

        encoder = tf.keras.layers.Conv2D(filters=96, kernel_size=3, padding='same')(encoder)
        encoder = tf.keras.layers.GroupNormalization(groups=-1)(encoder)
        encoder = tf.keras.layers.Activation(activation='swish')(encoder)
        encoder = tf.keras.layers.MaxPooling2D(pool_size=(2, 4), strides=(2, 4))(encoder)
        feats.append(encoder)

        encoder = GatedResidualConv2D(dropout=0.1)(encoder)
        encoder = tf.keras.layers.Dropout(rate=0.1)(encoder)

        encoder = tf.keras.layers.Conv2D(filters=112, kernel_size=3, padding='same')(encoder)
        encoder = tf.keras.layers.GroupNormalization(groups=-1)(encoder)
        encoder = tf.keras.layers.Activation(activation='swish')(encoder)
        encoder = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(encoder)
        feats.append(encoder)

        encoder = GatedResidualConv2D(dropout=0.1)(encoder)
        encoder = tf.keras.layers.Dropout(rate=0.1)(encoder)

        encoder = tf.keras.layers.Conv2D(filters=128, kernel_size=3, padding='same')(encoder)
        encoder = tf.keras.layers.GroupNormalization(groups=-1)(encoder)
        encoder = tf.keras.layers.Activation(activation='swish')(encoder)
        encoder = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(encoder)
        feats.append(encoder)

        encoder = tf.keras.layers.Flatten()(encoder)
        outputs = [encoder, feats] if self.return_features else encoder

        self.encoder = tf.keras.Model(name='writer_encoder', inputs=encoder_input, outputs=outputs)

        # decoder model
        decoder_input = tf.keras.Input(shape=encoder.shape[1:])

        decoder = tf.keras.layers.Dense(units=256)(decoder_input)
        decoder = tf.keras.layers.Activation(activation='swish')(decoder)

        decoder = tf.keras.layers.Dense(units=256)(decoder)
        decoder = tf.keras.layers.Activation(activation='swish')(decoder)

        decoder = tf.keras.layers.Dense(units=self.writers_shape[0])(decoder)

        self.decoder = tf.keras.Model(name='writer_decoder', inputs=decoder_input, outputs=decoder)

        # writer identification model
        if self.return_features:
            encoder_output, features = self.encoder(encoder_input)
            outputs = [self.decoder(encoder_output), features]
        else:
            outputs = self.decoder(self.encoder(encoder_input))

        self.writer_identification = tf.keras.Model(name=self.name,
                                                    inputs=self.encoder.input,
                                                    outputs=outputs)
