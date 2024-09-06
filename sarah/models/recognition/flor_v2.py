import tensorflow as tf

from sarah.models.components.base import BaseRecognitionModel
from sarah.models.components.layers import Bidirectional
from sarah.models.components.layers import GatedConv2D
from sarah.models.components.layers import SelfAttention


class RecognitionModel(BaseRecognitionModel):
    """
    References
    ----------
    A Robust Handwritten Recognition System for Learning on Different Data Restriction Scenarios
        https://www.sciencedirect.com/science/article/abs/pii/S0167865522001052

    HTR-Flor: A Deep Learning System for Offline Handwritten Text Recognition
        https://ieeexplore.ieee.org/document/9266005

    Layer Normalization
        https://arxiv.org/abs/1607.06450

    Searching for Activation Functions (Swish: a Self-Gated Activation Function)
        https://arxiv.org/abs/1710.05941

    Self-Attention Generative Adversarial Networks
        https://arxiv.org/abs/1805.08318
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
            learning_rate = 1e-3

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, beta_1=0.5, beta_2=0.95, epsilon=1e-8)

    def build_model(self):
        """
        Builds the neural network model.

        This method sets up the architecture of the model by defining layers, their connections,
            and configurations. It is typically called in the constructor to create the model structure.
        """

        # encoder model
        encoder_input = tf.keras.Input(shape=self.image_shape)

        encoder = tf.keras.layers.Conv2D(filters=16, kernel_size=3, padding='same')(encoder_input)
        encoder = tf.keras.layers.BatchNormalization()(encoder)
        encoder = tf.keras.layers.Activation(activation='swish')(encoder)

        encoder = GatedConv2D(mode='residual')(encoder)

        encoder = tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding='same')(encoder)
        encoder = tf.keras.layers.BatchNormalization()(encoder)
        encoder = tf.keras.layers.Activation(activation='swish')(encoder)
        encoder = tf.keras.layers.MaxPooling2D(pool_size=2, strides=2)(encoder)

        encoder = GatedConv2D(mode='residual', dropout=0.1)(encoder)
        encoder = tf.keras.layers.Dropout(rate=0.1)(encoder)

        encoder = tf.keras.layers.Conv2D(filters=48, kernel_size=3, padding='same')(encoder)
        encoder = tf.keras.layers.BatchNormalization()(encoder)
        encoder = tf.keras.layers.Activation(activation='swish')(encoder)
        encoder = tf.keras.layers.MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(encoder)

        encoder = GatedConv2D(mode='residual', dropout=0.1)(encoder)
        encoder = tf.keras.layers.Dropout(rate=0.1)(encoder)

        encoder = tf.keras.layers.Conv2D(filters=72, kernel_size=3, padding='same')(encoder)
        encoder = tf.keras.layers.BatchNormalization()(encoder)
        encoder = tf.keras.layers.Activation(activation='swish')(encoder)
        encoder = tf.keras.layers.MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(encoder)

        encoder = GatedConv2D(mode='residual', dropout=0.1)(encoder)
        encoder = tf.keras.layers.Dropout(rate=0.1)(encoder)

        encoder = tf.keras.layers.Conv2D(filters=96, kernel_size=3, padding='same')(encoder)
        encoder = tf.keras.layers.BatchNormalization()(encoder)
        encoder = tf.keras.layers.Activation(activation='swish')(encoder)
        encoder = tf.keras.layers.MaxPooling2D(pool_size=2, strides=2)(encoder)

        encoder = SelfAttention(pooling=True, dropout=0.1)(encoder)
        encoder = tf.keras.layers.Dropout(rate=0.1)(encoder)

        encoder = tf.keras.layers.Conv2D(filters=112, kernel_size=3, padding='same')(encoder)
        encoder = tf.keras.layers.BatchNormalization()(encoder)
        encoder = tf.keras.layers.Activation(activation='swish')(encoder)
        encoder = tf.keras.layers.MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(encoder)

        encoder = SelfAttention(pooling=True, dropout=0.1)(encoder)
        encoder = tf.keras.layers.Dropout(rate=0.1)(encoder)

        encoder = tf.keras.layers.Conv2D(filters=128, kernel_size=3, padding='same')(encoder)
        encoder = tf.keras.layers.BatchNormalization()(encoder)
        encoder = tf.keras.layers.Activation(activation='swish')(encoder)
        encoder = tf.keras.layers.MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(encoder)

        encoder = SelfAttention(pooling=False)(encoder)

        self.encoder = tf.keras.Model(name='encoder', inputs=encoder_input, outputs=encoder)

        # decoder model
        decoder_input = tf.keras.Input(shape=encoder.shape[1:])
        decoder = tf.keras.layers.Reshape(target_shape=(-1, encoder.shape[-1]))(decoder_input)

        decoder = Bidirectional(tf.keras.layers.LSTM(units=128, return_sequences=True), dropout=0.5)(decoder)
        decoder = tf.keras.layers.LayerNormalization()(decoder)

        decoder = Bidirectional(tf.keras.layers.LSTM(units=128, return_sequences=True), dropout=0.5)(decoder)

        decoder = tf.keras.layers.LayerNormalization()(decoder)
        decoder = Bidirectional(tf.keras.layers.LSTM(units=128, return_sequences=True), dropout=0.5)(decoder)

        decoder = tf.keras.layers.Dropout(rate=0.5)(decoder)

        decoder = tf.keras.layers.Dense(units=self.lexical_shape[-1])(decoder)
        decoder = tf.keras.layers.Activation(activation='softmax')(decoder)

        decoder = tf.keras.layers.Reshape(target_shape=encoder.shape[1:-1] + self.lexical_shape[-1:])(decoder)
        self.decoder = tf.keras.Model(name='decoder', inputs=decoder_input, outputs=decoder)

        # recognition model
        self.recognition = tf.keras.Model(name=self.name,
                                          inputs=self.encoder.input,
                                          outputs=self.decoder(self.encoder(encoder_input)))
