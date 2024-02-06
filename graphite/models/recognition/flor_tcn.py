import tensorflow as tf

from graphite.models.components.common import BaseRecognitionModel
from graphite.models.components.layers import GatedConv2D
from graphite.models.components.layers import MaskPadding
from graphite.models.components.layers import SelfAttention
from graphite.models.components.layers import TemporalConvolutional
from graphite.models.components.optimizers import NormalizedOptimizer


class RecognitionModel(BaseRecognitionModel):
    """
    References
    ----------
    A Robust Handwritten Recognition System for Learning on Different Data Restriction Scenarios
        https://www.sciencedirect.com/science/article/abs/pii/S0167865522001052

    Attention Is All You Need
        https://arxiv.org/abs/1706.03762

    Batch Renormalization: Towards Reducing Minibatch Dependence in Batch-Normalized Models
        https://arxiv.org/abs/1702.03275

    Block-Normalized Gradient Method: An Empirical Study for Training Deep Neural Network
        https://arxiv.org/abs/1707.04822

    Exact solutions to the nonlinear dynamics of learning in deep linear neural networks
        https://arxiv.org/abs/1312.6120

    HTR-Flor: A Deep Learning System for Offline Handwritten Text Recognition
        https://ieeexplore.ieee.org/document/9266005

    Temporal Convolutional Networks for Action Segmentation and Detection
        https://arxiv.org/abs/1611.05267
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

        self.optimizer = NormalizedOptimizer(
            tf.keras.optimizers.AdamW(learning_rate=learning_rate, weight_decay=0.1))

    def build_model(self):
        """
        Builds the neural network model.

        This method sets up the architecture of the model by defining layers, their connections,
            and configurations. It is typically called in the constructor to create the model structure.
        """

        initializer = tf.keras.initializers.random_normal(stddev=0.02)

        # encoder model
        encoder_input = tf.keras.Input(shape=self.image_shape)

        encoder = tf.keras.layers.Conv2D(filters=16,
                                         kernel_size=(3, 3),
                                         strides=(1, 1),
                                         padding='same',
                                         kernel_initializer=initializer)(encoder_input)

        encoder = tf.keras.layers.PReLU(shared_axes=[1, 2])(encoder)
        encoder = tf.keras.layers.BatchNormalization(renorm=True)(encoder)

        encoder = GatedConv2D(filters=16,
                              kernel_size=(3, 3),
                              strides=(1, 1),
                              padding='same',
                              kernel_initializer=initializer)(encoder)

        encoder = tf.keras.layers.Conv2D(filters=32,
                                         kernel_size=(2, 4),
                                         strides=(2, 4),
                                         padding='same',
                                         kernel_initializer=initializer)(encoder)

        encoder = tf.keras.layers.PReLU(shared_axes=[1, 2])(encoder)
        encoder = tf.keras.layers.BatchNormalization(renorm=True)(encoder)

        encoder = GatedConv2D(filters=32,
                              kernel_size=(3, 3),
                              strides=(1, 1),
                              padding='same',
                              kernel_initializer=initializer)(encoder)
        encoder = tf.keras.layers.Dropout(rate=0.2)(encoder)

        encoder = tf.keras.layers.Conv2D(filters=32,
                                         kernel_size=(3, 3),
                                         strides=(1, 1),
                                         padding='same',
                                         kernel_initializer=initializer)(encoder)

        encoder = tf.keras.layers.PReLU(shared_axes=[1, 2])(encoder)
        encoder = tf.keras.layers.BatchNormalization(renorm=True)(encoder)

        encoder = GatedConv2D(filters=32,
                              kernel_size=(3, 3),
                              strides=(1, 1),
                              padding='same',
                              kernel_initializer=initializer)(encoder)
        encoder = tf.keras.layers.Dropout(rate=0.1)(encoder)

        encoder = tf.keras.layers.Conv2D(filters=64,
                                         kernel_size=(2, 4),
                                         strides=(2, 4),
                                         padding='same',
                                         kernel_initializer=initializer)(encoder)

        encoder = tf.keras.layers.PReLU(shared_axes=[1, 2])(encoder)
        encoder = tf.keras.layers.BatchNormalization(renorm=True)(encoder)

        encoder = SelfAttention()(encoder)
        encoder = tf.keras.layers.Dropout(rate=0.2)(encoder)

        encoder = tf.keras.layers.Conv2D(filters=64,
                                         kernel_size=(3, 3),
                                         strides=(1, 1),
                                         padding='same',
                                         kernel_initializer=initializer)(encoder)

        encoder = tf.keras.layers.PReLU(shared_axes=[1, 2])(encoder)
        encoder = tf.keras.layers.BatchNormalization(renorm=True)(encoder)

        encoder = SelfAttention()(encoder)
        encoder = tf.keras.layers.Dropout(rate=0.1)(encoder)

        encoder = tf.keras.layers.Conv2D(filters=128,
                                         kernel_size=(3, 3),
                                         strides=(1, 1),
                                         padding='same',
                                         kernel_initializer=initializer)(encoder)

        encoder = tf.keras.layers.PReLU(shared_axes=[1, 2])(encoder)
        encoder = tf.keras.layers.BatchNormalization(renorm=True)(encoder)

        encoder = SelfAttention()(encoder)

        encoder = MaskPadding()([encoder_input, encoder])
        encoder = tf.keras.layers.Reshape(target_shape=(encoder.shape[1], -1))(encoder)
        encoder = tf.keras.layers.Dense(units=256, kernel_initializer=initializer)(encoder)

        self.encoder = tf.keras.Model(inputs=encoder_input, outputs=encoder, name='encoder')

        # decoder model
        decoder_input = tf.keras.Input(shape=encoder.shape[1:])

        decoder = TemporalConvolutional(filters=64,
                                        nb_stacks=3,
                                        kernel_size=3,
                                        dilations=(1, 2, 4, 8),
                                        padding='same',
                                        activation='prelu',
                                        kernel_initializer=initializer,
                                        batch_norm=True,
                                        return_sequences=True,
                                        dropout=0.2)(decoder_input)

        decoder = tf.keras.layers.Dense(units=self.lexical_shape[-1],
                                        kernel_initializer=initializer,
                                        activation='softmax')(decoder)

        decoder = tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=-2), name='expand_dims')(decoder)

        self.decoder = tf.keras.Model(inputs=decoder_input, outputs=decoder, name='decoder')

        # recognition model
        decoder_output = self.decoder(self.encoder.output)
        self.recognition = tf.keras.Model(inputs=encoder_input, outputs=decoder_output, name=self.name)
