import tensorflow as tf

from graphite.models.components.layers import GatedConv2D
from graphite.models.components.models import RecognitionBaseModel
from graphite.models.components.optimizers import NormalizedOptimizer


class RecognitionModel(RecognitionBaseModel):
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

        self.optimizer = NormalizedOptimizer(
            tf.keras.optimizers.RMSprop(learning_rate=learning_rate))

    def build_model(self):
        """
        Builds the neural network model.

        This method sets up the architecture of the model by defining layers, their connections,
            and configurations. It is typically called in the constructor to create the model structure.
        """

        inputs = tf.keras.Input(shape=self.image_shape)

        target_shape = (self.image_shape[0] // 2, self.image_shape[1] // 2, self.image_shape[2] * 4)
        conv = tf.keras.layers.Reshape(target_shape=target_shape)(inputs)

        conv = tf.keras.layers.Conv2D(filters=8,
                                      kernel_size=(3, 3),
                                      strides=(1, 1),
                                      padding='same',
                                      activation='tanh')(conv)

        conv = tf.keras.layers.Conv2D(filters=16,
                                      kernel_size=(2, 4),
                                      strides=(2, 4),
                                      padding='same',
                                      activation='tanh')(conv)

        conv = GatedConv2D(filters=16, fullgate=False)(conv)

        conv = tf.keras.layers.Conv2D(filters=32,
                                      kernel_size=(3, 3),
                                      strides=(1, 1),
                                      padding='same',
                                      activation='tanh')(conv)

        conv = GatedConv2D(filters=32, fullgate=False)(conv)

        conv = tf.keras.layers.Conv2D(filters=64,
                                      kernel_size=(2, 4),
                                      strides=(1, 4),
                                      padding='same',
                                      activation='tanh')(conv)

        conv = GatedConv2D(filters=64, fullgate=False)(conv)

        conv = tf.keras.layers.Conv2D(filters=128,
                                      kernel_size=(3, 3),
                                      strides=(1, 1),
                                      padding='same',
                                      activation='tanh')(conv)

        conv = tf.keras.layers.MaxPooling2D(pool_size=(1, 2), strides=(1, 2), padding='valid')(conv)

        blstm = tf.keras.layers.Reshape(target_shape=(conv.get_shape()[1], -1))(conv)

        blstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True))(blstm)
        blstm = tf.keras.layers.Dense(units=128, activation='tanh')(blstm)

        blstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True))(blstm)
        blstm = tf.keras.layers.Dense(units=self.lexical_shape[-1], activation='softmax')(blstm)

        outputs = tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=-2), name='expand_dims')(blstm)

        self.recognition = tf.keras.Model(inputs=inputs, outputs=outputs, name=self.name)
