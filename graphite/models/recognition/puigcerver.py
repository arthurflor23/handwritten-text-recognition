import tensorflow as tf

from graphite.models.components.models import RecognitionBaseModel
from graphite.models.components.optimizers import NormalizedOptimizer


class RecognitionModel(RecognitionBaseModel):
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

        self.optimizer = NormalizedOptimizer(
            tf.keras.optimizers.RMSProp(learning_rate=learning_rate))

    def build_model(self):
        """
        Builds the neural network model.

        This method sets up the architecture of the model by defining layers, their connections,
            and configurations. It is typically called in the constructor to create the model structure.
        """

        inputs = tf.keras.Input(shape=self.image_shape)

        conv = tf.keras.layers.Conv2D(filters=16,
                                      kernel_size=(3, 3),
                                      strides=(1, 1),
                                      padding='same')(inputs)

        conv = tf.keras.layers.BatchNormalization(renorm=True)(conv)
        conv = tf.keras.layers.LeakyReLU(alpha=0.2)(conv)
        conv = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(conv)

        conv = tf.keras.layers.Conv2D(filters=32,
                                      kernel_size=(3, 3),
                                      strides=(1, 1),
                                      padding='same')(conv)

        conv = tf.keras.layers.BatchNormalization(renorm=True)(conv)
        conv = tf.keras.layers.LeakyReLU(alpha=0.2)(conv)
        conv = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(conv)
        conv = tf.keras.layers.Dropout(rate=0.2)(conv)

        conv = tf.keras.layers.Conv2D(filters=48,
                                      kernel_size=(3, 3),
                                      strides=(1, 1),
                                      padding='same')(conv)

        conv = tf.keras.layers.BatchNormalization(renorm=True)(conv)
        conv = tf.keras.layers.LeakyReLU(alpha=0.2)(conv)
        conv = tf.keras.layers.Dropout(rate=0.2)(conv)

        conv = tf.keras.layers.Conv2D(filters=64,
                                      kernel_size=(3, 3),
                                      strides=(1, 1),
                                      padding='same')(conv)

        conv = tf.keras.layers.BatchNormalization(renorm=True)(conv)
        conv = tf.keras.layers.LeakyReLU(alpha=0.2)(conv)
        conv = tf.keras.layers.Dropout(rate=0.2)(conv)

        conv = tf.keras.layers.Conv2D(filters=80,
                                      kernel_size=(3, 3),
                                      strides=(1, 1),
                                      padding='same')(conv)

        conv = tf.keras.layers.BatchNormalization(renorm=True)(conv)
        conv = tf.keras.layers.LeakyReLU(alpha=0.2)(conv)

        blstm = tf.keras.layers.Reshape(target_shape=(conv.get_shape()[1], -1))(conv)

        blstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(256, return_sequences=True, dropout=0.5))(blstm)
        blstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(256, return_sequences=True, dropout=0.5))(blstm)
        blstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(256, return_sequences=True, dropout=0.5))(blstm)
        blstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(256, return_sequences=True, dropout=0.5))(blstm)
        blstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(256, return_sequences=True, dropout=0.5))(blstm)

        blstm = tf.keras.layers.Dropout(rate=0.5)(blstm)
        blstm = tf.keras.layers.Dense(units=self.lexical_shape[-1], activation='softmax')(blstm)

        outputs = tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=1), name='expand_dims')(blstm)

        self.recognition = tf.keras.Model(inputs=inputs, outputs=outputs, name=self.name)
