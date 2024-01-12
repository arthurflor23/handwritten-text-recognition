import tensorflow as tf

from graphite.models.components.layers import OctConv2D
from graphite.models.components.models import BaseRecognitionModel
from graphite.models.components.optimizers import NormalizedOptimizer


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
            tf.keras.optimizers.RMSprop(learning_rate=learning_rate))

    def build_model(self):
        """
        Builds the neural network model.

        This method sets up the architecture of the model by defining layers, their connections,
            and configurations. It is typically called in the constructor to create the model structure.
        """

        inputs = tf.keras.Input(shape=self.image_shape)

        high, low = OctConv2D(alpha=0.25, filters=16)([inputs, tf.keras.layers.AveragePooling2D(2)(inputs)])

        high = tf.keras.layers.BatchNormalization(renorm=True)(high)
        low = tf.keras.layers.BatchNormalization(renorm=True)(low)
        high = tf.keras.layers.LeakyReLU(alpha=0.2)(high)
        low = tf.keras.layers.LeakyReLU(alpha=0.2)(low)
        high = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(high)
        low = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(low)

        high, low = OctConv2D(alpha=0.25, filters=32)([high, low])

        high = tf.keras.layers.BatchNormalization(renorm=True)(high)
        low = tf.keras.layers.BatchNormalization(renorm=True)(low)
        high = tf.keras.layers.LeakyReLU(alpha=0.2)(high)
        low = tf.keras.layers.LeakyReLU(alpha=0.2)(low)
        high = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(high)
        low = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(low)

        high = tf.keras.layers.Dropout(rate=0.2)(high)
        low = tf.keras.layers.Dropout(rate=0.2)(low)

        high = tf.keras.layers.Conv2D(filters=48, kernel_size=(3, 3), strides=(1, 1), padding='same')(high)
        low = tf.keras.layers.Conv2D(filters=48, kernel_size=(3, 3), strides=(1, 1), padding='same')(low)

        high = tf.keras.layers.BatchNormalization(renorm=True)(high)
        low = tf.keras.layers.BatchNormalization(renorm=True)(low)
        high = tf.keras.layers.LeakyReLU(alpha=0.2)(high)
        low = tf.keras.layers.LeakyReLU(alpha=0.2)(low)

        high = tf.keras.layers.Dropout(rate=0.2)(high)
        low = tf.keras.layers.Dropout(rate=0.2)(low)

        high = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same')(high)
        low = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same')(low)

        high = tf.keras.layers.BatchNormalization(renorm=True)(high)
        low = tf.keras.layers.BatchNormalization(renorm=True)(low)
        high = tf.keras.layers.LeakyReLU(alpha=0.2)(high)
        low = tf.keras.layers.LeakyReLU(alpha=0.2)(low)

        high = tf.keras.layers.Dropout(rate=0.2)(high)
        low = tf.keras.layers.Dropout(rate=0.2)(low)

        high = tf.keras.layers.Conv2D(filters=80, kernel_size=(3, 3), strides=(1, 1), padding='same')(high)
        low = tf.keras.layers.Conv2D(filters=80, kernel_size=(3, 3), strides=(1, 1), padding='same')(low)

        high = tf.keras.layers.BatchNormalization(renorm=True)(high)
        low = tf.keras.layers.BatchNormalization(renorm=True)(low)
        high = tf.keras.layers.LeakyReLU(alpha=0.2)(high)
        low = tf.keras.layers.LeakyReLU(alpha=0.2)(low)

        high, low = OctConv2D(alpha=0.25, filters=80)([high, low])

        high = tf.keras.layers.BatchNormalization(renorm=True)(high)
        high = tf.keras.layers.Activation('relu')(high)

        low = tf.keras.layers.BatchNormalization(renorm=True)(low)
        low = tf.keras.layers.Activation('relu')(low)

        high_to_high = tf.keras.layers.Conv2D(80, 3, padding='same')(high)
        low_to_high = tf.keras.layers.Conv2D(80, 3, padding='same')(low)

        low_to_high = tf.keras.layers.Lambda(lambda x: tf.tile(x, [1, 2, 2, 1]), name='tile')(low_to_high)

        octconv = tf.keras.layers.Add()([high_to_high, low_to_high])
        octconv = tf.keras.layers.BatchNormalization(renorm=True)(octconv)
        octconv = tf.keras.layers.Activation('relu')(octconv)

        blstm = tf.keras.layers.Reshape(target_shape=(octconv.get_shape()[1], -1))(octconv)

        blstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(256, return_sequences=True, dropout=0.5))(blstm)
        blstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(256, return_sequences=True, dropout=0.5))(blstm)
        blstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(256, return_sequences=True, dropout=0.5))(blstm)
        blstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(256, return_sequences=True, dropout=0.5))(blstm)
        blstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(256, return_sequences=True, dropout=0.5))(blstm)

        blstm = tf.keras.layers.Dropout(rate=0.5)(blstm)
        blstm = tf.keras.layers.Dense(units=self.lexical_shape[-1], activation='softmax')(blstm)

        outputs = tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=-2), name='expand_dims')(blstm)

        self.recognition = tf.keras.Model(inputs=inputs, outputs=outputs, name=self.name)
