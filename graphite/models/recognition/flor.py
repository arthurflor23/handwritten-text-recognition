import tensorflow as tf

from models.components.models import SynthesisRecognitionBaseModel
from models.components.convolution import GatedConv2D


class SynthesisRecognitionModel(SynthesisRecognitionBaseModel):
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

    def build_model(self):
        """
        Initializes and builds the neural network model.

        This method sets up the architecture of the model by defining layers, their connections,
            and configurations. It is typically called in the constructor to create the model structure.
        """

        inputs = tf.keras.Input(shape=self.image_shape)

        conv = tf.keras.layers.Conv2D(filters=16,
                                      kernel_size=(3, 3),
                                      strides=(2, 2),
                                      padding='same',
                                      kernel_initializer='he_uniform')(inputs)

        conv = tf.keras.layers.PReLU(shared_axes=[1, 2])(conv)
        conv = tf.keras.layers.BatchNormalization(renorm=True)(conv)

        conv = GatedConv2D(filters=16)(conv)

        conv = tf.keras.layers.Conv2D(filters=32,
                                      kernel_size=(3, 3),
                                      strides=(1, 1),
                                      padding='same',
                                      kernel_initializer='he_uniform')(conv)

        conv = tf.keras.layers.PReLU(shared_axes=[1, 2])(conv)
        conv = tf.keras.layers.BatchNormalization(renorm=True)(conv)

        conv = GatedConv2D(filters=32)(conv)

        conv = tf.keras.layers.Conv2D(filters=40,
                                      kernel_size=(2, 4),
                                      strides=(2, 4),
                                      padding='same',
                                      kernel_initializer='he_uniform')(conv)

        conv = tf.keras.layers.PReLU(shared_axes=[1, 2])(conv)
        conv = tf.keras.layers.BatchNormalization(renorm=True)(conv)

        conv = GatedConv2D(filters=40, kernel_constraint=tf.keras.constraints.MaxNorm(4, [0, 1, 2]))(conv)

        conv = tf.keras.layers.Dropout(rate=0.2)(conv)

        conv = tf.keras.layers.Conv2D(filters=48,
                                      kernel_size=(3, 3),
                                      strides=(1, 1),
                                      padding='same',
                                      kernel_initializer='he_uniform')(conv)

        conv = tf.keras.layers.PReLU(shared_axes=[1, 2])(conv)
        conv = tf.keras.layers.BatchNormalization(renorm=True)(conv)

        conv = GatedConv2D(filters=48, kernel_constraint=tf.keras.constraints.MaxNorm(4, [0, 1, 2]))(conv)

        conv = tf.keras.layers.Dropout(rate=0.2)(conv)

        conv = tf.keras.layers.Conv2D(filters=56,
                                      kernel_size=(2, 4),
                                      strides=(2, 4),
                                      padding='same',
                                      kernel_initializer='he_uniform')(conv)

        conv = tf.keras.layers.PReLU(shared_axes=[1, 2])(conv)
        conv = tf.keras.layers.BatchNormalization(renorm=True)(conv)

        conv = GatedConv2D(filters=56, kernel_constraint=tf.keras.constraints.MaxNorm(4, [0, 1, 2]))(conv)

        conv = tf.keras.layers.Dropout(rate=0.2)(conv)

        conv = tf.keras.layers.Conv2D(filters=64,
                                      kernel_size=(3, 3),
                                      strides=(1, 1),
                                      padding='same',
                                      kernel_initializer='he_uniform')(conv)

        conv = tf.keras.layers.PReLU(shared_axes=[1, 2])(conv)
        conv = tf.keras.layers.BatchNormalization(renorm=True)(conv)

        conv = tf.keras.layers.MaxPooling2D(pool_size=(1, 2), strides=(1, 2), padding='valid')(conv)

        bgru = tf.keras.layers.Reshape(target_shape=(conv.get_shape()[1], -1))(conv)

        bgru = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(128, return_sequences=True, dropout=0.5))(bgru)
        bgru = tf.keras.layers.Dense(units=256)(bgru)

        bgru = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(128, return_sequences=True, dropout=0.5))(bgru)
        bgru = tf.keras.layers.Dense(units=self.lexical_shape[-1], activation='softmax')(bgru)

        outputs = tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=1), name='expand_dims')(bgru)

        self.model = tf.keras.Model(inputs=inputs, outputs=outputs, name=self.name)
