import tensorflow as tf

from models.components.convolution import OctConv2D
from models.components.loss import CTCLoss
from models.components.optimizer import NormalizedOptimizer
from models.components.processing import AdaptiveDenseReshape


class HandwritingRecognition(tf.keras.Model):
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

    def __init__(self,
                 image_shape,
                 lexical_shape,
                 **kwargs):
        """
        Initialize the handwriting recognition model with specified parameters.

        Parameters
        ----------
        image_shape : list or tuple
            Shape of the input image.
        lexical_shape : list or tuple
            Shape of the text sequences and vocabulary encoding.
        **kwargs : dict
            Additional keyword arguments for `tf.keras.Model`.
        """

        super().__init__(**kwargs)

        self.image_shape = image_shape
        self.lexical_shape = lexical_shape

        self.build_model()

        if hasattr(self, 'model'):
            self.summary = self.model.summary
            self.call = self.model.call

    def get_config(self):
        """
        Retrieves the configuration of the model for serialization.

        Returns
        -------
        dict
            A dictionary containing the configuration of the model.
        """

        config = super().get_config()

        config.update({
            'image_shape': self.image_shape,
            'lexical_shape': self.lexical_shape,
        })

        return config

    def compile(self, learning_rate=0.001):
        """
        Configure the sub-models for training.

        This method sets up the optimizers, loss functions, and metrics for the model.

        Parameters
        ----------
        learning_rate : float, optional
            The learning rate for the optimizer.
        """

        optimizer = NormalizedOptimizer(
            tf.keras.optimizers.RMSprop(learning_rate=learning_rate))

        super().compile(optimizer=optimizer, loss=CTCLoss(), run_eagerly=False)

    def build_model(self):
        """
        Initializes and builds the neural network model.

        This method sets up the architecture of the model by defining layers, their connections,
            and configurations. It is typically called in the constructor to create the model structure.
        """

        inputs = tf.keras.Input(shape=self.image_shape)

        high, low = OctConv2D(alpha=0.25, filters=16)([inputs, tf.keras.layers.AveragePooling2D(2)(inputs)])

        high = tf.keras.layers.BatchNormalization()(high)
        low = tf.keras.layers.BatchNormalization()(low)
        high = tf.keras.layers.LeakyReLU(alpha=0.01)(high)
        low = tf.keras.layers.LeakyReLU(alpha=0.01)(low)
        high = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(high)
        low = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(low)

        high, low = OctConv2D(alpha=0.25, filters=32)([high, low])

        high = tf.keras.layers.BatchNormalization()(high)
        low = tf.keras.layers.BatchNormalization()(low)
        high = tf.keras.layers.LeakyReLU(alpha=0.01)(high)
        low = tf.keras.layers.LeakyReLU(alpha=0.01)(low)
        high = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(high)
        low = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(low)

        high = tf.keras.layers.Dropout(rate=0.2)(high)
        low = tf.keras.layers.Dropout(rate=0.2)(low)

        high = tf.keras.layers.Conv2D(filters=48, kernel_size=(3, 3), strides=(1, 1), padding='same')(high)
        low = tf.keras.layers.Conv2D(filters=48, kernel_size=(3, 3), strides=(1, 1), padding='same')(low)

        high = tf.keras.layers.BatchNormalization()(high)
        low = tf.keras.layers.BatchNormalization()(low)
        high = tf.keras.layers.LeakyReLU(alpha=0.01)(high)
        low = tf.keras.layers.LeakyReLU(alpha=0.01)(low)
        high = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(high)
        low = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(low)

        high = tf.keras.layers.Dropout(rate=0.2)(high)
        low = tf.keras.layers.Dropout(rate=0.2)(low)

        high = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same')(high)
        low = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same')(low)

        high = tf.keras.layers.BatchNormalization()(high)
        low = tf.keras.layers.BatchNormalization()(low)
        high = tf.keras.layers.LeakyReLU(alpha=0.01)(high)
        low = tf.keras.layers.LeakyReLU(alpha=0.01)(low)

        high = tf.keras.layers.Dropout(rate=0.2)(high)
        low = tf.keras.layers.Dropout(rate=0.2)(low)

        high = tf.keras.layers.Conv2D(filters=80, kernel_size=(3, 3), strides=(1, 1), padding='same')(high)
        low = tf.keras.layers.Conv2D(filters=80, kernel_size=(3, 3), strides=(1, 1), padding='same')(low)

        high = tf.keras.layers.BatchNormalization()(high)
        low = tf.keras.layers.BatchNormalization()(low)
        high = tf.keras.layers.LeakyReLU(alpha=0.01)(high)
        low = tf.keras.layers.LeakyReLU(alpha=0.01)(low)

        high, low = OctConv2D(alpha=0.25, filters=80)([high, low])

        high = tf.keras.layers.BatchNormalization()(high)
        high = tf.keras.layers.Activation('relu')(high)

        low = tf.keras.layers.BatchNormalization()(low)
        low = tf.keras.layers.Activation('relu')(low)

        high_to_high = tf.keras.layers.Conv2D(80, 3, padding='same')(high)
        low_to_high = tf.keras.layers.Conv2D(80, 3, padding='same')(low)

        low_to_high = tf.keras.layers.Lambda(lambda x: tf.tile(x, [1, 2, 2, 1]), name='tile')(low_to_high)

        octconv = tf.keras.layers.Add()([high_to_high, low_to_high])
        octconv = tf.keras.layers.BatchNormalization()(octconv)
        octconv = tf.keras.layers.Activation('relu')(octconv)

        octconv = AdaptiveDenseReshape(target_shape=self.lexical_shape)(octconv)
        blstm = tf.keras.layers.Reshape(target_shape=(-1, octconv.get_shape()[-1]))(octconv)

        blstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(256, return_sequences=True, dropout=0.5))(blstm)
        blstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(256, return_sequences=True, dropout=0.5))(blstm)
        blstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(256, return_sequences=True, dropout=0.5))(blstm)
        blstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(256, return_sequences=True, dropout=0.5))(blstm)
        blstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(256, return_sequences=True, dropout=0.5))(blstm)

        blstm = tf.keras.layers.Dropout(rate=0.5)(blstm)
        dense = tf.keras.layers.Dense(units=self.lexical_shape[-1], activation='softmax')(blstm)

        outputs = tf.keras.layers.Reshape(target_shape=self.lexical_shape)(dense)

        self.model = tf.keras.Model(inputs=inputs, outputs=outputs, name=self.name)
