import tensorflow as tf

from models.components.convolution import GatedConv2D
from models.components.loss import CTCLoss
from models.components.optimizer import NormalizedOptimizer
from models.components.processing import AdaptiveDenseReshape


class HandwritingRecognition(tf.keras.Model):
    """
    TensorFlow model for multilingual handwriting recognition using CNNs and BiLSTMs.
    Features gated convolutional layers for enhanced feature extraction.

    References
    ----------
    Gated convolutional recurrent neural networks for multilingual handwriting recognition
        https://ieeexplore.ieee.org/document/8270042
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

        conv = GatedConv2D(filters=16,
                           kernel_size=(3, 3),
                           strides=(1, 1),
                           padding='same',
                           use_partial_gating=True)(conv)

        conv = tf.keras.layers.Conv2D(filters=32,
                                      kernel_size=(3, 3),
                                      strides=(1, 1),
                                      padding='same',
                                      activation='tanh')(conv)

        conv = GatedConv2D(filters=32,
                           kernel_size=(3, 3),
                           strides=(1, 1),
                           padding='same',
                           use_partial_gating=True)(conv)

        conv = tf.keras.layers.Conv2D(filters=64,
                                      kernel_size=(2, 4),
                                      strides=(2, 4),
                                      padding='same',
                                      activation='tanh')(conv)

        conv = GatedConv2D(filters=64,
                           kernel_size=(3, 3),
                           strides=(1, 1),
                           padding='same',
                           use_partial_gating=True)(conv)

        conv = tf.keras.layers.Conv2D(filters=128,
                                      kernel_size=(3, 3),
                                      strides=(1, 1),
                                      padding='same',
                                      activation='tanh')(conv)

        conv = tf.keras.layers.MaxPooling2D(pool_size=(1, 4), strides=(1, 4), padding='valid')(conv)

        conv = AdaptiveDenseReshape(target_shape=self.lexical_shape)(conv)
        blstm = tf.keras.layers.Reshape(target_shape=(-1, conv.get_shape()[-1]))(conv)

        blstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True))(blstm)
        blstm = tf.keras.layers.Dense(units=128, activation='tanh')(blstm)

        blstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True))(blstm)
        dense = tf.keras.layers.Dense(units=self.lexical_shape[-1], activation='softmax')(blstm)

        outputs = tf.keras.layers.Reshape(target_shape=self.lexical_shape)(dense)

        self.model = tf.keras.Model(inputs=inputs, outputs=outputs, name=self.name)
        self.summary = self.model.summary
        self.call = self.model.call
