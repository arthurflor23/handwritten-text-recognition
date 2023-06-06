import tensorflow as tf


class GatedConv2D(tf.keras.layers.Conv2D):
    """
    Tensorflow Keras layer implementation of gated convolution.

    References
    ----------
    T. Bluche, R. Messina,
    Gated convolutional recurrent neural networks for multilingual handwriting recognition.
    14th IAPR International Conference on Document Analysis and Recognition (ICDAR), p. 646-651, 11 2017.
    """

    def __init__(self, **kwargs):
        """
         Initialize the GatedConv2D layer.

         Parameters
         ----------
         **kwargs : dict
             Conv2D keyword arguments.

         """

        super().__init__(**kwargs)

    def call(self, inputs):
        """
        Apply gated convolution to the input.

        Parameters
        ----------
        inputs : tensor
            Input tensor.

        Returns
        -------
        tensor
            Tensor resulting from the gated convolution.
        """

        output = super().call(inputs)
        linear = tf.keras.layers.Activation('linear')(inputs)
        sigmoid = tf.keras.layers.Activation('sigmoid')(output)
        multiply = tf.keras.layers.Multiply()([linear, sigmoid])

        return multiply

    def get_config(self):
        """
        Return the configuration of the layer.

        Returns
        -------
        dict
            Configuration dictionary for the layer.
        """

        config = super().get_config()

        return config
