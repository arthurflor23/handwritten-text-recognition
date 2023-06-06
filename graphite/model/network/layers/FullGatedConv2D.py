import tensorflow as tf


class FullGatedConv2D(tf.keras.layers.Conv2D):
    """
    Gated Convolutional Layer.

    Reference:
        Y. N. Dauphin, A. Fan, M. Auli, and D. Grangier,
        Language modeling with gated convolutional networks, in
        34th International Conference on Machine Learning (ICML), vol. 70, p. 933-941, 2017.
    """

    def __init__(self, filters, **kwargs):
        """
        Initialize the FullGatedConv2D layer.

        Parameters
        ----------
        filters : int
            Number of filters for the convolutional layer.
        **kwargs : dict
            Additional Conv2D keyword arguments.
        """

        super().__init__(filters=filters * 2, **kwargs)
        self.nb_filters = filters

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
        linear = tf.keras.layers.Activation('linear')(output[:, :, :, :self.nb_filters])
        sigmoid = tf.keras.layers.Activation('sigmoid')(output[:, :, :, self.nb_filters:])
        multiply = tf.keras.layers.Multiply()([linear, sigmoid])

        return multiply

    def compute_output_shape(self, input_shape):
        """
        Compute shape of the layer output.

        Parameters
        ----------
        input_shape : tuple
            Shape of the input tensor.

        Returns
        -------
        tuple
            Shape of the layer output tensor.
        """

        output_shape = super().compute_output_shape(input_shape)
        output_shape = tuple(output_shape[:3]) + (self.nb_filters * 2,)

        return output_shape

    def get_config(self):
        """
        Return the configuration of the layer.

        Returns
        -------
        dict
            Configuration dictionary for the layer.
        """

        config = super().get_config()
        config['nb_filters'] = self.nb_filters
        del config['filters']

        return config
