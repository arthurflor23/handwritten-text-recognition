import tensorflow as tf


class GatedConv2D(tf.keras.layers.Conv2D):
    """
    Implements gated convolutional layer for TensorFlow.
    Combines linear and sigmoid activations for convolutional gating.

    References
    ----------
    Gated convolutional recurrent neural networks for multilingual handwriting recognition
        https://ieeexplore.ieee.org/document/8270042

    Language modeling with gated convolutional networks
        https://arxiv.org/abs/1612.08083
    """

    def __init__(self, filters, kernel_size, use_partial_gating=False, **kwargs):
        """
        Initializes the gated convolutional layer.

        Parameters
        ----------
        filters : int
            Number of filters for the convolution.
        kernel_size : int or tuple/list
            The size of the convolution window.
        use_partial_gating : bool, optional
            Whether to use partial gating.
        **kwargs : dict
            Conv2D keyword arguments.
        """

        super().__init__(filters, kernel_size, **kwargs)

        self.filters = filters
        self.kernel_size = kernel_size
        self.use_partial_gating = use_partial_gating

        if self.use_partial_gating:
            self.conv = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, **kwargs)
        else:
            self.conv = tf.keras.layers.Conv2D(filters=filters * 2, kernel_size=kernel_size, **kwargs)

    def get_config(self):
        """
        Returns the config of the optimizer.

        Returns
        -------
        dict
            A dictionary containing the configuration of the optimizer.
        """

        config = super().get_config()

        config.update({
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'use_partial_gating': self.use_partial_gating,
        })

        return config

    def call(self, inputs):
        """
        Apply gated convolution to the input.

        Parameters
        ----------
        inputs : tensor
            The inputs to the layer.

        Returns
        -------
        tf.Tensor
            Tensor resulting from the gated convolution.
        """

        conv = self.conv(inputs)

        if self.use_partial_gating:
            conv = tf.keras.layers.Activation('sigmoid')(conv)
            outputs = inputs * conv
        else:
            linear, sigmoid = tf.split(conv, 2, axis=-1)
            linear = tf.keras.layers.Activation('linear')(linear)
            sigmoid = tf.keras.layers.Activation('sigmoid')(sigmoid)
            outputs = linear * sigmoid

        return outputs
