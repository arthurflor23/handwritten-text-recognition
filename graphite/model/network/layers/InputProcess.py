import tensorflow as tf


class InputProcess(tf.keras.layers.Layer):
    """
    This class defines a custom layer for processing.
    """

    def __init__(self, target_shape, **kwargs):
        """
        Initialize the InputProcess class.

        Parameters
        ----------
        target_shape : int
            Target shape for resizing operation.
        **kwargs :
            Variable length argument list for parent class.
        """

        super().__init__(**kwargs)

        self.target_shape = target_shape

    def call(self, inputs):
        """
        Method for forward pass of inputs in the layer.

        Parameters
        ----------
        inputs : tf.Tensor
            Input tensor for processing.

        Returns
        -------
        inputs : tf.Tensor
            Processed tensor after applying operations.
        """

        inputs = tf.keras.layers.Lambda(lambda x: tf.image.rot90(image=x, k=-1))(inputs)
        inputs = tf.keras.layers.Lambda(lambda x: tf.image.flip_left_right(image=x))(inputs)
        inputs = tf.keras.layers.Lambda(lambda x: tf.image.resize_with_pad(image=x,
                                                                           target_height=self.target_shape[0],
                                                                           target_width=self.target_shape[1],
                                                                           method=tf.image.ResizeMethod.BILINEAR,
                                                                           antialias=True))(inputs)

        inputs = tf.keras.layers.Rescaling(scale=1./255, offset=0.0)(inputs)

        return inputs

    def get_config(self):
        """
        Return the configuration of the layer.

        Returns
        -------
        dict
            Configuration dictionary for the layer.
        """

        config = super().get_config()

        config.update({
            'target_shape': self.target_shape,
        })

        return config
