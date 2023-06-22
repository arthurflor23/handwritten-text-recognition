import tensorflow as tf


class InputProcessing(tf.keras.layers.Layer):
    """
    This class defines a custom layer for processing.
    """

    def __init__(self, height_shape=None, width_shape=None, pad_value=255, **kwargs):
        """
        Initialize the InputProcess class.

        Parameters
        ----------
        height_shape : int
            Target height shape for resizing operation. Default is None.
        width_shape : int
            Target width shape for resizing operation. Default is None.
        pad_value : int, optional
            Padding value. Default is 255.
        **kwargs :
            Variable length argument list for parent class.
        """

        super().__init__(**kwargs)
        self.height_shape = height_shape
        self.width_shape = width_shape
        self.pad_value = pad_value

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

        if self.height_shape is not None and self.width_shape is not None:
            _, height, width, _ = tf.unstack(tf.shape(inputs))

            if height > self.height_shape or width > self.width_shape:
                height = tf.cast(height, tf.float32)
                width = tf.cast(width, tf.float32)

                height_scale = self.height_shape / height
                width_scale = self.width_shape / width

                scale = tf.minimum(height_scale, width_scale)

                height = tf.cast(tf.round(height * scale), tf.int32)
                width = tf.cast(tf.round(width * scale), tf.int32)

                inputs = tf.image.resize(inputs, [height, width], method=tf.image.ResizeMethod.AREA)

            padding_height = self.height_shape - height
            padding_width = self.width_shape - width

            paddings = [[0, 0], [0, padding_height], [0, padding_width], [0, 0]]
            inputs = tf.pad(inputs, paddings, constant_values=self.pad_value)

        inputs.set_shape([None, self.height_shape, self.width_shape, 1])

        inputs = tf.image.transpose(inputs)
        inputs = tf.image.per_image_standardization(inputs)
        # inputs = tf.math.divide(inputs, 255.)

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
            'height_shape': self.height_shape,
            'width_shape': self.width_shape,
            'pad_value': self.pad_value,
        })

        return config
