import tensorflow as tf


class DynamicReshape(tf.keras.layers.Layer):
    """
    A TensorFlow Keras layer for dynamic input tensor reshaping.
    Reshapes an input tensor to a specified target shape.

    Args:
        target_shape: list of tuple
            Target shape for reshaping.
        **kwargs
            Additional Keras Layer keyword arguments.
    """

    def __init__(self, target_shape, **kwargs):
        """
        Initializes the DynamicReshape layer.

        Args:
            target_shape: list of int
                The target shape to reshape the input tensor to.
            **kwargs
                Additional keyword arguments for the Layer.
        """

        super().__init__(**kwargs)

        self.target_shape = list(target_shape[:-1])
        self.target_shape_prod = 1

        for x in self.target_shape:
            self.target_shape_prod *= x

    def call(self, inputs):
        """
        Reshapes the input tensor to the target shape.

        Args:
            inputs: tensor
                The input tensor to be reshaped.

        Returns:
            tensor
                The reshaped tensor.
        """

        input_shape_prod = 1
        for x in inputs.get_shape()[1:]:
            input_shape_prod *= x

        new_shape = input_shape_prod // self.target_shape_prod
        new_shape = [-1] + self.target_shape + [new_shape]

        return tf.reshape(inputs, new_shape)

    def get_config(self):
        """
        Returns the config of the wrapper.

        Returns:
            A dictionary containing the configuration of the layer.
        """

        config = {
            "target_shape": self.target_shape,
            "target_shape_prod": self.target_shape_prod,
        }
        base_config = super().get_config()
        return {**base_config, **config}
