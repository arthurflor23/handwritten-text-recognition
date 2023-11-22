import tensorflow as tf


class DynamicReshape(tf.keras.layers.Layer):
    """
    A TensorFlow Keras layer for dynamic input tensor reshaping.
    Reshapes an input tensor to a specified target shape.
    """

    def __init__(self, target_shape, **kwargs):
        """
        Initializes DynamicReshape layer.

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

        shape_prod = 1
        for x in inputs.get_shape()[1:]:
            shape_prod *= x

        new_shape = shape_prod // self.target_shape_prod
        new_shape = [-1] + self.target_shape + [new_shape]

        return tf.reshape(inputs, new_shape)

    def get_config(self):
        """
        Returns the config of the layer.

        Returns:
            A dictionary containing the configuration of the layer.
        """

        config = {
            "target_shape": self.target_shape,
            "target_shape_prod": self.target_shape_prod,
        }
        base_config = super().get_config()
        return {**base_config, **config}


class ExtractPatches(tf.keras.layers.Layer):
    """
    A Tensorflow Keras layer to extract patches from input images.
    """

    def __init__(self, patch_shape):
        """
        Initializes Patches layer.

        Args:
            patch_shape: list or tuple
                The target patch size to create.
            **kwargs
                Additional keyword arguments for the Layer.
        """

        super().__init__()

        self.patch_shape = list(patch_shape)

    def call(self, inputs):
        """
        Splits the input image into patches.

        Args:
            inputs: tensor
                The input tensor representing images.

        Returns:
            tensor
                A tensor containing the extracted patches.
        """

        shape = inputs.get_shape()

        patches = tf.image.extract_patches(
            images=inputs,
            sizes=[1, shape[1]//self.patch_shape[0], shape[2]//self.patch_shape[1], 1],
            strides=[1, shape[1]//self.patch_shape[0], shape[2]//self.patch_shape[1], 1],
            rates=[1, 1, 1, 1],
            padding='VALID',
        )

        return patches

    def get_config(self):
        """
        Returns the config of the layer.

        Returns:
            A dictionary containing the configuration of the layer.
        """

        config = {
            "patch_shape": self.patch_shape,
        }
        base_config = super().get_config()
        return {**base_config, **config}
