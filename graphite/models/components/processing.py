import tensorflow as tf


class ExtractPatches(tf.keras.layers.Layer):
    """
    A Tensorflow Keras layer to extract patches from input images.
    """

    def __init__(self, shape, patch_shape):
        """
        Initializes Patches layer.

        Args:
            shape: list or tuple
                The source shape.
            patch_shape: list or tuple
                The target patch size to create.
            **kwargs
                Additional keyword arguments for the Layer.
        """

        super().__init__()

        self.shape = list(shape)
        self.patch_shape = list(patch_shape)

        self.patch_height_ratio = self.shape[0] // self.patch_shape[0]
        self.patch_width_ratio = self.shape[1] // self.patch_shape[1]

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

        patches = tf.image.extract_patches(
            images=inputs,
            sizes=[1, self.patch_height_ratio, self.patch_width_ratio, 1],
            strides=[1, self.patch_height_ratio, self.patch_width_ratio, 1],
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
            "shape": self.shape,
            "patch_shape": self.patch_shape,
            "patch_height_ratio": self.patch_height_ratio,
            "patch_width_ratio": self.patch_width_ratio,
        }
        base_config = super().get_config()
        return {**base_config, **config}
