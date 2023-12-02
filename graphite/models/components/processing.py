import tensorflow as tf


class ExtractPatches(tf.keras.layers.Layer):
    """
    A Tensorflow Keras layer to extract patches from input images.
    """

    def __init__(self, patch_shape):
        """
        Initializes Patches layer.

        Args:
            patch_shape (list or tuple):
                The target patch size to create.
            **kwargs
                Additional keyword arguments for the Layer.
        """

        super().__init__()

        self.patch_shape = list(patch_shape)

    def build(self, input_shape):
        """
        Builds the layer with patches ratio values.

        Args:
            input_shape (tuple):
                Shape of the input tensor.
        """

        self.patch_height_ratio = input_shape[1] // self.patch_shape[0]
        self.patch_width_ratio = input_shape[2] // self.patch_shape[1]

    def call(self, inputs):
        """
        Splits the input image into patches.

        Args:
            inputs (tensor):
                The input tensor representing images.

        Returns:
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
            "patch_shape": self.patch_shape,
        }
        base_config = super().get_config()
        return {**base_config, **config}
