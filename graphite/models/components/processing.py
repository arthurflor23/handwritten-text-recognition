import tensorflow as tf


class AdaptiveDenseReshape(tf.keras.layers.Layer):
    """
    Layer that applies a dense layer followed by a reshape operation.
    """

    def __init__(self, target_shape, **kwargs):
        """
        Process that applies a dense layer followed by a reshape operation.

        Parameters
        ----------
        target_shape : list or tuple
            Target shape to reshape the output after dense layer.
        """

        super().__init__(**kwargs)

        self.target_shape = target_shape

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
            'target_shape': self.target_shape,
        })

        return config

    def build(self, input_shape):
        """
        Builds the dense layer based on the input shape and target reshape shape.

        Parameters
        ----------
        input_shape : tuple
            Shape of the input to the layer.
        """

        target_prod = tf.math.reduce_prod(self.target_shape[:-1]).numpy()
        units = tf.math.ceil((input_shape[-1] * target_prod) / (input_shape[1] * input_shape[2]))

        self.dense = tf.keras.layers.Dense(units=int(units))
        self.target_reshape = self.target_shape[:-1] + [-1]

    def call(self, inputs):
        """
        Calls the dense and reshape layers on the input tensors.

        Parameters
        ----------
        inputs : tensor
            Input tensor to the layer.

        Returns
        -------
        tensor
            Output tensor reshaped.
        """

        x = self.dense(inputs)
        return tf.keras.layers.Reshape(target_shape=self.target_reshape)(x)


class ExtractPatches(tf.keras.layers.Layer):
    """
    A Tensorflow Keras layer to extract patches from input images.
    """

    def __init__(self, patch_shape, **kwargs):
        """
        Initializes Patches layer.

        Parameters
        ----------
        patch_shape : list or tuple
            The target patch size to create.
        **kwargs
            Additional keyword arguments for the Layer.
        """

        super().__init__(**kwargs)

        self.patch_shape = list(patch_shape)

    def get_config(self):
        """
        Returns the config of the layer.

        Returns
        -------
        dict
            A dictionary containing the configuration of the layer.
        """

        config = super().get_config()

        config.update({
            'patch_shape': self.patch_shape,
        })

        return config

    def build(self, input_shape):
        """
        Builds the layer with patches ratio values.

        Parameters
        ----------
        input_shape : tuple
            Shape of the input tensor.
        """

        self.patch_height_ratio = input_shape[1] // self.patch_shape[0]
        self.patch_width_ratio = input_shape[2] // self.patch_shape[1]

    def call(self, inputs):
        """
        Splits the input image into patches.

        Parameters
        ----------
        inputs : tensor
            The input tensor representing images.

        Returns
        -------
        tf.Tensor
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
