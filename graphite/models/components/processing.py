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

        self.dense = []
        self.batch_norm = []

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

        super().build(input_shape)

        target_shape = [-1, input_shape[-1] * input_shape[-2]]
        self.merge_last_dims = tf.keras.layers.Reshape(target_shape=target_shape)

        for units in self.target_shape[:-1]:
            self.dense.append(tf.keras.layers.Dense(units, activation='tanh'))
            self.batch_norm.append(tf.keras.layers.BatchNormalization())

    def call(self, inputs, training=None):
        """
        Calls the dense and reshape layers on the input tensors.

        Parameters
        ----------
        inputs : tensor
            Input tensor to the layer.
        training : bool, optional
            If True, apply spectral normalization during training.

        Returns
        -------
        tensor
            Output tensor reshaped.
        """

        inputs = self.merge_last_dims(inputs)

        input_shape = tf.shape(inputs)
        input_dims = len(input_shape) - 1
        target_dims = len(self.target_shape)

        if target_dims > input_dims:
            dim_size = tf.ones(target_dims - input_dims, dtype=tf.int32)
            new_shape = tf.concat([input_shape[:1], dim_size, input_shape[1:]], axis=0)
            inputs = tf.reshape(inputs, new_shape)

        elif target_dims < input_dims:
            dim_size = tf.reduce_prod(input_shape[-(input_dims - target_dims + 1):])
            new_shape = tf.concat([input_shape[:-(input_dims - target_dims)], [dim_size]], axis=0)
            inputs = tf.reshape(inputs, new_shape)

        for i in range(len(self.dense)):
            perm = [0] + [j+1 for j in range(target_dims) if i != j] + [i+1]
            inputs = tf.transpose(inputs, perm=perm)

            inputs = self.dense[i](inputs)
            inputs = self.batch_norm[i](inputs, training=training)

            reset_perm = [j for j in range(target_dims)]
            reset_perm = reset_perm[:i+1] + [target_dims] + reset_perm[i+1:]
            inputs = tf.transpose(inputs, perm=reset_perm)

        return inputs


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
