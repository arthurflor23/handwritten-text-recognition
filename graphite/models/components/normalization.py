import tensorflow as tf


class ConditionalBatchNormalization(tf.keras.layers.Layer):
    """
    Conditional Batch Normalization for TensorFlow models.
    Enhances conditional GANs by using unique parameters for each condition.

    References
    ----------
    - [Modulating early visual processing by language](https://arxiv.org/abs/1707.00683v3).
    """

    def __init__(self, momentum=0.9, epsilon=1e-5, **kwargs):
        """
        Initializes the conditional batch normalization layer.

        Parameters
        ----------
        momentum : float, optional
            Momentum for the moving average of mean and variance.
        epsilon : float, optional
            Small constant to avoid division by zero.
        **kwargs : dict
            Additional keyword arguments for the layer.
        """

        super().__init__(**kwargs)

        self.momentum = momentum
        self.epsilon = epsilon

    def build(self, input_shape):
        """
        Create the layer's weights.

        Parameters
        ----------
        input_shape : list of TensorShape
            Shape of the input tensor.
        """

        self.num_channels = input_shape[0][-1]

        self.beta_mapping = SpectralNormalization(tf.keras.layers.Dense(self.num_channels))
        self.gamma_mapping = SpectralNormalization(tf.keras.layers.Dense(self.num_channels))

        self.mean = self.add_weight(shape=(self.num_channels,),
                                    initializer='zeros',
                                    trainable=False,
                                    name=f"{self.name}_mean")

        self.variance = self.add_weight(shape=(self.num_channels,),
                                        initializer='ones',
                                        trainable=False,
                                        name=f"{self.name}_variance")

    def call(self, inputs, training=None):
        """
        Call the layer with the specified inputs.

        Parameters
        ----------
        inputs : list or tuple
            The inputs tensor and the conditional data tensor.
        training : bool, optional
            Whether the layer should behave in training mode or in inference mode.

        Returns
        -------
        tf.Tensor
            The normalized output tensor.
        """

        inputs, conditions = inputs

        beta = self.beta_mapping(conditions)
        gamma = self.gamma_mapping(conditions)

        beta = tf.reshape(beta, shape=[-1, 1, 1, self.num_channels])
        gamma = tf.reshape(gamma, shape=[-1, 1, 1, self.num_channels])

        if training:
            mean, variance = tf.nn.moments(x=inputs, axes=[0, 1, 2])

            self.mean.assign(self.mean * self.momentum + mean * (1 - self.momentum))
            self.variance.assign(self.variance * self.momentum + variance * (1 - self.momentum))

        else:
            mean = self.mean
            variance = self.variance

        normalized = tf.nn.batch_normalization(inputs, mean, variance, beta, gamma, self.epsilon)

        return normalized

    def get_config(self):
        """
        Return the config of the layer.

        Returns
        -------
        dict
            A dictionary containing the configuration of the layer.
        """

        config = {
            'momentum': self.momentum,
            'epsilon': self.epsilon,
            'mean': self.mean,
            'variance': self.variance,
        }
        base_config = super().get_config()
        return {**base_config, **config}


class SpectralNormalization(tf.keras.layers.Wrapper):
    """
    Spectral Normalization for TensorFlow models.
    Optimizes GAN training stability by normalizing layer weights.

    References
    ----------
    - [Spectral Normalization for GANs](https://arxiv.org/abs/1802.05957).
    """

    def __init__(self, layer, power_iterations=1, **kwargs):
        """
        Initializes the spectral normalization wrapper.

        Parameters
        ----------
        layer : tf.keras.layers.Layer
            The layer to which spectral normalization will be applied.
        power_iterations : int, optional
            Number of power iterations to perform for normalization.
        **kwargs : dict
            Additional keyword arguments for the wrapper.
        """

        super().__init__(layer, **kwargs)

        self.power_iterations = power_iterations

    def build(self, input_shape):
        """
        Build the wrapper for the specified input shape.

        Parameters
        ----------
        input_shape : tuple or list
            The shape of the input to the layer.
        """

        super().build(input_shape)

        input_shape = tf.TensorShape(input_shape)
        self.input_spec = tf.keras.layers.InputSpec(shape=[None] + input_shape[1:])

        if hasattr(self.layer, 'kernel'):
            self.kernel = self.layer.kernel
        elif hasattr(self.layer, 'embeddings'):
            self.kernel = self.layer.embeddings
        else:
            raise ValueError(
                f"{type(self.layer).__name__} object has no attribute 'kernel' nor 'embeddings'"
            )

        self.kernel_shape = self.kernel.shape.as_list()

        self.vector_u = self.add_weight(
            shape=(1, self.kernel_shape[-1]),
            initializer=tf.initializers.TruncatedNormal(stddev=0.02),
            trainable=False,
            dtype=self.kernel.dtype,
            name=f"{self.name}_vector_u",
        )

    def call(self, inputs, training=None):
        """
        Call the wrapped layer with spectral normalization.

        Parameters
        ----------
        inputs : tf.Tensor or array-like
            The inputs to the layer.
        training : bool, optional
            If True, apply spectral normalization during training.

        Returns
        -------
        tf.Tensor
            The output tensor from the wrapped layer.
        """

        if training:
            self.normalize_weights()

        output = self.layer(inputs)
        return output

    def normalize_weights(self):
        """
        Normalizes the layer's weights using the power iteration method.
        """

        weights = tf.reshape(self.kernel, [-1, self.kernel_shape[-1]])
        vector_u = self.vector_u

        if not tf.reduce_all(tf.equal(weights, 0.0)):
            for _ in range(self.power_iterations):
                vector_v = tf.math.l2_normalize(tf.matmul(vector_u, weights, transpose_b=True))
                vector_u = tf.math.l2_normalize(tf.matmul(vector_v, weights))

            vector_u = tf.stop_gradient(vector_u)
            vector_v = tf.stop_gradient(vector_v)

            sigma = tf.matmul(tf.matmul(vector_v, weights), vector_u, transpose_b=True)

            self.vector_u.assign(tf.cast(vector_u, self.vector_u.dtype))
            self.kernel.assign(tf.cast(tf.reshape(self.kernel / sigma, self.kernel_shape), self.kernel.dtype))

    def compute_output_shape(self, input_shape):
        """
        Compute the output shape of the wrapped layer.

        Parameters
        ----------
        input_shape : tuple or list
            The shape of the input to the layer.

        Returns
        -------
        tf.TensorShape
            The computed shape of the output from the wrapped layer.
        """

        return tf.TensorShape(self.layer.compute_output_shape(input_shape).as_list())

    def get_config(self):
        """
        Return the config of the wrapper.

        Returns
        -------
        dict
            A dictionary containing the configuration of the wrapper.
        """

        config = {
            'power_iterations': self.power_iterations,
        }
        base_config = super().get_config()
        return {**base_config, **config}
