import tensorflow as tf


class ConditionalBatchNormalization(tf.keras.layers.Layer):

    def __init__(self, units, momentum=0.1, epsilon=1e-5, **kwargs):
        super().__init__(**kwargs)

        self.units = units

        self.gain = tf.keras.layers.SpectralNormalization(tf.keras.layers.Dense(units=units, use_bias=False))
        self.bias = tf.keras.layers.SpectralNormalization(tf.keras.layers.Dense(units=units, use_bias=False))
        self.bn = tf.keras.layers.BatchNormalization(momentum=momentum, epsilon=epsilon)

    def call(self, inputs, conditional_inputs, training=None):
        if training is None:
            training = tf.keras.backend.learning_phase()

        gain = 1 + self.gain(conditional_inputs)
        bias = self.bias(conditional_inputs)

        gain = tf.reshape(gain, (-1, 1, 1, self.units))
        bias = tf.reshape(bias, (-1, 1, 1, self.units))

        out = self.bn(inputs, training=training)

        return gain * out + bias

    def compute_output_shape(self, input_shape):
        return input_shape


class SpectralNormalization(tf.keras.layers.Wrapper):
    """
    Applies spectral normalization to a layer to stabilize GAN training.

    Args:
        layer: tf.keras.layers.Layer
            Layer with `kernel` or `embeddings` to normalize.
        power_iterations: int, optional
            Iterations for normalization (default: 1).
        **kwargs
            Additional keyword arguments for `tf.keras.layers.Wrapper`.

    Raises:
        ValueError: If `power_iterations` is non-positive or layer is incompatible.
        AttributeError: If `layer` lacks `kernel` or `embeddings`.

    Reference:
        [Spectral Normalization for GANs](https://arxiv.org/abs/1802.05957).
    """

    def __init__(self, layer, power_iterations=1, **kwargs):
        """
        Initializes the spectral normalization wrapper.

        Args:
            layer: tf.keras.layers.Layer
                The layer to apply spectral normalization to.
            power_iterations: int, optional
                Number of power iterations for normalization.
            **kwargs
                Additional keyword arguments for the wrapper.
        """

        super().__init__(layer, **kwargs)

        if power_iterations <= 0:
            raise ValueError(
                "`power_iterations` should be greater than zero"
            )

        self.power_iterations = power_iterations

    def build(self, input_shape):
        """
        Builds the layer by setting up the weights.

        Args:
            input_shape: TensorShape or tuple/list
                Shape of the input to the layer.
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
            name='vector_u',
            dtype=self.kernel.dtype,
        )

    def call(self, inputs, training=None):
        """
        Calls the wrapped layer with spectral normalization.

        Args:
            inputs: tensor or array-like
                Inputs to the layer.
            training: bool, optional
                If True, applies spectral normalization during training.

        Returns:
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
        Computes the output shape of the wrapped layer.

        Args:
            input_shape: TensorShape or tuple/list
                Shape of the input to the layer.

        Returns:
            TensorShape
                Computed shape of the output from the wrapped layer.
        """

        return tf.TensorShape(self.layer.compute_output_shape(input_shape).as_list())

    def get_config(self):
        """
        Returns the config of the wrapper.

        Returns:
            A dictionary containing the configuration of the wrapper.
        """

        config = {"power_iterations": self.power_iterations}
        base_config = super().get_config()
        return {**base_config, **config}
