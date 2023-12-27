import tensorflow as tf


class ConditionalBatchNormalization(tf.keras.layers.Layer):
    """
    Conditional Batch Normalization for TensorFlow models.
    Enhances conditional GANs by using unique parameters for each condition.

    References
    ----------
    Modulating early visual processing by language
        https://arxiv.org/abs/1707.00683v3
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

    def get_config(self):
        """
        Return the config of the layer.

        Returns
        -------
        dict
            A dictionary containing the configuration of the layer.
        """

        config = super().get_config()

        config.update({
            'momentum': self.momentum,
            'epsilon': self.epsilon,
            'mean': self.mean,
            'variance': self.variance,
        })

        return config

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
            The inputs tensors (data and conditional data).
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


class GatedConv2D(tf.keras.layers.Layer):
    """
    Implements gated convolutional layer for TensorFlow.
    Combines linear and sigmoid activations for convolutional gating.

    References
    ----------
    Gated convolutional recurrent neural networks for multilingual handwriting recognition
        https://ieeexplore.ieee.org/document/8270042

    Language modeling with gated convolutional networks
        https://arxiv.org/abs/1612.08083
    """

    def __init__(self,
                 filters,
                 kernel_size=(3, 3),
                 strides=(1, 1),
                 padding='same',
                 kernel_initializer='glorot_uniform',
                 kernel_regularizer=None,
                 kernel_constraint=None,
                 fullgate=True,
                 **kwargs):
        """
        Initializes the gated convolutional layer.

        Parameters
        ----------
        filters : int
            Number of output filters.
        kernel_size : tuple of 2 ints, optional
            Convolution window size.
        strides : tuple of 2 ints, optional
            Convolution strides.
        padding : str, optional
            Padding type.
        kernel_initializer : initializer, optional
            Kernel weights initializer.
        kernel_regularizer : regularizer, optional
            Kernel weights regularizer.
        kernel_constraint : constraint, optional
            Kernel weights constraint.
        fullgate : bool, optional
            Whether to use full gating.
        **kwargs : dict
            Conv2D keyword arguments.
        """

        super().__init__(**kwargs)

        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.kernel_initializer = kernel_initializer
        self.kernel_regularizer = kernel_regularizer
        self.kernel_constraint = kernel_constraint
        self.fullgate = fullgate

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
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'strides': self.strides,
            'padding': self.padding,
            'kernel_initializer': self.kernel_initializer,
            'kernel_regularizer': self.kernel_regularizer,
            'kernel_constraint': self.kernel_constraint,
            'fullgate': self.fullgate,
        })

        return config

    def build(self, input_shape):
        """
        Initializes layer weights.

        Parameters
        ----------
        input_shape : tuple
            Shape of the input to the layer.
        """

        super().build(input_shape)

        self.conv = tf.keras.layers.Conv2D(filters=self.filters * (2 if self.fullgate else 1),
                                           kernel_size=self.kernel_size,
                                           strides=self.strides,
                                           padding=self.padding,
                                           kernel_initializer=self.kernel_initializer,
                                           kernel_regularizer=self.kernel_regularizer,
                                           kernel_constraint=self.kernel_constraint)

    def call(self, inputs):
        """
        Apply gated convolution to the input.

        Parameters
        ----------
        inputs : tensor
            The inputs to the layer.

        Returns
        -------
        tf.Tensor
            Tensor resulting from the gated convolution.
        """

        conv = self.conv(inputs)

        if self.fullgate:
            linear, sigmoid = tf.split(conv, 2, axis=-1)
            linear = tf.keras.layers.Activation('linear')(linear)
            sigmoid = tf.keras.layers.Activation('sigmoid')(sigmoid)
            outputs = linear * sigmoid
        else:
            conv = tf.keras.layers.Activation('sigmoid')(conv)
            outputs = inputs * conv

        return outputs


class OctConv2D(tf.keras.layers.Layer):
    """
    Implements octave convolutional layer for TensorFlow.
    This layer processes input feature maps by splitting them into high and low frequency components

    References
    ----------
    Drop an Octave: Reducing Spatial Redundancy in Convolutional Neural Networks with Octave Convolution
        https://arxiv.org/abs/1904.05049
    """

    def __init__(self,
                 alpha,
                 filters,
                 kernel_size=(3, 3),
                 strides=(1, 1),
                 padding='same',
                 kernel_initializer='glorot_uniform',
                 kernel_regularizer=None,
                 kernel_constraint=None,
                 **kwargs):
        """
        Initialize the octave convolutional layer.

        Parameters
        ----------
        alpha : float
            Fraction of filters for low frequency.
        filters : int
            Number of output filters.
        kernel_size : tuple of 2 ints, optional
            Convolution window size.
        strides : tuple of 2 ints, optional
            Convolution strides.
        padding : str, optional
            Padding type.
        kernel_initializer : initializer, optional
            Kernel weights initializer.
        kernel_regularizer : regularizer, optional
            Kernel weights regularizer.
        kernel_constraint : constraint, optional
            Kernel weights constraint.
        """

        super().__init__(**kwargs)

        self.alpha = alpha
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.kernel_initializer = kernel_initializer
        self.kernel_regularizer = kernel_regularizer
        self.kernel_constraint = kernel_constraint

        self.low_channels = int(self.filters * self.alpha)
        self.high_channels = self.filters - self.low_channels

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
            'alpha': self.alpha,
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'strides': self.strides,
            'padding': self.padding,
            'kernel_initializer': self.kernel_initializer,
            'kernel_regularizer': self.kernel_regularizer,
            'kernel_constraint': self.kernel_constraint,
        })

        return config

    def build(self, input_shape):
        """
        Initializes layer weights.

        Parameters
        ----------
        input_shape : tuple
            Shape of the input to the layer.
        """

        super().build(input_shape)

        high_in = int(input_shape[0][3])
        low_in = int(input_shape[1][3])

        self.high_to_high_kernel = self.add_weight(name='high_to_high_kernel',
                                                   shape=(*self.kernel_size, high_in, self.high_channels),
                                                   initializer=self.kernel_initializer,
                                                   regularizer=self.kernel_regularizer,
                                                   constraint=self.kernel_constraint)

        self.high_to_low_kernel = self.add_weight(name='high_to_low_kernel',
                                                  shape=(*self.kernel_size, high_in, self.low_channels),
                                                  initializer=self.kernel_initializer,
                                                  regularizer=self.kernel_regularizer,
                                                  constraint=self.kernel_constraint)

        self.low_to_high_kernel = self.add_weight(name='low_to_high_kernel',
                                                  shape=(*self.kernel_size, low_in, self.high_channels),
                                                  initializer=self.kernel_initializer,
                                                  regularizer=self.kernel_regularizer,
                                                  constraint=self.kernel_constraint)

        self.low_to_low_kernel = self.add_weight(name='low_to_low_kernel',
                                                 shape=(*self.kernel_size, low_in, self.low_channels),
                                                 initializer=self.kernel_initializer,
                                                 regularizer=self.kernel_regularizer,
                                                 constraint=self.kernel_constraint)

    def call(self, inputs):
        """
        Processes the input tensors through the layer.

        Parameters
        ----------
        inputs : list of two tensors
            High and low frequency components of the input.

        Returns
        -------
        list of two tensors
            Processed high and low frequency outputs.
        """

        high_input, low_input = inputs

        high_to_high = tf.nn.conv2d(input=high_input,
                                    filters=self.high_to_high_kernel,
                                    strides=[1, *self.strides, 1],
                                    padding=self.padding.upper())

        high_to_low = tf.nn.avg_pool2d(input=high_input,
                                       ksize=[1, 2, 2, 1],
                                       strides=[1, 2, 2, 1],
                                       padding='SAME')

        high_to_low = tf.nn.conv2d(input=high_to_low,
                                   filters=self.high_to_low_kernel,
                                   strides=[1, *self.strides, 1],
                                   padding=self.padding.upper())

        low_to_high = tf.nn.conv2d(input=low_input,
                                   filters=self.low_to_high_kernel,
                                   strides=[1, *self.strides, 1],
                                   padding=self.padding.upper())

        low_to_high = tf.tile(low_to_high, [1, 2, 2, 1])

        low_to_low = tf.nn.conv2d(input=low_input,
                                  filters=self.low_to_low_kernel,
                                  strides=[1, *self.strides, 1],
                                  padding=self.padding.upper())

        high_add = high_to_high + low_to_high
        low_add = low_to_low + high_to_low

        return [high_add, low_add]

    def compute_output_shape(self, input_shapes):
        """
        Compute the output shape of the layer.

        Parameters
        ----------
        input_shape : tuple or list
            The shape of the input to the layer.

        Returns
        -------
        tf.TensorShape
            The computed shape of the output from the layer.
        """

        high_in_shape, low_in_shape = input_shapes
        high_out_shape = (*high_in_shape[:3], self.high_channels)
        low_out_shape = (*low_in_shape[:3], self.low_channels)

        return [high_out_shape, low_out_shape]


class SpectralNormalization(tf.keras.layers.Wrapper):
    """
    Spectral Normalization for TensorFlow models.
    Optimizes GAN training stability by normalizing layer weights.

    References
    ----------
    Spectral Normalization for GANs
        https://arxiv.org/abs/1802.05957
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

        super().__init__(layer, name=layer.name, **kwargs)

        self.power_iterations = power_iterations

    def get_config(self):
        """
        Return the config of the wrapper.

        Returns
        -------
        dict
            A dictionary containing the configuration of the wrapper.
        """

        config = super().get_config()

        config.update({
            'power_iterations': self.power_iterations,
        })

        return config

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
            raise ValueError('Object has no attribute "kernel" nor "embeddings"')

        self.kernel_shape = self.kernel.shape.as_list()

        self.vector_u = self.add_weight(
            shape=(1, self.kernel_shape[-1]),
            initializer=tf.initializers.TruncatedNormal(stddev=0.02),
            trainable=False,
            dtype=self.kernel.dtype,
            name=f"{self.name}_{self.layer.name}_vector_u",
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


class SpectralSelfAttention(tf.keras.layers.Layer):
    """
    Spectral Self-Attention layer for TensorFlow models.

    Implements a self-attention mechanism with spectral normalization, suitable for tasks like
        image generation. The layer applies self-attention to the input tensor, enhancing feature
        representations for better model performance.

    References
    ----------
    Attention Is All You Need
        https://arxiv.org/abs/1706.03762

    Spectral Normalization for GANs
        https://arxiv.org/abs/1802.05957
    """

    def __init__(self, **kwargs):
        """
        Initialize the SpectralSelfAttention layer.

        Parameters
        ----------
        **kwargs : dict
            Additional keyword arguments for the layer.
        """

        super().__init__(**kwargs)

    def build(self, input_shape):
        """
        Build the layer with spectral normalization on convolutional layers.

        Parameters
        ----------
        input_shape : tuple
            Shape of the input tensor.
        """

        self.shape = input_shape
        self.num_channels = input_shape[-1]
        self.hw = input_shape[1] * input_shape[2]

        self.conv_f = SpectralNormalization(tf.keras.layers.Conv2D(self.num_channels // 8, 1))
        self.conv_g = SpectralNormalization(tf.keras.layers.Conv2D(self.num_channels // 8, 1))
        self.conv_h = SpectralNormalization(tf.keras.layers.Conv2D(self.num_channels // 2, 1))
        self.conv_o = SpectralNormalization(tf.keras.layers.Conv2D(self.num_channels, 1))

        self.gamma = self.add_weight(shape=(1,),
                                     initializer='zeros',
                                     trainable=True,
                                     name=f"{self.name}_gamma")

    def call(self, x):
        """
        Apply self-attention to the input tensor.

        Parameters
        ----------
        x : tf.Tensor
            Input tensor to the layer.

        Returns
        -------
        tf.Tensor
            Processed tensor with self-attention applied.
        """

        f = self.conv_f(x)
        g = self.conv_g(x)
        h = self.conv_h(x)

        f = tf.reshape(tensor=f, shape=[-1, self.hw, f.shape[-1]])
        g = tf.reshape(tensor=g, shape=[-1, self.hw, g.shape[-1]])
        h = tf.reshape(tensor=h, shape=[-1, self.hw, h.shape[-1]])

        s = tf.matmul(g, f, transpose_b=True)
        beta = tf.nn.softmax(logits=s)

        o = tf.matmul(beta, h)
        o = tf.reshape(tensor=o, shape=[-1, self.shape[1], self.shape[2], self.num_channels // 2])
        o = self.conv_o(o)

        return self.gamma * o + x
