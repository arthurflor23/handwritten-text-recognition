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

    def compute_output_shape(self, input_shape):
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

        if not self.built:
            self.build(input_shape)

        high_in_shape, low_in_shape = input_shape
        high_out_shape = (*high_in_shape[:3], self.high_channels)
        low_out_shape = (*low_in_shape[:3], self.low_channels)

        return tf.TensorShape([high_out_shape, low_out_shape])


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

        if not self.built:
            self.build(input_shape)

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


class TemporalConvolutional(tf.keras.layers.Layer):
    """
    A Temporal Convolutional Layer for sequence modeling.

    References
    ----------
    Temporal Convolutional Networks for Action Segmentation and Detection
        https://arxiv.org/abs/1611.05267

    An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling
        https://arxiv.org/abs/1803.01271
    """

    def __init__(self,
                 filters,
                 kernel_size,
                 nb_stacks=1,
                 dilations=(1, 2, 4, 8),
                 padding='causal',
                 dropout=0.0,
                 use_skip_connections=True,
                 return_sequences=False,
                 activation='tanh',
                 kernel_initializer='glorot_normal',
                 use_batch_norm=False,
                 use_layer_norm=False,
                 go_backwards=False,
                 **kwargs):
        """
        Initializes the temporal convolutional layer.

        Parameters
        ----------
        filters : int or list
            The dimensionality of the output space.
        kernel_size : int
            Length of the convolution window.
        nb_stacks : int
            Number of stacks of residual blocks.
        dilations : tuple
            The dilation rate to use for dilated convolution.
        padding : str
            One of 'causal' or 'same'.
        dropout : float
            Dropout rate.
        use_skip_connections : bool
            Whether to use skip connections.
        return_sequences : bool
            Whether to return the last output, or the full sequence.
        activation : str
            Activation function to use.
        kernel_initializer : str
            Initializer for the kernel weights matrix.
        use_batch_norm : bool
            Whether to use batch normalization.
        use_layer_norm : bool
            Whether to use layer normalization.
        go_backwards : bool
            Process the input sequence backwards and return the reversed sequence.
        """

        super().__init__(**kwargs)

        if use_batch_norm and use_layer_norm:
            raise ValueError('Only one of batch normalization or layer normalization can be used.')

        if isinstance(filters, list) and len(filters) != len(dilations):
            raise ValueError('Length of filters must match length of dilations.')

        if use_skip_connections and isinstance(filters, list) and len(set(filters)) > 1:
            raise ValueError('Skip connections require identical filter sizes for all layers.')

        if padding not in ['causal', 'same']:
            raise ValueError("Padding must be either 'causal' or 'same'.")

        self.filters = filters
        self.kernel_size = kernel_size
        self.nb_stacks = nb_stacks
        self.dilations = dilations
        self.padding = padding
        self.dropout = dropout
        self.use_skip_connections = use_skip_connections
        self.return_sequences = return_sequences
        self.activation_name = activation
        self.kernel_initializer = kernel_initializer
        self.use_batch_norm = use_batch_norm
        self.use_layer_norm = use_layer_norm
        self.go_backwards = go_backwards

        self.skip_connections = []
        self.residual_blocks = []
        self.layers_outputs = []
        self.build_output_shape = None
        self.slicer = None
        self.output_slice_index = None
        self.time_dim_unknown = False

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
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'nb_stacks': self.nb_stacks,
            'dilations': self.dilations,
            'padding': self.padding,
            'dropout': self.dropout,
            'use_skip_connections': self.use_skip_connections,
            'return_sequences': self.return_sequences,
            'activation': self.activation_name,
            'kernel_initializer': self.kernel_initializer,
            'use_batch_norm': self.use_batch_norm,
            'use_layer_norm': self.use_layer_norm,
            'go_backwards': self.go_backwards,
        })

        return config

    def build(self, input_shape):
        """
        Build the layer structure based on the input shape.

        Parameters
        ----------
        input_shape : tuple
            Shape tuple of the input tensor.
        """

        self.build_output_shape = input_shape
        self.residual_blocks = []

        for stack in range(self.nb_stacks):
            for i, dilation in enumerate(self.dilations):
                filters = self.filters[i] if isinstance(self.filters, list) else self.filters

                res_block = TemporalResidualBlock(filters=filters,
                                                  dilation_rate=dilation,
                                                  kernel_size=self.kernel_size,
                                                  padding=self.padding,
                                                  activation=self.activation_name,
                                                  dropout=self.dropout,
                                                  use_batch_norm=self.use_batch_norm,
                                                  use_layer_norm=self.use_layer_norm,
                                                  kernel_initializer=self.kernel_initializer,
                                                  name=f"residual_block_{stack}_{i}")

                res_block.build(self.build_output_shape)
                self.build_output_shape = res_block.res_output_shape
                self.residual_blocks.append(res_block)

        if self.padding == 'same':
            if self.build_output_shape.as_list()[1] is None:
                self.time_dim_unknown = True
            else:
                self.output_slice_index = int(self.build_output_shape.as_list()[1] / 2)
        else:
            self.output_slice_index = -1

        self.slicer = tf.keras.layers.Lambda(lambda x: x[:, self.output_slice_index, :], name='output_slice')
        self.slicer.build(self.build_output_shape.as_list())

    def call(self, inputs, training=None):
        """
        Call the layer with the given inputs and training flag.

        Parameters
        ----------
        inputs : tensor
            Input tensor.
        training : bool, optional
            Whether the layer should behave in training mode or in inference mode.

        Returns
        -------
        tensor
            The output tensor of the layer.
        """

        x = tf.reverse(inputs, axis=[1]) if self.go_backwards else inputs

        self.layers_outputs = [x]
        self.skip_connections = []

        for res_block in self.residual_blocks:
            x, skip_out = res_block(x, training=training)

            self.skip_connections.append(skip_out)
            self.layers_outputs.append(x)

        if self.use_skip_connections:
            if len(self.skip_connections) > 1:
                x = tf.keras.layers.Add()(self.skip_connections)
            else:
                x = self.skip_connections[0]

            self.layers_outputs.append(x)

        if not self.return_sequences:
            if self.time_dim_unknown:
                self.output_slice_index = tf.shape(self.layers_outputs[-1])[1] // 2

            x = self.slicer(x)
            self.layers_outputs.append(x)

        return x

    def compute_output_shape(self, input_shape):
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

        if not self.built:
            self.build(input_shape)

        output_shape = None

        if self.return_sequences:
            output_shape = [x.value if hasattr(x, 'value') else x for x in self.build_output_shape]
        else:
            batch_size = self.build_output_shape[0]
            batch_size = batch_size.value if hasattr(batch_size, 'value') else batch_size
            output_shape = [batch_size, self.build_output_shape[-1]]

        return tf.TensorShape(output_shape)


class TemporalResidualBlock(tf.keras.layers.Layer):
    """
    A Residual Block within the Temporal Convolutional Network.
    """

    def __init__(self,
                 filters,
                 dilation_rate,
                 kernel_size,
                 padding,
                 dropout=0.0,
                 activation='tanh',
                 kernel_initializer='glorot_normal',
                 use_batch_norm=False,
                 use_layer_norm=False,
                 **kwargs):
        """
        Initializes the residual block layer.

        Parameters
        ----------
        filters : int
            The dimensionality of the output space.
        dilation_rate : int
            The dilation rate to use for dilated convolution.
        kernel_size : int
            Length of the convolution window.
        padding : str
            One of 'valid' or 'same'.
        dropout : float
            Dropout rate.
        activation : str
            Activation function to use.
        kernel_initializer : str
            Initializer for the kernel weights matrix.
        use_batch_norm : bool
            Whether to use batch normalization.
        use_layer_norm : bool
            Whether to use layer normalization.
        """

        super().__init__(**kwargs)

        self.filters = filters
        self.dilation_rate = dilation_rate
        self.kernel_size = kernel_size
        self.padding = padding
        self.dropout = dropout
        self.activation = activation
        self.kernel_initializer = kernel_initializer
        self.use_batch_norm = use_batch_norm
        self.use_layer_norm = use_layer_norm

        self.layers = []
        self.shape_match_conv = None
        self.res_output_shape = None
        self.final_activation = None

    def _build_layer(self, layer):
        """
        Helper function to build a layer and update the output shape.

        Parameters
        ----------
        layer : tf.keras.layers.Layer
            The layer to build and add to the block.
        """

        layer.build(self.res_output_shape)
        self.res_output_shape = layer.compute_output_shape(self.res_output_shape)
        self.layers.append(layer)

    def build(self, input_shape):
        """
        Build the internal structure of the residual block based on the input shape.

        Parameters
        ----------
        input_shape : tuple
            Shape tuple of the input tensor.
        """

        super().build(input_shape)

        with tf.name_scope(self.name):
            self.res_output_shape = input_shape
            self.layers = []

            for i in range(2):
                with tf.name_scope(f"conv_block_{i}"):
                    conv = tf.keras.layers.Conv1D(filters=self.filters,
                                                  kernel_size=self.kernel_size,
                                                  dilation_rate=self.dilation_rate,
                                                  padding=self.padding,
                                                  kernel_initializer=self.kernel_initializer)
                    self._build_layer(conv)

                    if self.use_batch_norm:
                        self._build_layer(tf.keras.layers.BatchNormalization(renorm=True))
                    elif self.use_layer_norm:
                        self._build_layer(tf.keras.layers.LayerNormalization())

                    self._build_layer(tf.keras.layers.Activation(self.activation))
                    self._build_layer(tf.keras.layers.SpatialDropout1D(rate=self.dropout))

            if self.filters != input_shape[-1]:
                self.shape_match_conv = tf.keras.layers.Conv1D(filters=self.filters,
                                                               kernel_size=1,
                                                               padding='same',
                                                               kernel_initializer=self.kernel_initializer,
                                                               name='match_conv1D')
            else:
                self.shape_match_conv = tf.keras.layers.Lambda(lambda x: x, name='match_identity')

            self.shape_match_conv.build(input_shape)
            self.res_output_shape = self.shape_match_conv.compute_output_shape(input_shape)

            self._build_layer(tf.keras.layers.Activation(self.activation))
            self.final_activation = tf.keras.layers.Activation(self.activation)

    def call(self, inputs, training=None):
        """
        Call the residual block with the given inputs.

        Parameters
        ----------
        inputs : tensor
            Input tensor.
        training : bool, optional
            Whether the layer should behave in training mode or in inference mode.

        Returns
        -------
        list of tensors
            A list containing the output tensor and the skip connection output.
        """

        x = inputs
        for layer in self.layers:
            x = layer(x, training=training)

        x2 = self.shape_match_conv(inputs)
        x1_x2 = self.final_activation(tf.keras.layers.Add()([x2, x]))

        return [x1_x2, x]

    def compute_output_shape(self, input_shape):
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

        if not self.built:
            self.build(input_shape)

        return tf.TensorShape([self.res_output_shape, self.res_output_shape])
