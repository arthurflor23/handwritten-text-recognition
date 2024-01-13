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

    def __init__(self, patch_shape=None, **kwargs):
        """
        Initializes Patches layer.

        Parameters
        ----------
        patch_shape : list, tuple or None
            The target patch size to create.
        **kwargs
            Additional keyword arguments for the Layer.
        """

        super().__init__(**kwargs)

        self.patch_shape = patch_shape

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

        x = inputs

        if self.patch_shape is not None:
            patches = tf.image.extract_patches(images=x,
                                               sizes=[1] + self.patch_shape,
                                               strides=[1, 8, 8, 1],
                                               rates=[1, 1, 1, 1],
                                               padding='VALID')

            x = tf.reshape(patches, shape=[-1] + self.patch_shape)

            patch_means = tf.reduce_mean(x, axis=[1, 2, 3])
            mask = tf.not_equal(patch_means, 1.0)

            x = tf.cond(pred=tf.reduce_any(mask),
                        true_fn=lambda: tf.boolean_mask(x, mask),
                        false_fn=lambda: x)

            indices = tf.random.shuffle(tf.range(tf.shape(x)[0]))
            indices = tf.stop_gradient(indices[:tf.shape(inputs)[0]])

            x = tf.gather(x, indices)

        return x


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


class SelfAttentionGan(tf.keras.layers.Layer):
    """
    Self-Attention GAN layer with spectral normalization on convolutional layers.

    References
    ----------
    Attention Is All You Need
        https://arxiv.org/abs/1706.03762

    Self-Attention Generative Adversarial Networks
        https://arxiv.org/abs/1805.08318

    Spectral Normalization for GANs
        https://arxiv.org/abs/1802.05957
    """

    def __init__(self, **kwargs):
        """
        Initialize the self-attention gan layer.

        Parameters
        ----------
        **kwargs : dict
            Additional keyword arguments for the layer.
        """

        super().__init__(**kwargs)

    def build(self, input_shape):
        """
        Build the layer structure based on the input shape.

        Parameters
        ----------
        input_shape : TensorShape
            Shape of the input tensor.
        """

        self.shape = input_shape

        self.q_filters = self.shape[-1] // 8
        self.k_filters = self.shape[-1] // 8
        self.v_filters = self.shape[-1]

        self.query_conv = SpectralNormalization(
            tf.keras.layers.Conv2D(self.q_filters, kernel_size=1, padding='same', use_bias=False))

        self.key_conv = SpectralNormalization(
            tf.keras.layers.Conv2D(self.k_filters, kernel_size=1, padding='same', use_bias=False))

        self.value_conv = SpectralNormalization(
            tf.keras.layers.Conv2D(self.v_filters, kernel_size=1, padding='same', use_bias=False))

        self.gamma = self.add_weight(shape=(1,), initializer='zeros', name=f"{self.name}_gamma", trainable=True)

    def call(self, x):
        """
        Processes the input tensors through the layer.

        Parameters
        ----------
        x : tf.Tensor
            Input tensor to the layer.

        Returns
        -------
        tf.Tensor
            Output tensor after applying self-attention.
        """

        _, height, width, channels = self.shape

        proj_query = self.query_conv(x)
        proj_query = tf.nn.relu(proj_query)
        proj_query = tf.reshape(proj_query, [-1, self.q_filters, height * width])
        proj_query = tf.transpose(proj_query, [0, 2, 1])

        proj_key = self.key_conv(x)
        proj_key = tf.nn.relu(proj_key)
        proj_key = tf.reshape(proj_key, [-1, self.k_filters, height * width])

        energy = tf.matmul(proj_query, proj_key)
        attention = tf.nn.softmax(energy, axis=-1)

        proj_value = self.value_conv(x)
        proj_value = tf.reshape(proj_value, [-1, self.v_filters, height * width])

        out = tf.matmul(proj_value, attention, transpose_b=True)
        out = tf.reshape(out, [-1, height, width, channels])

        return self.gamma * out + x


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
            kernel_matrix = tf.reshape(self.kernel, [-1, self.kernel_shape[-1]])
            vector_u = self.vector_u

            if not tf.reduce_all(tf.equal(kernel_matrix, 0.0)):
                for _ in range(self.power_iterations):
                    vector_v = tf.math.l2_normalize(tf.matmul(vector_u, kernel_matrix, transpose_b=True))
                    vector_u = tf.math.l2_normalize(tf.matmul(vector_v, kernel_matrix))

                vector_u = tf.stop_gradient(vector_u)
                vector_v = tf.stop_gradient(vector_v)

                sigma = tf.matmul(tf.matmul(vector_v, kernel_matrix), vector_u, transpose_b=True)

                self.vector_u.assign(tf.cast(vector_u, self.vector_u.dtype))
                self.kernel.assign(tf.cast(tf.reshape(self.kernel / sigma, self.kernel_shape), self.kernel.dtype))

        output = self.layer(inputs)
        return output

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
                 dilations=(1, 2, 4),
                 padding='causal',
                 dropout=0.0,
                 use_skip_connections=True,
                 return_sequences=False,
                 activation='relu',
                 kernel_initializer='glorot_uniform',
                 use_batch_norm=False,
                 use_layer_norm=False,
                 use_weight_norm=False,
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
        use_weight_norm : bool
            Whether to use weight normalization.
        go_backwards : bool
            Process the input sequence backwards and return the reversed sequence.
        **kwargs : dict
            Additional keyword arguments for the layer.
        """

        super().__init__(**kwargs)

        if sum([use_batch_norm, use_layer_norm, use_weight_norm]) > 1:
            raise ValueError('Only one normalization can be used.')

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
        self.use_weight_norm = use_weight_norm
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
                                                  use_weight_norm=self.use_weight_norm,
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
                 dropout,
                 activation,
                 kernel_initializer,
                 use_batch_norm,
                 use_layer_norm,
                 use_weight_norm,
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
        use_weight_norm : bool
            Whether to use weight normalization.
        **kwargs : dict
            Additional keyword arguments for the layer.
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
        self.use_weight_norm = use_weight_norm

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
            'data_init': self.data_init,
        })

        return config

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

            if self.activation == 'prelu':
                activation = tf.keras.layers.PReLU()
            else:
                activation = tf.keras.layers.Activation(self.activation)

            for i in range(2):
                with tf.name_scope(f"conv_block_{i}"):
                    conv = tf.keras.layers.Conv1D(filters=self.filters,
                                                  kernel_size=self.kernel_size,
                                                  dilation_rate=self.dilation_rate,
                                                  padding=self.padding,
                                                  kernel_initializer=self.kernel_initializer)

                    if self.use_weight_norm:
                        conv = WeightNormalization(conv)

                    self._build_layer(conv)
                    self._build_layer(activation)

                    if self.use_batch_norm:
                        self._build_layer(tf.keras.layers.BatchNormalization(renorm=True))

                    elif self.use_layer_norm:
                        self._build_layer(tf.keras.layers.LayerNormalization())

                    self._build_layer(tf.keras.layers.Dropout(rate=self.dropout))

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

            self._build_layer(activation)
            self.final_activation = activation

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


class WeightNormalization(tf.keras.layers.Wrapper):
    """
    Performs weight normalization.
    This wrapper reparameterizes a layer by decoupling the weight's magnitude and direction.
    This speeds up convergence by improving the conditioning of the optimization problem.

    References
    ----------
    Weight Normalization: A Simple Reparameterization to Accelerate Training of Deep Neural Networks
        https://arxiv.org/abs/1602.07868
    """

    def __init__(self, layer, data_init=True, **kwargs):
        """
        Initializes the weight normalization wrapper.

        Parameters
        ----------
        layer : tf.keras.layers.Layer
            The layer to be wrapped.
        data_init : bool, optional
            Whether to use data-dependent initialization.
        **kwargs : dict
            Additional keyword arguments for the wrapper.

        Raises
        ------
        ValueError
            If data_init is True and the layer is an RNN, as advised against in the paper.
        """

        super().__init__(layer, **kwargs)

        self.data_init = data_init
        self._track_trackable(layer, name='layer')
        self.is_rnn = isinstance(self.layer, tf.keras.layers.RNN)

        if self.data_init and self.is_rnn:
            raise ValueError('WeightNormalization: Using `data_init=True` with RNNs '
                             'is advised against by the paper. Use `data_init=False`.')

    def build(self, input_shape):
        """
        Build the WeightNormalization wrapper.

        Parameters
        ----------
        input_shape : tuple
            The expected input shape for the layer.
        """

        input_shape = tf.TensorShape(input_shape)
        self.input_spec = tf.keras.layers.InputSpec(shape=[None] + input_shape[1:])

        if not self.layer.built:
            self.layer.build(input_shape)

        kernel_layer = self.layer.cell if self.is_rnn else self.layer

        if not hasattr(kernel_layer, 'kernel'):
            raise ValueError('`WeightNormalization` must wrap a layer that '
                             'contains a `kernel` for weights')

        if self.is_rnn:
            kernel = kernel_layer.recurrent_kernel
        else:
            kernel = kernel_layer.kernel

        self.layer_depth = int(kernel.shape[-1])
        self.kernel_norm_axes = list(range(kernel.shape.rank - 1))

        self.v = kernel

        self.g = self.add_weight(name='g',
                                 shape=(self.layer_depth,),
                                 initializer='ones',
                                 dtype=kernel.dtype,
                                 trainable=True)

        self._initialized = self.add_weight(name='initialized',
                                            shape=None,
                                            initializer='zeros',
                                            dtype=tf.dtypes.bool,
                                            trainable=False)

        if self.data_init:
            with tf.name_scope('data_dep_init'):
                layer_config = tf.keras.layers.serialize(self.layer)
                layer_config['config']['trainable'] = False

                self._naked_clone_layer = tf.keras.layers.deserialize(layer_config)
                self._naked_clone_layer.build(input_shape)
                self._naked_clone_layer.set_weights(self.layer.get_weights())

                if not self.is_rnn:
                    self._naked_clone_layer.activation = None

        self.built = True

    def call(self, inputs):
        """
        Call the wrapped layer with inputs.

        Parameters
        ----------
        inputs : tensor
            Input tensor.

        Returns
        -------
        tensor
            The output tensor of the layer.
        """

        def _do_nothing():
            return tf.identity(self.g)

        def _update_weights():
            with tf.control_dependencies(self._initialize_weights(inputs)):
                return tf.identity(self.g)

        g = tf.cond(self._initialized, _do_nothing, _update_weights)

        with tf.name_scope('compute_weights'):
            kernel = tf.nn.l2_normalize(self.v, axis=self.kernel_norm_axes) * g

            if self.is_rnn:
                self.layer.cell.recurrent_kernel = kernel
                update_kernel = tf.identity(self.layer.cell.recurrent_kernel)
            else:
                self.layer.kernel = kernel
                update_kernel = tf.identity(self.layer.kernel)

            with tf.control_dependencies([update_kernel]):
                outputs = self.layer(inputs)
                return outputs

    def _initialize_weights(self, inputs):
        """
        Initialize the weights of the wrapped layer.

        Parameters
        ----------
        inputs : tensor
            Input tensor for data-dependent initialization.

        Returns
        -------
        list of tensors
            A list of tensors for weight initialization.
        """

        dependencies = [tf.debugging.assert_equal(self._initialized, False, message='The layer has been initialized.')]

        with tf.control_dependencies(dependencies):
            if self.data_init:
                with tf.name_scope('data_dep_init'):
                    x_init = self._naked_clone_layer(inputs)
                    data_norm_axes = list(range(x_init.shape.rank - 1))
                    m_init, v_init = tf.nn.moments(x_init, data_norm_axes)
                    scale_init = 1.0 / tf.math.sqrt(v_init + 1e-10)

                    if scale_init.shape[0] != self.g.shape[0]:
                        rep = int(self.g.shape[0] / scale_init.shape[0])
                        scale_init = tf.tile(scale_init, [rep])

                    g_tensor = self.g.assign(self.g * scale_init)

                    if hasattr(self.layer, 'bias') and self.layer.bias is not None:
                        bias_tensor = self.layer.bias.assign(-m_init * scale_init)
                        assign_tensors = [g_tensor, bias_tensor]
                    else:
                        assign_tensors = [g_tensor]
            else:
                with tf.name_scope('init_norm'):
                    v_flat = tf.reshape(self.v, [-1, self.layer_depth])
                    v_norm = tf.linalg.norm(v_flat, axis=0)
                    g_tensor = self.g.assign(tf.reshape(v_norm, (self.layer_depth,)))
                    assign_tensors = [g_tensor]

            assign_tensors.append(self._initialized.assign(True))

            return assign_tensors

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
            The computed shape of the output from the layer.
        """

        if not self.built:
            self.build(input_shape)

        return tf.TensorShape(self.layer.compute_output_shape(input_shape).as_list())
