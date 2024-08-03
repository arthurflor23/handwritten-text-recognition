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

    def __init__(self, spectral=False, momentum=0.99, epsilon=1e-3, **kwargs):
        """
        Initializes the conditional batch normalization layer.

        Parameters
        ----------
        spectral : bool, optional
            Whether apply spectral normalization or not.
        momentum : float, optional
            Momentum for the moving average of mean and variance.
        epsilon : float, optional
            Small float added to variance to avoid dividing by zero.
        **kwargs : dict
            Additional keyword arguments for the layer.
        """

        super().__init__(**kwargs)

        self.spectral = spectral
        self.momentum = momentum
        self.epsilon = epsilon
        self.mean = None
        self.variance = None

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
            'spectral': self.spectral,
            'momentum': self.momentum,
            'epsilon': self.epsilon,
            'mean': self.mean,
            'variance': self.variance,
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

        self.num_channels = input_shape[0][-1]

        self.gain = tf.keras.layers.Dense(self.num_channels, use_bias=False)
        self.bias = tf.keras.layers.Dense(self.num_channels, use_bias=False)

        if self.spectral:
            self.gain = tf.keras.layers.SpectralNormalization(self.gain)
            self.bias = tf.keras.layers.SpectralNormalization(self.bias)

        self.mean = self.add_weight(name=f"{self.name}_mean",
                                    shape=(self.num_channels,),
                                    initializer='zeros',
                                    trainable=False)

        self.variance = self.add_weight(name=f"{self.name}_variance",
                                        shape=(self.num_channels,),
                                        initializer='ones',
                                        trainable=False)

    def call(self, inputs, training=None):
        """
        Call the layer with the specified inputs.

        Parameters
        ----------
        inputs : list or tuple
            The inputs tensors (data and conditional data).
        training : bool, optional
            Whether the layer should behave in training or inference mode.

        Returns
        -------
        tf.Tensor
            The normalized output tensor.
        """

        inputs, conditional = inputs

        if training:
            mean, variance = tf.nn.moments(x=inputs, axes=[0, 1, 2], keepdims=False)

            self.mean.assign(self.mean * self.momentum + mean * (1 - self.momentum))
            self.variance.assign(self.variance * self.momentum + variance * (1 - self.momentum))

        else:
            mean = self.mean
            variance = self.variance

        gain = tf.reshape(1 + self.gain(conditional), shape=[tf.shape(conditional)[0], 1, 1, -1])
        bias = tf.reshape(self.bias(conditional), shape=[tf.shape(conditional)[0], 1, 1, -1])

        out = tf.nn.batch_normalization(x=inputs,
                                        mean=mean,
                                        variance=variance,
                                        offset=None,
                                        scale=None,
                                        variance_epsilon=self.epsilon)

        return out * gain + bias


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
                                               strides=[1] + self.patch_shape,
                                               rates=[1, 1, 1, 1],
                                               padding='VALID')

            x = tf.reshape(patches, shape=[-1] + self.patch_shape)

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
                 kernel_initializer='glorot_uniform',
                 kernel_regularizer=None,
                 kernel_constraint=None,
                 gamma_initializer='zeros',
                 dropout=0.0,
                 mode=None,
                 **kwargs):
        """
        Initializes the gated convolutional layer.

        Parameters
        ----------
        kernel_initializer : initializer, optional
            Kernel weights initializer.
        kernel_regularizer : regularizer, optional
            Kernel weights regularizer.
        kernel_constraint : constraint, optional
            Kernel weights constraint.
        gamma_initializer : initializer, optional
            Gamma weights initializer.
        dropout : float, optional
            Whether apply dropout or not.
        mode : str, optional
            Whether to use None, 'dual' or 'residual' gating.
        **kwargs : dict
            Conv2D keyword arguments.
        """

        super().__init__(**kwargs)

        self.kernel_initializer = kernel_initializer
        self.kernel_regularizer = kernel_regularizer
        self.kernel_constraint = kernel_constraint
        self.gamma_initializer = gamma_initializer
        self.dropout = dropout
        self.mode = mode

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
            'kernel_initializer': self.kernel_initializer,
            'kernel_regularizer': self.kernel_regularizer,
            'kernel_constraint': self.kernel_constraint,
            'gamma_initializer': self.gamma_initializer,
            'dropout': self.dropout,
            'mode': self.mode,
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

        self.filters = input_shape[-1]

        self.s_conv = tf.keras.layers.Conv2D(filters=self.filters * (2 if self.mode == 'dual' else 1),
                                             kernel_size=(3, 3),
                                             strides=(1, 1),
                                             padding='same',
                                             kernel_initializer=self.kernel_initializer,
                                             kernel_regularizer=self.kernel_regularizer,
                                             kernel_constraint=self.kernel_constraint)

        if self.mode == 'residual':
            self.t_conv = tf.keras.layers.Conv2D(filters=self.filters,
                                                 kernel_size=(3, 3),
                                                 strides=(1, 1),
                                                 padding='same',
                                                 kernel_initializer=self.kernel_initializer,
                                                 kernel_regularizer=self.kernel_regularizer,
                                                 kernel_constraint=self.kernel_constraint)

            self.gamma = self.add_weight(name=f"{self.name}_gamma",
                                         shape=(1,),
                                         initializer=self.gamma_initializer,
                                         trainable=True)

    def call(self, inputs, training=False):
        """
        Apply gated convolution to the input.

        Parameters
        ----------
        inputs : tensor
            The inputs to the layer.
        training : bool, optional
            Whether the layer should behave in training or inference mode.

        Returns
        -------
        tf.Tensor
            Tensor resulting from the gated convolution.
        """

        s = self.s_conv(inputs)

        if self.mode == 'dual':
            s1, s2 = tf.split(s, num_or_size_splits=2, axis=-1)
            linear = tf.keras.layers.Activation('linear')(s1)
            sigmoid = tf.keras.layers.Activation('sigmoid')(s2)
            outputs = linear * sigmoid

        elif self.mode == 'residual':
            t = self.t_conv(inputs)
            linear = tf.keras.layers.Activation('linear')(t)
            sigmoid = tf.keras.layers.Activation('sigmoid')(s)
            beta = self.gamma * linear * sigmoid

            if training and self.dropout:
                beta = tf.nn.dropout(beta, rate=self.dropout)

            outputs = beta + inputs

        else:
            sigmoid = tf.keras.layers.Activation('sigmoid')(s)
            outputs = sigmoid * inputs

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

        return [high_out_shape, low_out_shape]


class Reparameterization(tf.keras.layers.Layer):
    """
    Layer that applies the reparameterization trick for Gaussian sampling.

    This layer takes a list of tensors, composed of the mean (mu) and the log-variance (logvar),
        and returns a sampled latent variable by applying the reparameterization trick.

    References
    ----------
    Auto-Encoding Variational Bayes
        https://arxiv.org/abs/1312.6114
    """

    def call(self, inputs):
        """
        Applies the reparameterization trick on the input tensors.

        Parameters
        ----------
        inputs : list of tf.Tensor
            A list containing two tensors: mean (mu) and log-variance (logvar).

        Returns
        -------
        tf.Tensor
            A sampled latent variable after applying the reparameterization trick.
        """

        mu, logvar = tf.unstack(inputs)

        std = tf.exp(0.5 * logvar)
        eps = tf.random.normal(shape=tf.shape(mu))

        return eps * std + mu


class SelfAttention(tf.keras.layers.Layer):
    """
    Self-Attention layer for capturing long-range dependencies.

    References
    ----------
    Attention Is All You Need
        https://arxiv.org/abs/1706.03762

    Self-Attention Generative Adversarial Networks
        https://arxiv.org/abs/1805.08318

    Spectral Normalization for GANs
        https://arxiv.org/abs/1802.05957
    """

    def __init__(self,
                 kernel_initializer='glorot_uniform',
                 kernel_regularizer=None,
                 kernel_constraint=None,
                 gamma_initializer='zeros',
                 dropout=0.0,
                 pooling=True,
                 spectral=False,
                 **kwargs):
        """
        Initialize the self-attention gan layer.

        Parameters
        ----------
        kernel_initializer : initializer, optional
            Kernel weights initializer.
        kernel_regularizer : regularizer, optional
            Kernel weights regularizer.
        kernel_constraint : constraint, optional
            Kernel weights constraint.
        gamma_initializer : initializer, optional
            Gamma weights initializer.
        dropout : float, optional
            Whether apply dropout or not.
        pooling : bool, optional
            Whether apply max pooling or not.
        spectral : bool, optional
            Whether apply spectral normalization or not.
        **kwargs : dict
            Additional keyword arguments for the layer.
        """

        super().__init__(**kwargs)

        self.kernel_initializer = kernel_initializer
        self.kernel_regularizer = kernel_regularizer
        self.kernel_constraint = kernel_constraint
        self.gamma_initializer = gamma_initializer
        self.dropout = dropout
        self.pooling = pooling
        self.spectral = spectral

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
            'kernel_initializer': self.kernel_initializer,
            'kernel_regularizer': self.kernel_regularizer,
            'kernel_constraint': self.kernel_constraint,
            'gamma_initializer': self.gamma_initializer,
            'dropout': self.dropout,
            'pooling': self.pooling,
            'spectral': self.spectral,
        })

        return config

    def build(self, input_shape):
        """
        Build the layer structure based on the input shape.

        Parameters
        ----------
        input_shape : TensorShape
            Shape of the input tensor.
        """

        super().build(input_shape)

        if len(input_shape) == 3:
            pool_size = strides = 2 if input_shape[-2] > 1 else 1
            conv_layer = tf.keras.layers.Conv1D
            pooling_layer = tf.keras.layers.MaxPooling1D

        elif len(input_shape) == 4:
            pool_size = strides = (2 if input_shape[-3] > 1 else 1, 2 if input_shape[-2] > 1 else 1)
            conv_layer = tf.keras.layers.Conv2D
            pooling_layer = tf.keras.layers.MaxPooling2D

        else:
            raise ValueError('Unsupported input shape: must be 1D or 2D')

        self.filters = input_shape[-1]
        self.divisor = (2 if self.pooling else 1)

        self.f_conv = conv_layer(filters=self.filters // 8,
                                 kernel_size=1,
                                 padding='same',
                                 kernel_initializer=self.kernel_initializer,
                                 kernel_regularizer=self.kernel_regularizer,
                                 kernel_constraint=self.kernel_constraint,
                                 use_bias=False)

        self.g_conv = conv_layer(filters=self.filters // 8,
                                 kernel_size=1,
                                 padding='same',
                                 kernel_initializer=self.kernel_initializer,
                                 kernel_regularizer=self.kernel_regularizer,
                                 kernel_constraint=self.kernel_constraint,
                                 use_bias=False)

        self.h_conv = conv_layer(filters=self.filters // self.divisor,
                                 kernel_size=1,
                                 padding='same',
                                 kernel_initializer=self.kernel_initializer,
                                 kernel_regularizer=self.kernel_regularizer,
                                 kernel_constraint=self.kernel_constraint,
                                 use_bias=False)

        if self.pooling:
            self.f_pooling = pooling_layer(pool_size=pool_size, strides=strides)
            self.h_pooling = pooling_layer(pool_size=pool_size, strides=strides)

            self.o_conv = conv_layer(filters=self.filters,
                                     kernel_size=1,
                                     padding='same',
                                     kernel_initializer=self.kernel_initializer,
                                     kernel_regularizer=self.kernel_regularizer,
                                     kernel_constraint=self.kernel_constraint,
                                     use_bias=False)

            if self.spectral:
                self.o_conv = tf.keras.layers.SpectralNormalization(self.o_conv)

        if self.spectral:
            self.f_conv = tf.keras.layers.SpectralNormalization(self.f_conv)
            self.g_conv = tf.keras.layers.SpectralNormalization(self.g_conv)
            self.h_conv = tf.keras.layers.SpectralNormalization(self.h_conv)

        self.gamma = self.add_weight(name=f"{self.name}_gamma",
                                     shape=(1,),
                                     initializer=self.gamma_initializer,
                                     trainable=True)

    def call(self, inputs, training=False):
        """
        Processes the input tensors through the layer.

        Parameters
        ----------
        inputs : tf.Tensor
            Input tensor to the layer.
        training : bool, optional
            Whether the layer should behave in training or inference mode.

        Returns
        -------
        tf.Tensor
            Output tensor after applying self-attention.
        """

        shape = tf.unstack(tf.shape(inputs))

        f = self.f_conv(inputs)

        if self.pooling:
            f = self.f_pooling(f)

        f = tf.reshape(f, shape=(shape[0], -1, f.shape[-1]))

        g = self.g_conv(inputs)
        g = tf.reshape(g, shape=(shape[0], -1, g.shape[-1]))

        s = tf.matmul(g, f, transpose_b=True)
        beta = tf.nn.softmax(s, axis=-1)

        if training and self.dropout:
            beta = tf.nn.dropout(beta, rate=self.dropout)

        h = self.h_conv(inputs)

        if self.pooling:
            h = self.h_pooling(h)

        h = tf.reshape(h, shape=(shape[0], -1, h.shape[-1]))

        o = tf.matmul(beta, h)
        o = tf.reshape(o, shape=[shape[0]] + shape[1:-1] + [shape[-1] // self.divisor])

        if self.pooling:
            o = self.o_conv(o)

        return self.gamma * o + inputs
