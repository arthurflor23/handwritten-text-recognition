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

    def __init__(self, momentum=0.99, epsilon=1e-3, spectral_norm=False, **kwargs):
        """
        Initializes the conditional batch normalization layer.

        Parameters
        ----------
        momentum : float, optional
            Momentum for the moving average of mean and variance.
        epsilon : float, optional
            Small constant to avoid division by zero.
        spectral_norm : bool, optional
            Wheter apply spectral normalization or not.
        **kwargs : dict
            Additional keyword arguments for the layer.
        """

        super().__init__(**kwargs)

        self.momentum = momentum
        self.epsilon = epsilon
        self.spectral_norm = spectral_norm
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
            'momentum': self.momentum,
            'epsilon': self.epsilon,
            'spectral_norm': self.spectral_norm,
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

        if self.spectral_norm:
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
            Whether the layer should behave in training mode or in inference mode.

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
                 filters=None,
                 kernel_size=(3, 3),
                 strides=(1, 1),
                 padding='same',
                 kernel_initializer='glorot_uniform',
                 kernel_regularizer=None,
                 kernel_constraint=None,
                 dualgate=False,
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
        dualgate : bool, optional
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
        self.dualgate = dualgate

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
            'dualgate': self.dualgate,
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

        if self.filters is None:
            self.filters = input_shape[-1]

        self.conv = tf.keras.layers.Conv2D(filters=self.filters * (2 if self.dualgate else 1),
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

        if self.dualgate:
            linear, sigmoid = tf.split(conv, 2, axis=-1)
            linear = tf.keras.layers.Activation('linear')(linear)
            sigmoid = tf.keras.layers.Activation('sigmoid')(sigmoid)
            outputs = linear * sigmoid
        else:
            conv = tf.keras.layers.Activation('sigmoid')(conv)
            outputs = inputs * conv

        return outputs


class MaskPadding(tf.keras.layers.Layer):
    """
    Layer to mask padding in tensors.
    """

    def __init__(self, mask_value='min', pad_value='max', epsilon=1e-7, **kwargs):
        """
        Parameters
        ----------
        mask_value : float, int, or str, optional
            The value for masking input data.
        pad_value : float, int, or str, optional
            Value used for identify padding.
        epsilon : float, optional
            Small value for numerical stability.
        **kwargs : dict
            Additional keyword arguments for Layer.
        """

        super().__init__(**kwargs)

        self.mask_value = mask_value
        self.pad_value = pad_value
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
            'mask_value': self.mask_value,
            'pad_value': self.pad_value,
            'epsilon': self.epsilon,
        })

        return config

    def build(self, input_shape):
        """
        Initializes layer shapes.

        Parameters
        ----------
        input_shape : tuple
            Shape of the input to the layer.
        """

        super().build(input_shape)

        self.has_target_input = isinstance(input_shape, list)

        if self.has_target_input:
            self.origin_shape = input_shape[0]
            self.target_shape = input_shape[1]
        else:
            self.origin_shape = input_shape
            self.target_shape = input_shape

    def call(self, inputs):
        """
        Applies masking to inputs.

        Parameters
        ----------
        inputs : tuple of tf.Tensor
            Input and target data tensors.

        Returns
        -------
        tf.Tensor
            Target data with applied mask.
        """

        if self.has_target_input:
            input_data, target_data = inputs
        else:
            input_data = target_data = inputs

        mask_value = self._resolve_value(self.mask_value, input_data)
        pad_value = self._resolve_value(self.pad_value, input_data)

        v_mask = self._generate_mask(input_data, pad_value, transpose=False)
        h_mask = self._generate_mask(input_data, pad_value, transpose=True)

        mask = tf.cast(tf.logical_and(v_mask, h_mask), dtype=target_data.dtype)

        output = tf.where(tf.equal(mask, 0), mask_value, target_data)
        output = tf.stop_gradient(output)

        return output

    def _resolve_value(self, value, tensor):
        """
        Resolve a value that can be 'min', 'max' or a numeric value.

        Parameters
        ----------
        value : str, float or int
            The value to resolve.
        tensor : tf.Tensor
            The tensor to apply min/max operations on.

        Returns
        -------
        resolved_value : float or int
            The resolved numeric value.
        """

        if value == 'min':
            return tf.reduce_min(tensor)

        elif value == 'max':
            return tf.reduce_max(tensor)

        return tf.cast(value, tensor.dtype)

    def _generate_mask(self, input_data, pad_value, transpose=False):
        """
        Generate a mask for the input data based on the padding value.

        Parameters
        ----------
        input_data : tf.Tensor
            The input tensor.
        pad_value : float, or int,
            The padding value to identify in the input data.
        transpose : bool, optional
            Whether to transpose the input and target data.

        Returns
        -------
        tf.Tensor
            Boolean mask tensor.
        """

        origin_shape = self.origin_shape[1:-1]
        target_shape = self.target_shape[1:-1]

        if transpose:
            origin_shape = origin_shape[::-1]
            target_shape = target_shape[::-1]
            input_data = tf.transpose(input_data, perm=[0, 2, 1, 3])

        input_mean = tf.reduce_mean(input_data, axis=[2, 3])

        data_reversed = tf.reverse(input_mean, axis=[1])
        padding_mask = tf.equal(data_reversed, pad_value)

        lengths = tf.argmax(tf.cast(~padding_mask, tf.int32), axis=1, output_type=tf.int32)
        origin_lens = tf.where(tf.equal(lengths, 0), origin_shape[0], origin_shape[0] - lengths)

        scale = tf.math.divide(tf.cast(origin_lens, tf.float32), (origin_shape[0] + self.epsilon))
        target_lens = tf.math.multiply(tf.cast(target_shape[0], tf.float32), scale)

        mask = tf.sequence_mask(tf.math.ceil(target_lens), maxlen=target_shape[0])
        mask = tf.expand_dims(tf.expand_dims(mask, axis=-1), axis=-1)

        mask = tf.tile(mask, multiples=[1, 1, target_shape[1], 1])

        if transpose:
            mask = tf.transpose(mask, perm=[0, 2, 1, 3])

        return mask


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
    Self-Attention layer.

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
                 filters=None,
                 kernel_initializer='glorot_uniform',
                 kernel_regularizer=None,
                 kernel_constraint=None,
                 spectral_norm=False,
                 **kwargs):
        """
        Initialize the self-attention gan layer.

        Parameters
        ----------
        filters : int
            Number of output filters.
        kernel_initializer : initializer, optional
            Kernel weights initializer.
        kernel_regularizer : regularizer, optional
            Kernel weights regularizer.
        kernel_constraint : constraint, optional
            Kernel weights constraint.
        spectral_norm : bool, optional
            Wheter apply spectral normalization or not.
        **kwargs : dict
            Additional keyword arguments for the layer.
        """

        super().__init__(**kwargs)

        self.filters = filters
        self.kernel_initializer = kernel_initializer
        self.kernel_regularizer = kernel_regularizer
        self.kernel_constraint = kernel_constraint
        self.spectral_norm = spectral_norm

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
            'kernel_initializer': self.kernel_initializer,
            'kernel_regularizer': self.kernel_regularizer,
            'kernel_constraint': self.kernel_constraint,
            'spectral_norm': self.spectral_norm,
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

        if self.filters is None:
            self.filters = input_shape[-1]

        if len(input_shape) == 3:
            pool_size = 1
            conv_layer = tf.keras.layers.Conv1D
            pooling_layer = tf.keras.layers.MaxPooling1D

        elif len(input_shape) == 4:
            pool_size = 2 if input_shape[-3] > 1 and input_shape[-2] > 1 else 1
            conv_layer = tf.keras.layers.Conv2D
            pooling_layer = tf.keras.layers.MaxPooling2D

        else:
            raise ValueError('Unsupported input shape: must be 1D or 2D')

        self.f_conv = conv_layer(filters=self.filters // 8,
                                 kernel_size=1,
                                 padding='same',
                                 kernel_initializer=self.kernel_initializer,
                                 kernel_regularizer=self.kernel_regularizer,
                                 kernel_constraint=self.kernel_constraint,
                                 use_bias=False)

        self.f_pooling = pooling_layer(pool_size=pool_size, strides=pool_size, padding='valid')

        self.g_conv = conv_layer(filters=self.filters // 8,
                                 kernel_size=1,
                                 padding='same',
                                 kernel_initializer=self.kernel_initializer,
                                 kernel_regularizer=self.kernel_regularizer,
                                 kernel_constraint=self.kernel_constraint,
                                 use_bias=False)

        self.h_conv = conv_layer(filters=self.filters // 2,
                                 kernel_size=1,
                                 padding='same',
                                 kernel_initializer=self.kernel_initializer,
                                 kernel_regularizer=self.kernel_regularizer,
                                 kernel_constraint=self.kernel_constraint,
                                 use_bias=False)

        self.h_pooling = pooling_layer(pool_size=pool_size, strides=pool_size, padding='valid')

        self.o_conv = conv_layer(filters=self.filters,
                                 kernel_size=1,
                                 padding='same',
                                 kernel_initializer=self.kernel_initializer,
                                 kernel_regularizer=self.kernel_regularizer,
                                 kernel_constraint=self.kernel_constraint,
                                 use_bias=False)

        if self.spectral_norm:
            self.f_conv = tf.keras.layers.SpectralNormalization(self.f_conv)
            self.g_conv = tf.keras.layers.SpectralNormalization(self.g_conv)
            self.h_conv = tf.keras.layers.SpectralNormalization(self.h_conv)
            self.o_conv = tf.keras.layers.SpectralNormalization(self.o_conv)

        self.gamma = self.add_weight(name=f"{self.name}_gamma",
                                     shape=(1,),
                                     initializer='zeros',
                                     trainable=True)

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

        shape = tf.unstack(tf.shape(x))

        f = self.f_pooling(self.f_conv(x))
        f = tf.reshape(f, shape=(shape[0], -1, f.shape[-1]))

        g = self.g_conv(x)
        g = tf.reshape(g, shape=(shape[0], -1, g.shape[-1]))

        s = tf.matmul(g, f, transpose_b=True)
        beta = tf.nn.softmax(s, axis=-1)

        h = self.h_pooling(self.h_conv(x))
        h = tf.reshape(h, shape=(shape[0], -1, h.shape[-1]))

        o = tf.matmul(beta, h)
        o = tf.reshape(o, shape=[shape[0]] + shape[1:-1] + [shape[-1] // 2])

        o = self.o_conv(o)

        return self.gamma * o + x
