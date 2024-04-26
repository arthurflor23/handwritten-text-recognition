import numpy as np
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

    def __init__(self, spectral_norm=False, momentum=0.99, epsilon=1e-3, **kwargs):
        """
        Initializes the conditional batch normalization layer.

        Parameters
        ----------
        spectral_norm : bool, optional
            Wheter apply spectral normalization or not.
        momentum : float, optional
            Momentum for the moving average of mean and variance.
        epsilon : float, optional
            Small constant to avoid division by zero.
        **kwargs : dict
            Additional keyword arguments for the layer.
        """

        super().__init__(**kwargs)

        self.spectral_norm = spectral_norm
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
            'spectral_norm': self.spectral_norm,
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

        self.gain = tf.keras.layers.Dense(self.num_channels, use_bias=False)
        self.bias = tf.keras.layers.Dense(self.num_channels, use_bias=False)

        if self.spectral_norm:
            self.gain = SpectralNormalization(self.gain)
            self.bias = SpectralNormalization(self.bias)

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
                 spectral_norm=False,
                 kernel_initializer='glorot_uniform',
                 kernel_regularizer=None,
                 kernel_constraint=None,
                 **kwargs):
        """
        Initialize the self-attention gan layer.

        Parameters
        ----------
        filters : int
            Number of output filters.
        spectral_norm : bool, optional
            Wheter apply spectral normalization or not.
        kernel_initializer : initializer, optional
            Kernel weights initializer.
        kernel_regularizer : regularizer, optional
            Kernel weights regularizer.
        kernel_constraint : constraint, optional
            Kernel weights constraint.
        **kwargs : dict
            Additional keyword arguments for the layer.
        """

        super().__init__(**kwargs)

        self.filters = filters
        self.spectral_norm = spectral_norm
        self.kernel_initializer = kernel_initializer
        self.kernel_regularizer = kernel_regularizer
        self.kernel_constraint = kernel_constraint

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
            'spectral_norm': self.spectral_norm,
            'kernel_initializer': self.kernel_initializer,
            'kernel_regularizer': self.kernel_regularizer,
            'kernel_constraint': self.kernel_constraint,
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
            self.f_conv = SpectralNormalization(self.f_conv)
            self.g_conv = SpectralNormalization(self.g_conv)
            self.h_conv = SpectralNormalization(self.h_conv)
            self.o_conv = SpectralNormalization(self.o_conv)

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


class SpectralNormalization(tf.keras.layers.Wrapper):
    """
    Spectral Normalization for TensorFlow models.
    Optimizes GAN training stability by normalizing layer weights.

    References
    ----------
    Spectral Norm Regularization for Improving the Generalizability of Deep Learning.
        https://arxiv.org/abs/1705.10941

    Spectral Normalization for GANs
        https://arxiv.org/abs/1802.05957

    Regularisation of neural networks by enforcing lipschitz continuity.
        https://arxiv.org/abs/1804.04368
    """

    def __init__(self,
                 layer,
                 power_iteration=1,
                 norm_multiplier=0.95,
                 aggregation=tf.VariableAggregation.MEAN,
                 **kwargs):
        """
        Initializes the spectral normalization wrapper.

        Parameters
        ----------
        layer : tf.keras.layers.Layer
            Keras layer to be normalized.
        power_iteration : int, optional
            Number of power iterations for singular value estimation.
        norm_multiplier : float, optional
            Threshold for normalization.
        aggregation : tf.VariableAggregation, optional
            Aggregation method for distributed variables.
        **kwargs : dict
            Additional arguments for layers.Wrapper class.
        """

        super().__init__(layer, name=layer.name, **kwargs)

        self.power_iteration = power_iteration
        self.norm_multiplier = norm_multiplier
        self.aggregation = aggregation

        if not isinstance(layer, tf.keras.layers.Layer):
            raise ValueError('`layer` must be a `tf.keras.layer.Layer`.')

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
            'power_iteration': self.power_iteration,
            'norm_multiplier': self.norm_multiplier,
            'aggregation': self.aggregation,
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

        self._dtype = self.layer.kernel.dtype
        self.layer.kernel._aggregation = self.aggregation

        self.w = self.layer.kernel
        self.w_shape = self.w.shape.as_list()

        self.v = self.add_weight(name='v',
                                 shape=(1, np.prod(self.w_shape[:-1])),
                                 initializer=tf.initializers.random_normal(mean=0.0, stddev=0.02),
                                 aggregation=self.aggregation,
                                 dtype=self.dtype,
                                 trainable=False,)

        self.u = self.add_weight(name='u',
                                 shape=(1, self.w_shape[-1]),
                                 initializer=tf.initializers.random_normal(mean=0.0, stddev=0.02),
                                 aggregation=self.aggregation,
                                 dtype=self.dtype,
                                 trainable=False,)

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

        u_update_op, v_update_op, w_update_op = self.update_weights(training=training)
        output = self.layer(inputs)
        w_restore_op = self.restore_weights()

        self.add_update(u_update_op)
        self.add_update(v_update_op)
        self.add_update(w_update_op)
        self.add_update(w_restore_op)

        return output

    def update_weights(self, training=None):
        """
        Updates the weights of the wrapped layer.

        Parameters
        ----------
        training : bool, optional
            If True, performs power iteration to update weights.

        Returns
        -------
        tuple
            Update operations for u, v, and w weights.
        """

        w_reshaped = tf.reshape(self.w, [-1, self.w_shape[-1]])

        u_hat = self.u
        v_hat = self.v

        if training:
            for _ in range(self.power_iteration):
                v_hat = tf.nn.l2_normalize(tf.matmul(u_hat, tf.transpose(w_reshaped)))
                u_hat = tf.nn.l2_normalize(tf.matmul(v_hat, w_reshaped))

        sigma = tf.matmul(tf.matmul(v_hat, w_reshaped), tf.transpose(u_hat))
        sigma = tf.reshape(sigma, [])

        u_update_op = self.u.assign(u_hat)
        v_update_op = self.v.assign(v_hat)

        w_norm = tf.cond((self.norm_multiplier / sigma) < 1,
                         lambda: (self.norm_multiplier / sigma) * self.w, lambda: self.w)

        w_update_op = self.layer.kernel.assign(w_norm)

        return u_update_op, v_update_op, w_update_op

    def restore_weights(self):
        """
        Restores the weights of the layer after updates.

        Returns
        -------
        tf.Operation
            An operation that restores the weights.
        """

        return self.layer.kernel.assign(self.w)
