import tensorflow as tf


class Bidirectional(tf.keras.layers.Layer):
    """
    A bidirectional wrapper for RNN layers.

    This layer processes the input in both forward and backward directions using
        two RNN layers and combines their outputs based on the specified merge mode.
    """

    def __init__(self, layer, dropout=0.0, merge_mode='concat', **kwargs):
        """
        Initializes the Bidirectional layer.

        Parameters
        ----------
        layer : tf.keras.layers.RNN or tf.keras.layers.Layer
            RNN layer instance (e.g., `LSTM`, `GRU`) or any layer that meets
            the required criteria for bidirectional processing.
        dropout : float, optional
            Dropout rate applied to the inputs.
        merge_mode : str or None, optional
            Mode to merge outputs from forward and backward RNNs.
            Options: {"sum", "mul", "concat", "ave", None}.
        **kwargs : dict
            Additional keyword arguments for the layer.
        """

        super().__init__(**kwargs)

        if not isinstance(layer, tf.keras.layers.Layer):
            raise ValueError(
                "Please initialize `Bidirectional` layer with a "
                f"`keras.layers.Layer` instance. Received: {layer}"
            )

        if merge_mode not in ['sum', 'mul', 'ave', 'concat', None]:
            raise ValueError(
                f"Invalid merge mode. Received: {merge_mode}. "
                "Merge mode should be one of "
                '{"sum", "mul", "concat", "ave", None}'
            )

        config = tf.keras.utils.serialize_keras_object(layer)
        config['config']['name'] = 'forward_' + layer.name.lstrip('forward_')
        config['config']['go_backwards'] = False
        self.forward_layer = tf.keras.utils.deserialize_keras_object(config)

        config = tf.keras.utils.serialize_keras_object(layer)
        config['config']['name'] = 'backward_' + layer.name.lstrip('backward_')
        config['config']['go_backwards'] = True
        self.backward_layer = tf.keras.utils.deserialize_keras_object(config)

        if self.forward_layer.go_backwards == self.backward_layer.go_backwards:
            raise ValueError(
                "Forward layer and backward layer should have different "
                "`go_backwards` value. Received: "
                "forward_layer.go_backwards "
                f"{self.forward_layer.go_backwards}, "
                "backward_layer.go_backwards="
                f"{self.backward_layer.go_backwards}"
            )

        for a in ('stateful', 'return_sequences', 'return_state'):
            forward_value = getattr(self.forward_layer, a)
            backward_value = getattr(self.backward_layer, a)

            if forward_value != backward_value:
                raise ValueError(
                    "Forward layer and backward layer are expected to have "
                    f'the same value for attribute "{a}", got '
                    f'"{forward_value}" for forward layer and '
                    f'"{backward_value}" for backward layer'
                )

        if getattr(self.forward_layer, 'zero_output_for_mask', None) is not None:
            self.forward_layer.zero_output_for_mask = self.forward_layer.return_sequences

        if getattr(self.backward_layer, 'zero_output_for_mask', None) is not None:
            self.backward_layer.zero_output_for_mask = self.backward_layer.return_sequences

        self.dropout = dropout
        self.merge_mode = merge_mode

        self.stateful = layer.stateful
        self.return_sequences = layer.return_sequences
        self.return_state = layer.return_state
        self.input_spec = layer.input_spec
        self.supports_masking = True

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
            'layer': tf.keras.utils.serialize_keras_object(self.forward_layer),
            'backward_layer': tf.keras.utils.serialize_keras_object(self.backward_layer),
            'dropout': self.dropout,
            'merge_mode': self.merge_mode,
        })

        return config

    def build(self, inputs_shape):
        """
        Initializes layer weights.

        Parameters
        ----------
        input_shape : tuple
            Shape of the input to the layer.
        """

        if not self.forward_layer.built:
            self.forward_layer.build(inputs_shape)

        if not self.backward_layer.built:
            self.backward_layer.build(inputs_shape)

        self.built = True

    def call(self, inputs, initial_state=None, mask=None, training=False):
        """
        Invokes the layer with inputs and optional arguments.

        Parameters
        ----------
        inputs : tf.Tensor
            Input tensor.
        initial_state : list of tf.Tensor, optional
            Initial states for forward and backward RNNs.
        mask : tf.Tensor, optional
            Mask tensor.
        training : bool, optional
            Whether in training mode.

        Returns
        -------
        tf.Tensor or tuple
            Output tensor or tuple of output and states.
        """

        kwargs = {}

        if self.forward_layer._call_has_training_arg:
            kwargs['training'] = training

        if self.forward_layer._call_has_mask_arg:
            kwargs['mask'] = mask

        if initial_state is not None:
            forward_inputs, backward_inputs = inputs, inputs
            half = len(initial_state) // 2
            forward_state = initial_state[:half]
            backward_state = initial_state[half:]
        else:
            forward_inputs, backward_inputs = inputs, inputs
            forward_state, backward_state = None, None

        if training and self.dropout:
            forward_inputs = tf.keras.random.dropout(forward_inputs, rate=self.dropout)
            backward_inputs = tf.keras.random.dropout(backward_inputs, rate=self.dropout)

        y = self.forward_layer(forward_inputs, initial_state=forward_state, **kwargs)
        y_rev = self.backward_layer(backward_inputs, initial_state=backward_state, **kwargs)

        if self.return_state:
            states = tuple(y[1:] + y_rev[1:])
            y = y[0]
            y_rev = y_rev[0]

        y = tf.keras.ops.cast(y, self.compute_dtype)
        y_rev = tf.keras.ops.cast(y_rev, self.compute_dtype)

        if self.return_sequences:
            y_rev = tf.keras.ops.flip(y_rev, axis=1)

        if self.merge_mode == 'concat':
            output = tf.keras.ops.concatenate([y, y_rev], axis=-1)
        elif self.merge_mode == 'sum':
            output = y + y_rev
        elif self.merge_mode == 'ave':
            output = (y + y_rev) / 2
        elif self.merge_mode == 'mul':
            output = y * y_rev
        elif self.merge_mode is None:
            output = (y, y_rev)
        else:
            raise ValueError(
                "Unrecognized value for `merge_mode`. "
                f"Received: {self.merge_mode}"
                'Expected one of {"concat", "sum", "ave", "mul"}.'
            )

        if self.return_state:
            if self.merge_mode is None:
                return output + states
            return (output,) + states

        return output

    def compute_output_shape(self, inputs_shape):
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

        output_shape = self.forward_layer.compute_output_shape(inputs_shape)

        if self.return_state:
            output_shape, state_shape = output_shape[0], output_shape[1:]

        if self.merge_mode == 'concat':
            output_shape = list(output_shape[:-1]) + [output_shape[-1] * 2]

        elif self.merge_mode is None:
            output_shape = [output_shape, output_shape]

        if self.return_state:
            if self.merge_mode is None:
                return tuple(output_shape) + state_shape + state_shape
            return tuple([output_shape]) + (state_shape) + (state_shape)

        return tuple(output_shape)

    def reset_states(self):
        """
        Resets the states of the forward and backward layers.
        """

        self.reset_state()

    def reset_state(self):
        """
        Resets the state of the forward and backward layers.

        Raises
        ------
        AttributeError
            If the layer is not stateful.
        """

        if not self.stateful:
            raise AttributeError("Layer must be stateful.")

        self.forward_layer.reset_state()
        self.backward_layer.reset_state()

    def compute_mask(self, _, mask):
        """
        Computes the output mask tensor.

        Parameters
        ----------
        inputs : tf.Tensor
            Input tensor.
        mask : tf.Tensor or list, optional
            Input mask.

        Returns
        -------
        tf.Tensor or tuple
            Output mask tensor or tuple of masks.
        """

        if isinstance(mask, list):
            mask = mask[0]

        if self.return_sequences:
            if not self.merge_mode:
                output_mask = (mask, mask)
            else:
                output_mask = mask
        else:
            output_mask = (None, None) if not self.merge_mode else None

        if self.return_state and self.states is not None:
            state_mask = (None for _ in self.states)

            if isinstance(output_mask, list):
                return output_mask + state_mask * 2
            return (output_mask,) + state_mask * 2

        return output_mask

    @property
    def states(self):
        """
        Returns the states of both layers.

        Returns
        -------
        tuple of tf.Tensor or None
            Tuple of states or `None`.
        """

        if self.forward_layer.states and self.backward_layer.states:
            return tuple(self.forward_layer.states + self.backward_layer.states)

        return None

    @classmethod
    def from_config(self, config, custom_objects=None):
        """
        Creates an instance of the layer from its config.

        Parameters
        ----------
        config : dict
            Dictionary containing the configuration of the layer.
        custom_objects : dict, optional
            Dictionary mapping class names to custom classes or functions.

        Returns
        -------
        Bidirectional
            A Bidirectional layer instance created from the provided config.
        """

        config = config.copy()

        config['layer'] = tf.keras.utils.deserialize_keras_object(config=config['layer'],
                                                                  custom_objects=custom_objects)

        backward_layer_config = config.pop('backward_layer', None)

        if backward_layer_config is not None:
            backward_layer = tf.keras.utils.deserialize_keras_object(config=backward_layer_config,
                                                                     custom_objects=custom_objects)
            config['backward_layer'] = backward_layer

        layer = self(**config)

        return layer


class ConditionalBatchNormalization(tf.keras.layers.Layer):
    """
    Conditional Batch Normalization for TensorFlow models.
    Enhances conditional GANs by using unique parameters for each condition.

    References
    ----------
    Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift
        https://arxiv.org/abs/1502.03167

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

        self.beta = tf.keras.layers.Dense(self.num_channels, use_bias=False)
        self.gamma = tf.keras.layers.Dense(self.num_channels, use_bias=False)

        if self.spectral:
            self.beta = tf.keras.layers.SpectralNormalization(self.beta)
            self.gamma = tf.keras.layers.SpectralNormalization(self.gamma)

        self.mean = self.add_weight(name=f"{self.name}_mean",
                                    shape=(self.num_channels,),
                                    initializer='zeros',
                                    trainable=False)

        self.variance = self.add_weight(name=f"{self.name}_variance",
                                        shape=(self.num_channels,),
                                        initializer='ones',
                                        trainable=False)

    def call(self, inputs, training=False):
        """
        Call the layer with the specified inputs.

        Parameters
        ----------
        inputs : tf.Tensor
            Input tensor.
        training : bool, optional
            Whether in training mode.

        Returns
        -------
        tf.Tensor
            The normalized output tensor.
        """

        x, conditional = inputs

        beta = self.beta(conditional)
        gamma = self.gamma(conditional)

        beta = tf.reshape(beta, shape=[-1, 1, 1, self.num_channels])
        gamma = tf.reshape(gamma, shape=[-1, 1, 1, self.num_channels])

        if training:
            mean, variance = tf.nn.moments(x=x, axes=[0, 1, 2], keepdims=False)

            self.mean.assign(self.mean * self.momentum + mean * (1 - self.momentum))
            self.variance.assign(self.variance * self.momentum + variance * (1 - self.momentum))

        else:
            mean = self.mean
            variance = self.variance

        outputs = tf.nn.batch_normalization(x=x,
                                            mean=mean,
                                            variance=variance,
                                            offset=beta,
                                            scale=gamma,
                                            variance_epsilon=self.epsilon)

        return outputs


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
                 mode=None,
                 spectral=False,
                 kernel_initializer='glorot_uniform',
                 kernel_regularizer=None,
                 kernel_constraint=None,
                 gamma_initializer='zeros',
                 dropout=0.0,
                 **kwargs):
        """
        Initializes the gated convolutional layer.

        Parameters
        ----------
        mode : str, optional
            Whether to use None, 'dual' or 'residual' gating.
        spectral : bool, optional
            Whether apply spectral normalization or not.
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
        **kwargs : dict
            Conv2D keyword arguments.
        """

        super().__init__(**kwargs)

        self.mode = mode
        self.spectral = spectral
        self.kernel_initializer = kernel_initializer
        self.kernel_regularizer = kernel_regularizer
        self.kernel_constraint = kernel_constraint
        self.gamma_initializer = gamma_initializer
        self.dropout = dropout

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
            'mode': self.mode,
            'spectral': self.spectral,
            'kernel_initializer': self.kernel_initializer,
            'kernel_regularizer': self.kernel_regularizer,
            'kernel_constraint': self.kernel_constraint,
            'gamma_initializer': self.gamma_initializer,
            'dropout': self.dropout,
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
                                             kernel_constraint=self.kernel_constraint,
                                             use_bias=True)

        if self.spectral:
            self.s_conv = tf.keras.layers.SpectralNormalization(self.s_conv, name=self.s_conv.name)

        if self.mode == 'residual':
            self.gamma = self.add_weight(name=f"{self.name}_gamma",
                                         shape=(1,),
                                         initializer=self.gamma_initializer,
                                         trainable=True)

            self.l_conv = tf.keras.layers.Conv2D(filters=self.filters,
                                                 kernel_size=(3, 3),
                                                 strides=(1, 1),
                                                 padding='same',
                                                 kernel_initializer=self.kernel_initializer,
                                                 kernel_regularizer=self.kernel_regularizer,
                                                 kernel_constraint=self.kernel_constraint,
                                                 use_bias=True)

            if self.spectral:
                self.l_conv = tf.keras.layers.SpectralNormalization(self.l_conv, name=self.l_conv.name)

    def call(self, inputs, training=False):
        """
        Apply gated convolution to the input.

        Parameters
        ----------
        inputs : tf.Tensor
            Input tensor.
        training : bool, optional
            Whether in training mode.

        Returns
        -------
        tf.Tensor
            Tensor resulting from the gated convolution.
        """

        s_conv = self.s_conv(inputs)

        if self.mode == 'dual':
            l_conv, s_conv = tf.split(s_conv, num_or_size_splits=2, axis=-1)
            l_conv = tf.keras.layers.Activation('linear')(l_conv)
            s_conv = tf.keras.layers.Activation('sigmoid')(s_conv)
            outputs = l_conv * s_conv

        elif self.mode == 'residual':
            l_conv = self.l_conv(inputs)
            l_conv = tf.keras.layers.Activation('linear')(l_conv)
            s_conv = tf.keras.layers.Activation('sigmoid')(s_conv)
            g_conv = self.gamma * l_conv * s_conv

            if training and self.dropout:
                g_conv = tf.nn.dropout(g_conv, rate=self.dropout)

            outputs = g_conv + inputs

        else:
            s_conv = tf.keras.layers.Activation('sigmoid')(s_conv)
            outputs = inputs * s_conv

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

    DropAttention: A Regularization Method for Fully-Connected Self-Attention Networks
        https://arxiv.org/abs/1907.11065

    Self-Attention Generative Adversarial Networks
        https://arxiv.org/abs/1805.08318

    Spectral Normalization for GANs
        https://arxiv.org/abs/1802.05957
    """

    def __init__(self,
                 downrate=1,
                 pooling=False,
                 spectral=False,
                 kernel_initializer='glorot_uniform',
                 kernel_regularizer=None,
                 kernel_constraint=None,
                 gamma_initializer='zeros',
                 dropout=0.0,
                 **kwargs):
        """
        Initialize the self-attention gan layer.

        Parameters
        ----------
        downrate : int, optional
            Reduce the channels dimension by number factor.
        pooling : bool, optional
            Whether apply max pooling or not.
        spectral : bool, optional
            Whether apply spectral normalization or not.
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
        **kwargs : dict
            Additional keyword arguments for the layer.
        """

        super().__init__(**kwargs)

        self.downrate = downrate
        self.pooling = pooling
        self.spectral = spectral
        self.kernel_initializer = kernel_initializer
        self.kernel_regularizer = kernel_regularizer
        self.kernel_constraint = kernel_constraint
        self.gamma_initializer = gamma_initializer
        self.dropout = dropout

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
            'downrate': self.downrate,
            'pooling': self.pooling,
            'spectral': self.spectral,
            'kernel_initializer': self.kernel_initializer,
            'kernel_regularizer': self.kernel_regularizer,
            'kernel_constraint': self.kernel_constraint,
            'gamma_initializer': self.gamma_initializer,
            'dropout': self.dropout,
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
            raise ValueError("Unsupported input shape: must be 1D or 2D")

        self.filters = input_shape[-1]

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

        self.h_conv = conv_layer(filters=self.filters // self.downrate,
                                 kernel_size=1,
                                 padding='same',
                                 kernel_initializer=self.kernel_initializer,
                                 kernel_regularizer=self.kernel_regularizer,
                                 kernel_constraint=self.kernel_constraint,
                                 use_bias=False)

        if self.downrate > 1:
            self.o_conv = conv_layer(filters=self.filters,
                                     kernel_size=1,
                                     padding='same',
                                     kernel_initializer=self.kernel_initializer,
                                     kernel_regularizer=self.kernel_regularizer,
                                     kernel_constraint=self.kernel_constraint,
                                     use_bias=False)

            if self.spectral:
                self.o_conv = tf.keras.layers.SpectralNormalization(self.o_conv, name=self.o_conv.name)

        if self.spectral:
            self.f_conv = tf.keras.layers.SpectralNormalization(self.f_conv, name=self.f_conv.name)
            self.g_conv = tf.keras.layers.SpectralNormalization(self.g_conv, name=self.g_conv.name)
            self.h_conv = tf.keras.layers.SpectralNormalization(self.h_conv, name=self.h_conv.name)

        if self.pooling:
            self.f_pooling = pooling_layer(pool_size=pool_size, strides=strides)
            self.h_pooling = pooling_layer(pool_size=pool_size, strides=strides)

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
            Input tensor.
        training : bool, optional
            Whether in training mode.

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
        o = tf.reshape(o, shape=[shape[0]] + shape[1:-1] + [shape[-1] // self.downrate])

        if self.downrate > 1:
            o = self.o_conv(o)

        return self.gamma * o + inputs
