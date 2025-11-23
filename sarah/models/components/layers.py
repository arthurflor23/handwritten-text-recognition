import tensorflow as tf


class AdaptiveInstanceNormalization(tf.keras.layers.Layer):
    """
    Adaptive Instance Normalization Layer.
    Adjusts the mean and variance of the input tensor based on a conditional tensor.

    References
    ----------
    Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization
        https://arxiv.org/abs/1703.06868v2
    """

    def __init__(self, epsilon=1e-3, **kwargs):
        """
        Initializes the adaptive instance normalization layer.

        Parameters
        ----------
        epsilon : float, optional
            Small float added to variance to avoid dividing by zero.
        **kwargs : dict
            Additional arguments.
        """

        super().__init__(**kwargs)

        self.epsilon = epsilon

    def get_config(self):
        """
        Return the configuration of the layer.

        Returns
        -------
        dict
            Configuration dictionary.
        """

        config = super().get_config()

        config.update({
            'epsilon': self.epsilon,
        })

        return config

    def build(self, input_shape):
        """
        Initializes layer weights.

        Parameters
        ----------
        input_shape : tuple
            Shape of the input tensors.
        """

        super().build(input_shape)

        self.channels = input_shape[0][-1]

        self.beta_dense = tf.keras.layers.Dense(self.channels, use_bias=False)
        self.gamma_dense = tf.keras.layers.Dense(self.channels, use_bias=False)

        self.norm_layer = tf.keras.layers.GroupNormalization(groups=-1,
                                                             scale=False,
                                                             center=False,
                                                             epsilon=self.epsilon)

    def call(self, inputs, training=False):
        """
        Forward pass of the layer.

        Parameters
        ----------
        inputs : tf.Tensor
            A tuple containing the input tensor and the conditional tensor.
        training : bool, optional
            Whether the call is for training or inference.

        Returns
        -------
        tf.Tensor
            Tensor after applying AdaIN.
        """

        x, conditional = inputs

        beta = self.beta_dense(conditional)
        gamma = self.gamma_dense(conditional)

        beta = tf.reshape(beta, [-1, 1, 1, self.channels])
        gamma = tf.reshape(gamma, [-1, 1, 1, self.channels])

        normed = self.norm_layer(x, training=training)
        outputs = normed * gamma + beta

        return outputs


class ConditionalAttentionConv1D(tf.keras.layers.Layer):
    """
    1D Attention layer for capturing long-range dependencies in sequence data.

    References
    ----------
    Attention Is All You Need
        https://arxiv.org/abs/1706.03762

    DropAttention: A Regularization Method for Fully-Connected Self-Attention Networks
        https://arxiv.org/abs/1907.11065

    Self-Attention Generative Adversarial Networks
        https://arxiv.org/abs/1805.08318
    """

    def __init__(self,
                 k=8,
                 h=1.0,
                 kernel_initializer='glorot_uniform',
                 kernel_regularizer=None,
                 kernel_constraint=None,
                 beta_initializer='zeros',
                 dropout=0.0,
                 pooling=False,
                 use_bias=True,
                 **kwargs):
        """
        Initialize the attention layer.

        Parameters
        ----------
        k : int, optional
            Number of groups for input channels.
        h : int or float, optional
            Projection factor for value features.
        kernel_initializer : initializer, optional
            Kernel weights initializer.
        kernel_regularizer : regularizer, optional
            Kernel weights regularizer.
        kernel_constraint : constraint, optional
            Kernel weights constraint.
        beta_initializer : initializer, optional
            Beta weights initializer.
        dropout : float, optional
            Dropout rate to apply on attention weights.
        pooling : bool, optional
            Whether apply pooling reducing or not.
        use_bias : bool, optional
            Whether the layers use bias vectors/matrices.
        **kwargs : dict
            Additional keyword arguments.
        """

        super().__init__(**kwargs)

        self.k = k
        self.h = h
        self.kernel_initializer = kernel_initializer
        self.kernel_regularizer = kernel_regularizer
        self.kernel_constraint = kernel_constraint
        self.beta_initializer = beta_initializer
        self.dropout = dropout
        self.pooling = pooling
        self.use_bias = use_bias

    def get_config(self):
        """
        Returns the config of the layer.

        Returns
        -------
        dict
            Configuration dictionary.
        """

        config = super().get_config()

        config.update({
            'k': self.k,
            'h': self.h,
            'kernel_initializer': self.kernel_initializer,
            'kernel_regularizer': self.kernel_regularizer,
            'kernel_constraint': self.kernel_constraint,
            'beta_initializer': self.beta_initializer,
            'dropout': self.dropout,
            'pooling': self.pooling,
            'use_bias': self.use_bias,
        })

        return config

    def build(self, query_shape, value_shape=None, key_shape=None):
        """
        Build the layer structure based on the input shape.

        Parameters
        ----------
        query_shape : TensorShape
            Shape of the Query input tensor.
        value_shape : TensorShape
            Shape of the Value input tensor.
        key_shape : TensorShape
            Shape of the Key input tensor.
        """

        super().build(query_shape)

        if value_shape is None:
            value_shape = query_shape

        if key_shape is None:
            key_shape = value_shape

        if len(query_shape) == 3:
            pool_size = strides = 2 if query_shape[-2] > 1 else 1
        else:
            raise ValueError("Unsupported input shape: must be 1D")

        self.filters = query_shape[-1]

        if self.pooling:
            self.k_pooling = tf.keras.layers.MaxPooling1D(pool_size=pool_size, strides=strides)
            self.v_pooling = tf.keras.layers.MaxPooling1D(pool_size=pool_size, strides=strides)

        self.k_conv = tf.keras.layers.Conv1D(filters=self.filters // self.k,
                                             kernel_size=1,
                                             padding='valid',
                                             kernel_initializer=self.kernel_initializer,
                                             kernel_regularizer=self.kernel_regularizer,
                                             kernel_constraint=self.kernel_constraint,
                                             use_bias=self.use_bias)

        self.q_conv = tf.keras.layers.Conv1D(filters=self.filters // self.k,
                                             kernel_size=1,
                                             padding='valid',
                                             kernel_initializer=self.kernel_initializer,
                                             kernel_regularizer=self.kernel_regularizer,
                                             kernel_constraint=self.kernel_constraint,
                                             use_bias=self.use_bias)

        self.v_conv = tf.keras.layers.Conv1D(filters=self.features * self.h,
                                             kernel_size=1,
                                             padding='valid',
                                             kernel_initializer=self.kernel_initializer,
                                             kernel_regularizer=self.kernel_regularizer,
                                             kernel_constraint=self.kernel_constraint,
                                             use_bias=self.use_bias)

        if self.h != 1:
            self.o_conv = tf.keras.layers.Conv1D(filters=self.filters,
                                                 kernel_size=1,
                                                 padding='valid',
                                                 kernel_initializer=self.kernel_initializer,
                                                 kernel_regularizer=self.kernel_regularizer,
                                                 kernel_constraint=self.kernel_constraint,
                                                 use_bias=self.use_bias)

        self.beta = self.add_weight(name=f"{self.name}_beta",
                                    shape=(1,),
                                    initializer=self.beta_initializer,
                                    trainable=True)

        self.dropout_layer = tf.keras.layers.Dropout(rate=self.dropout)

    def call(self, query, value=None, key=None, training=False):
        """
        Processes the input tensors through the layer.

        Parameters
        ----------
        query : tf.Tensor
            Query tensor.
        value : tf.Tensor, optional
            Value tensor.
        key : tf.Tensor, optional
            Key tensor.
        training : bool, optional
            Whether the call is for training or inference.

        Returns
        -------
        tf.Tensor
            Output tensor after applying self-attention.
        """

        if value is None:
            value = query

        if key is None:
            key = value

        shape = tf.unstack(tf.shape(query))
        B, T = shape[0], shape[1]

        k = self.k_conv(key)

        if self.pooling:
            k = self.k_pooling(k)

        k = tf.reshape(k, shape=[B, -1, k.shape[-1]])

        q = self.q_conv(query)
        q = tf.reshape(q, shape=[B, -1, q.shape[-1]])

        s = tf.matmul(q, k, transpose_b=True)
        s = tf.nn.softmax(s, axis=-1)

        if training and self.dropout:
            s = self.dropout_layer(s)
            s = tf.keras.ops.divide_no_nan(s, tf.reduce_sum(s, axis=-1, keepdims=True))

        v = self.v_conv(value)

        if self.pooling:
            v = self.v_pooling(v)

        v = tf.reshape(v, shape=[B, -1, v.shape[-1]])

        o = tf.matmul(s, v)
        o = tf.reshape(o, shape=[B, T, self.features * self.h])

        if self.h != 1:
            o = self.o_conv(o)

        return query + o * self.beta


class ConditionalAttentionConv2D(tf.keras.layers.Layer):
    """
    2D Attention layer for capturing long-range dependencies in spatial data.

    References
    ----------
    Attention Is All You Need
        https://arxiv.org/abs/1706.03762

    DropAttention: A Regularization Method for Fully-Connected Self-Attention Networks
        https://arxiv.org/abs/1907.11065

    Self-Attention Generative Adversarial Networks
        https://arxiv.org/abs/1805.08318
    """

    def __init__(self,
                 k=8,
                 h=1.0,
                 kernel_initializer='glorot_uniform',
                 kernel_regularizer=None,
                 kernel_constraint=None,
                 beta_initializer='zeros',
                 dropout=0.0,
                 pooling=False,
                 use_bias=True,
                 **kwargs):
        """
        Initialize the attention layer.

        Parameters
        ----------
        k : int, optional
            Number of groups for input channels.
        h : int or float, optional
            Projection factor for value features.
        kernel_initializer : initializer, optional
            Kernel weights initializer.
        kernel_regularizer : regularizer, optional
            Kernel weights regularizer.
        kernel_constraint : constraint, optional
            Kernel weights constraint.
        beta_initializer : initializer, optional
            Beta weights initializer.
        dropout : float, optional
            Dropout rate to apply on attention weights.
        pooling : bool, optional
            Whether apply pooling reducing or not.
        use_bias : bool, optional
            Whether the layers use bias vectors/matrices.
        **kwargs : dict
            Additional keyword arguments.
        """

        super().__init__(**kwargs)

        self.k = k
        self.h = h
        self.kernel_initializer = kernel_initializer
        self.kernel_regularizer = kernel_regularizer
        self.kernel_constraint = kernel_constraint
        self.beta_initializer = beta_initializer
        self.dropout = dropout
        self.pooling = pooling
        self.use_bias = use_bias

    def get_config(self):
        """
        Returns the config of the layer.

        Returns
        -------
        dict
            Configuration dictionary.
        """

        config = super().get_config()

        config.update({
            'k': self.k,
            'h': self.h,
            'kernel_initializer': self.kernel_initializer,
            'kernel_regularizer': self.kernel_regularizer,
            'kernel_constraint': self.kernel_constraint,
            'beta_initializer': self.beta_initializer,
            'dropout': self.dropout,
            'pooling': self.pooling,
            'use_bias': self.use_bias,
        })

        return config

    def build(self, query_shape, value_shape=None, key_shape=None):
        """
        Build the layer structure based on the input shape.

        Parameters
        ----------
        query_shape : TensorShape
            Shape of the Query input tensor.
        value_shape : TensorShape
            Shape of the Value input tensor.
        key_shape : TensorShape
            Shape of the Key input tensor.
        """

        super().build(query_shape)

        if value_shape is None:
            value_shape = query_shape

        if key_shape is None:
            key_shape = value_shape

        if len(query_shape) == 4:
            pool_size = strides = (2 if query_shape[-3] > 1 else 1,
                                   2 if query_shape[-2] > 1 else 1)
        else:
            raise ValueError("Unsupported input shape: must be 2D")

        self.filters = query_shape[-1]

        if self.pooling:
            self.k_pooling = tf.keras.layers.MaxPooling2D(pool_size=pool_size, strides=strides)
            self.v_pooling = tf.keras.layers.MaxPooling2D(pool_size=pool_size, strides=strides)

        self.k_conv = tf.keras.layers.Conv2D(filters=self.filters // self.k,
                                             kernel_size=1,
                                             padding='valid',
                                             kernel_initializer=self.kernel_initializer,
                                             kernel_regularizer=self.kernel_regularizer,
                                             kernel_constraint=self.kernel_constraint,
                                             use_bias=self.use_bias)

        self.q_conv = tf.keras.layers.Conv2D(filters=self.filters // self.k,
                                             kernel_size=1,
                                             padding='valid',
                                             kernel_initializer=self.kernel_initializer,
                                             kernel_regularizer=self.kernel_regularizer,
                                             kernel_constraint=self.kernel_constraint,
                                             use_bias=self.use_bias)

        self.v_conv = tf.keras.layers.Conv2D(filters=self.features * self.h,
                                             kernel_size=1,
                                             padding='valid',
                                             kernel_initializer=self.kernel_initializer,
                                             kernel_regularizer=self.kernel_regularizer,
                                             kernel_constraint=self.kernel_constraint,
                                             use_bias=self.use_bias)

        if self.h != 1:
            self.o_conv = tf.keras.layers.Conv2D(filters=self.filters,
                                                 kernel_size=1,
                                                 padding='valid',
                                                 kernel_initializer=self.kernel_initializer,
                                                 kernel_regularizer=self.kernel_regularizer,
                                                 kernel_constraint=self.kernel_constraint,
                                                 use_bias=self.use_bias)

        self.beta = self.add_weight(name=f"{self.name}_beta",
                                    shape=(1,),
                                    initializer=self.beta_initializer,
                                    trainable=True)

        self.dropout_layer = tf.keras.layers.Dropout(rate=self.dropout)

    def call(self, query, value=None, key=None, training=False):
        """
        Processes the input tensors through the layer.

        Parameters
        ----------
        query : tf.Tensor
            Query tensor.
        value : tf.Tensor, optional
            Value tensor.
        key : tf.Tensor, optional
            Key tensor.
        training : bool, optional
            Whether the call is for training or inference.

        Returns
        -------
        tf.Tensor
            Output tensor after applying self-attention.
        """

        if value is None:
            value = query

        if key is None:
            key = value

        shape = tf.unstack(tf.shape(query))
        B, H, W = shape[0], shape[1], shape[2]

        k = self.k_conv(key)

        if self.pooling:
            k = self.k_pooling(k)

        k = tf.reshape(k, shape=[B, -1, k.shape[-1]])

        q = self.q_conv(query)
        q = tf.reshape(q, shape=[B, -1, q.shape[-1]])

        s = tf.matmul(q, k, transpose_b=True)
        s = tf.nn.softmax(s, axis=-1)

        if training and self.dropout:
            s = self.dropout_layer(s)
            s = tf.keras.ops.divide_no_nan(s, tf.reduce_sum(s, axis=-1, keepdims=True))

        v = self.v_conv(value)

        if self.pooling:
            v = self.v_pooling(v)

        v = tf.reshape(v, shape=[B, -1, v.shape[-1]])

        o = tf.matmul(s, v)
        o = tf.reshape(o, shape=[B, H, W, self.features * self.h])

        if self.h != 1:
            o = self.o_conv(o)

        return query + o * self.beta


class ConditionalAttentionDense(tf.keras.layers.Layer):
    """
    Fully-Connected Attention layer for modeling global dependencies in dense inputs.

    References
    ----------
    Attention Is All You Need
        https://arxiv.org/abs/1706.03762

    DropAttention: A Regularization Method for Fully-Connected Self-Attention Networks
        https://arxiv.org/abs/1907.11065

    Self-Attention Generative Adversarial Networks
        https://arxiv.org/abs/1805.08318
    """

    def __init__(self,
                 k=8,
                 h=1.0,
                 kernel_initializer='glorot_uniform',
                 kernel_regularizer=None,
                 kernel_constraint=None,
                 beta_initializer='zeros',
                 dropout=0.0,
                 pooling=False,
                 use_bias=True,
                 **kwargs):
        """
        Initialize the attention layer.

        Parameters
        ----------
        k : int, optional
            Number of groups for input channels.
        h : int or float, optional
            Projection factor for value features.
        kernel_initializer : initializer, optional
            Kernel weights initializer.
        kernel_regularizer : regularizer, optional
            Kernel weights regularizer.
        kernel_constraint : constraint, optional
            Kernel weights constraint.
        beta_initializer : initializer, optional
            Beta weights initializer.
        dropout : float, optional
            Dropout rate to apply on attention weights.
        pooling : bool, optional
            Whether apply pooling reducing or not.
        use_bias : bool, optional
            Whether the layers use bias vectors/matrices.
        **kwargs : dict
            Additional keyword arguments.
        """

        super().__init__(**kwargs)

        self.k = k
        self.h = h
        self.kernel_initializer = kernel_initializer
        self.kernel_regularizer = kernel_regularizer
        self.kernel_constraint = kernel_constraint
        self.beta_initializer = beta_initializer
        self.dropout = dropout
        self.pooling = pooling
        self.use_bias = use_bias

    def get_config(self):
        """
        Returns the config of the layer.

        Returns
        -------
        dict
            Configuration dictionary.
        """

        config = super().get_config()

        config.update({
            'k': self.k,
            'h': self.h,
            'kernel_initializer': self.kernel_initializer,
            'kernel_regularizer': self.kernel_regularizer,
            'kernel_constraint': self.kernel_constraint,
            'beta_initializer': self.beta_initializer,
            'dropout': self.dropout,
            'pooling': self.pooling,
            'use_bias': self.use_bias,
        })

        return config

    def build(self, query_shape, value_shape=None, key_shape=None):
        """
        Build the layer structure based on the input shape.

        Parameters
        ----------
        query_shape : TensorShape
            Shape of the Query input tensor.
        value_shape : TensorShape
            Shape of the Value input tensor.
        key_shape : TensorShape
            Shape of the Key input tensor.
        """

        super().build(query_shape)

        if value_shape is None:
            value_shape = query_shape

        if key_shape is None:
            key_shape = value_shape

        if len(query_shape) == 3:
            pooling_layer = tf.keras.layers.MaxPooling1D
            pool_size = strides = 2 if query_shape[-2] > 1 else 1

        elif len(query_shape) == 4:
            pooling_layer = tf.keras.layers.MaxPooling2D
            pool_size = strides = (2 if query_shape[-3] > 1 else 1,
                                   2 if query_shape[-2] > 1 else 1)
        else:
            raise ValueError("Unsupported input shape: must be 1D or 2D")

        self.features = query_shape[-1]

        if self.pooling:
            self.k_pooling = pooling_layer(pool_size=pool_size, strides=strides)
            self.v_pooling = pooling_layer(pool_size=pool_size, strides=strides)

        self.k_dense = tf.keras.layers.Dense(units=self.features // self.k,
                                             kernel_initializer=self.kernel_initializer,
                                             kernel_regularizer=self.kernel_regularizer,
                                             kernel_constraint=self.kernel_constraint,
                                             use_bias=self.use_bias)

        self.q_dense = tf.keras.layers.Dense(units=self.features // self.k,
                                             kernel_initializer=self.kernel_initializer,
                                             kernel_regularizer=self.kernel_regularizer,
                                             kernel_constraint=self.kernel_constraint,
                                             use_bias=self.use_bias)

        self.v_dense = tf.keras.layers.Dense(units=self.features * self.h,
                                             kernel_initializer=self.kernel_initializer,
                                             kernel_regularizer=self.kernel_regularizer,
                                             kernel_constraint=self.kernel_constraint,
                                             use_bias=self.use_bias)

        if self.h != 1:
            self.o_dense = tf.keras.layers.Dense(units=self.features,
                                                 kernel_initializer=self.kernel_initializer,
                                                 kernel_regularizer=self.kernel_regularizer,
                                                 kernel_constraint=self.kernel_constraint,
                                                 use_bias=self.use_bias)

        self.beta = self.add_weight(name=f"{self.name}_beta",
                                    shape=(1,),
                                    initializer=self.beta_initializer,
                                    trainable=True)

        self.dropout_layer = tf.keras.layers.Dropout(rate=self.dropout)

    def call(self, query, value=None, key=None, training=False):
        """
        Processes the input tensors through the layer.

        Parameters
        ----------
        query : tf.Tensor
            Query tensor.
        value : tf.Tensor, optional
            Value tensor.
        key : tf.Tensor, optional
            Key tensor.
        training : bool, optional
            Whether the call is for training or inference.

        Returns
        -------
        tf.Tensor
            Output tensor after applying self-attention.
        """

        if value is None:
            value = query

        if key is None:
            key = value

        shape = tf.unstack(tf.shape(query))

        k = self.k_dense(key)

        if self.pooling:
            k = self.k_pooling(k)

        k = tf.reshape(k, shape=(shape[0], -1, k.shape[-1]))

        q = self.q_dense(query)
        q = tf.reshape(q, shape=(shape[0], -1, q.shape[-1]))

        s = tf.matmul(q, k, transpose_b=True)
        s = tf.nn.softmax(s, axis=-1)

        if training and self.dropout:
            s = self.dropout_layer(s)
            s = tf.keras.ops.divide_no_nan(s, tf.reduce_sum(s, axis=-1, keepdims=True))

        v = self.v_dense(value)

        if self.pooling:
            v = self.v_pooling(v)

        v = tf.reshape(v, shape=(shape[0], -1, v.shape[-1]))

        o = tf.matmul(s, v)
        o = tf.reshape(o, shape=[shape[0]] + shape[1:-1] + [self.features * self.h])

        if self.h != 1:
            o = self.o_dense(o)

        return query + o * self.beta


class ConditionalBatchNormalization(tf.keras.layers.Layer):
    """
    Conditional Batch Normalization layer.
    Applies batch normalization conditioned on external data.

    References
    ----------
    Modulating early visual processing by language
        https://arxiv.org/abs/1707.00683v3
    """

    def __init__(self, momentum=0.99, epsilon=1e-3, **kwargs):
        """
        Initializes the conditional batch normalization layer.

        Parameters
        ----------
        momentum : float, optional
            Momentum for the moving average of mean and variance.
        epsilon : float, optional
            Small float added to variance to avoid dividing by zero.
        **kwargs : dict
            Additional arguments.
        """

        super().__init__(**kwargs)

        self.momentum = momentum
        self.epsilon = epsilon

    def get_config(self):
        """
        Return the configuration of the layer.

        Returns
        -------
        dict
            Configuration dictionary.
        """

        config = super().get_config()

        config.update({
            'momentum': self.momentum,
            'epsilon': self.epsilon,
        })

        return config

    def build(self, input_shape):
        """
        Initializes layer weights.

        Parameters
        ----------
        input_shape : tuple
            Shape of the input tensors.
        """

        super().build(input_shape)

        self.channels = input_shape[0][-1]

        self.beta_dense = tf.keras.layers.Dense(self.channels, use_bias=False)
        self.gamma_dense = tf.keras.layers.Dense(self.channels, use_bias=False)

        self.norm_layer = tf.keras.layers.BatchNormalization(momentum=self.momentum,
                                                             scale=False,
                                                             center=False,
                                                             epsilon=self.epsilon)

    def call(self, inputs, training=False):
        """
        Forward pass of the layer.

        Parameters
        ----------
        inputs : tf.Tensor
            A tuple containing the input tensor and the conditional tensor.
        training : bool, optional
            Whether the call is for training or inference.

        Returns
        -------
        tf.Tensor
            Tensor after applying CBN.
        """

        x, conditional = inputs

        beta = self.beta_dense(conditional)
        gamma = self.gamma_dense(conditional)

        beta = tf.reshape(beta, [-1, 1, 1, self.channels])
        gamma = tf.reshape(gamma, [-1, 1, 1, self.channels])

        normed = self.norm_layer(x, training=training)
        outputs = normed * gamma + beta

        return outputs


class ContentAlignment(tf.keras.layers.Layer):
    """
    Aligns the input content to match a target mask.
    """

    def __init__(self,
                 char_height_ratio,
                 char_width_ratio,
                 image_padding_value=-1,
                 text_padding_value=0,
                 mask_padding_value=0,
                 resize_method='bilinear',
                 **kwargs):
        """
        Initializes the layer.

        Parameters
        ----------
        char_height_ratio : int
            The height factor of the character in the image.
        char_width_ratio : int
            The width factor of the character in the image.
        image_padding_value : int, optional
            Padding value for image inputs.
        text_padding_value : int, optional
            Padding value for text inputs.
        mask_padding_value : int, optional
            Padding value for mask inputs.
        resize_method : str, optional
            Resize method name.
        **kwargs : dict
            Additional keyword arguments.
        """

        super().__init__(**kwargs)

        self.char_height_ratio = char_height_ratio
        self.char_width_ratio = char_width_ratio
        self.image_padding_value = image_padding_value
        self.text_padding_value = text_padding_value
        self.mask_padding_value = mask_padding_value
        self.resize_method = resize_method

    def get_config(self):
        """
        Return the configuration of the layer.

        Returns
        -------
        dict
            Configuration dictionary.
        """

        config = super().get_config()

        config.update({
            'char_height_ratio': self.char_height_ratio,
            'char_width_ratio': self.char_width_ratio,
            'image_padding_value': self.image_padding_value,
            'text_padding_value': self.text_padding_value,
            'mask_padding_value': self.mask_padding_value,
            'resize_method': self.resize_method,
        })

        return config

    def build(self, input_shape):
        """
        Initializes layer weights.

        Parameters
        ----------
        input_shape : tuple
            Shape of the input tensors.
        """

        super().build(input_shape)

        self.input_shape = input_shape[0]
        self.target_shape = input_shape[-1]

    @tf.function()
    def call(self, inputs):
        """
        Applies content alignment to the input image.

        Parameters
        ----------
        inputs : tuple of tf.Tensor
            A tuple containing the input tensor and the mask.

        Returns
        -------
        tf.Tensor
            Aligned image tensor.
        """

        input_data, input_text, input_mask = inputs

        text_height = self.get_content_length(input_data=input_text,
                                              pad_value=self.text_padding_value,
                                              scale_by=self.char_height_ratio,
                                              clip_by=self.input_shape[1],
                                              axis=1)

        text_width = self.get_content_length(input_data=input_text,
                                             pad_value=self.text_padding_value,
                                             scale_by=self.char_width_ratio,
                                             clip_by=self.input_shape[2],
                                             axis=2)

        mask_height = self.get_content_length(input_mask, self.mask_padding_value, axis=1)
        mask_width = self.get_content_length(input_mask, self.mask_padding_value, axis=2)

        def content_alignment(elems):
            img, text_h, text_w, mask_h, mask_w = elems
            image = tf.image.resize(img[:text_h, :text_w, :], size=(mask_h, mask_w), method=self.resize_method)

            if tf.shape(img)[1] > text_w and self.target_shape[2] > mask_w:
                size = [mask_h, self.target_shape[2] - mask_w]
                chunk = tf.image.resize(img[:text_h, text_w:, :], size=size, method=self.resize_method)
                image = tf.concat([image, chunk], axis=1)

            if self.target_shape[2] > tf.shape(image)[1]:
                size = [tf.shape(image)[0], self.target_shape[2] - tf.shape(image)[1], 1]
                chunk = tf.cast(tf.fill(size, value=self.image_padding_value), dtype=image.dtype)
                image = tf.concat([image, tf.stop_gradient(chunk)], axis=1)

            if tf.shape(img)[0] > text_h and self.target_shape[1] > mask_h:
                size = [self.target_shape[1] - mask_h, self.target_shape[2]]
                chunk = tf.image.resize(img[text_h:, :text_w, :], size=size, method=self.resize_method)
                image = tf.concat([image, chunk], axis=0)

            if self.target_shape[1] > tf.shape(image)[0]:
                size = [self.target_shape[1] - tf.shape(image)[0], tf.shape(image)[1], 1]
                chunk = tf.cast(tf.fill(size, value=self.image_padding_value), dtype=image.dtype)
                image = tf.concat([image, tf.stop_gradient(chunk)], axis=0)

            return image

        elems = (input_data, text_height, text_width, mask_height, mask_width)

        outputs = tf.map_fn(content_alignment, elems=elems, fn_output_signature=tf.float32)
        outputs = (outputs * input_mask) + tf.clip_by_value(input_mask - 1, self.image_padding_value, 0)

        return outputs

    def get_content_length(self, input_data, pad_value, scale_by=None, clip_by=None, axis=1):
        """
        Computes content length along an axis.

        Parameters
        ----------
        input_data : tf.Tensor
            Input data tensor.
        pad_value : float
            Padding value.
        scale_by : float, optional
            Factor to scale the length values.
        clip_by : float, optional
            Maximum value to clip the lengths.
        axis : int
            Axis to compute length.

        Returns
        -------
        tf.Tensor
            Content lengths.
        """

        input_rank = len(tf.shape(input_data))

        if axis not in [1, 2]:
            raise ValueError('Unsupported axis. Only axis=1 and axis=2 are supported.')
        elif input_rank <= 1 or input_rank >= 5:
            raise ValueError('Unsupported rank. Only rank<5 and rank>1 are supported.')

        if input_rank == 2:
            input_data = tf.expand_dims(input_data, axis=1)
        elif input_rank == 4:
            input_data = tf.squeeze(input_data, axis=-1)

        reduced = tf.reduce_sum(input_data, axis=(2 if axis == 1 else 1))
        content = tf.cast(tf.not_equal(reduced, pad_value), tf.int32)

        lengths = tf.reduce_sum(content, axis=1)
        lengths = tf.stop_gradient(lengths)

        if scale_by is not None:
            lengths = lengths * scale_by

        if clip_by is not None:
            lengths = tf.clip_by_value(lengths, 0, clip_by)

        return lengths

    def compute_output_shape(self, input_shape):
        """
        Compute the output shape of the layer.

        Parameters
        ----------
        input_shape : tuple or list
            The Shape of the input tensors.

        Returns
        -------
        tf.TensorShape
            The computed shape of the output from the layer.
        """

        if not self.built:
            self.build(input_shape)

        return tf.TensorShape(input_shape[-1])


class ExtractPatches(tf.keras.layers.Layer):
    """
    Layer to extract patches from input images.

    References
    ----------
    Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network
        https://arxiv.org/abs/1609.04802
    """

    def __init__(self, patch_shape=None, strides=(2, 2), padding='valid', **kwargs):
        """
        Initializes Patches layer.

        Parameters
        ----------
        patch_shape : list, tuple or None
            The target patch size to create.
        strides : list or tuple, optional
            Stride factors for the patches.
        padding : str, optional
            Padding method ('valid' or 'same').
        **kwargs : dict
            Additional keyword arguments.
        """

        super().__init__(**kwargs)

        self.patch_shape = patch_shape
        self.strides = (strides, strides) if isinstance(strides, int) else strides
        self.padding = padding

    def get_config(self):
        """
        Returns the config of the layer.

        Returns
        -------
        dict
            Configuration dictionary.
        """

        config = super().get_config()

        config.update({
            'patch_shape': self.patch_shape,
            'strides': self.strides,
            'padding': self.padding,
        })

        return config

    def call(self, inputs, training=False):
        """
        Splits the input image into patches.

        Parameters
        ----------
        inputs : tensor
            The input tensor representing images.
        training : bool, optional
            Whether the call is for training or inference.

        Returns
        -------
        tf.Tensor
            A tensor containing the extracted patches.
        """

        if not self.patch_shape:
            return inputs

        sizes = [1, self.patch_shape[0], self.patch_shape[1], 1]
        strides = [1, self.patch_shape[0] // self.strides[0], self.patch_shape[1] // self.strides[1], 1]

        patches = tf.image.extract_patches(images=inputs,
                                           sizes=sizes,
                                           strides=strides,
                                           rates=[1, 1, 1, 1],
                                           padding=self.padding.upper())

        patches = tf.reshape(patches, shape=[-1, self.patch_shape[0], self.patch_shape[1], 1])

        if training and self.trainable:
            patches = tf.stop_gradient(patches)

        return patches


class GatedConv2D(tf.keras.layers.Layer):
    """
    Implements a Gated Convolutional layer.

    References
    ----------
    Gated convolutional recurrent neural networks for multilingual handwriting recognition
        https://ieeexplore.ieee.org/document/8270042
    """

    def __init__(self,
                 kernel_size=(3, 3),
                 kernel_initializer='glorot_uniform',
                 kernel_regularizer=None,
                 kernel_constraint=None,
                 use_bias=True,
                 **kwargs):
        """
        Initializes the layer.

        Parameters
        ----------
        kernel_size : int or tuple, optional
            Convolution window size.
        kernel_initializer : initializer, optional
            Kernel weights initializer.
        kernel_regularizer : regularizer, optional
            Kernel weights regularizer.
        kernel_constraint : constraint, optional
            Kernel weights constraint.
        use_bias : bool, optional
            Whether the layers use bias vectors/matrices.
        **kwargs : dict
            Conv2D keyword arguments.
        """

        super().__init__(**kwargs)

        self.kernel_size = kernel_size
        self.kernel_initializer = kernel_initializer
        self.kernel_regularizer = kernel_regularizer
        self.kernel_constraint = kernel_constraint
        self.use_bias = use_bias

    def get_config(self):
        """
        Returns the config of the layer.

        Returns
        -------
        dict
            Configuration dictionary.
        """

        config = super().get_config()

        config.update({
            'kernel_size': self.kernel_size,
            'kernel_initializer': self.kernel_initializer,
            'kernel_regularizer': self.kernel_regularizer,
            'kernel_constraint': self.kernel_constraint,
            'use_bias': self.use_bias,
        })

        return config

    def build(self, input_shape):
        """
        Initializes layer weights.

        Parameters
        ----------
        input_shape : tuple
            Shape of the input tensors.
        """

        super().build(input_shape)

        self.s_conv = tf.keras.layers.Conv2D(filters=input_shape[-1],
                                             kernel_size=self.kernel_size,
                                             padding='same',
                                             kernel_initializer=self.kernel_initializer,
                                             kernel_regularizer=self.kernel_regularizer,
                                             kernel_constraint=self.kernel_constraint,
                                             use_bias=self.use_bias)

    def call(self, inputs):
        """
        Apply gated convolution to the input.

        Parameters
        ----------
        inputs : tf.Tensor
            Input tensor.

        Returns
        -------
        tf.Tensor
            Tensor resulting from the gated convolution.
        """

        s_conv = self.s_conv(inputs)
        s_conv = tf.nn.sigmoid(s_conv)

        return s_conv * inputs


class GatedDualConv2D(tf.keras.layers.Layer):
    """
    Implements a Dual Gated Convolutional layer.

    References
    ----------
    HTR-Flor: A Deep Learning System for Offline Handwritten Text Recognition
        https://ieeexplore.ieee.org/document/9266005

    Language modeling with gated convolutional networks
        https://arxiv.org/abs/1612.08083
    """

    def __init__(self,
                 kernel_size=(3, 3),
                 kernel_initializer='glorot_uniform',
                 kernel_regularizer=None,
                 kernel_constraint=None,
                 use_bias=True,
                 **kwargs):
        """
        Initializes the layer.

        Parameters
        ----------
        kernel_size : int or tuple, optional
            Convolution window size.
        kernel_initializer : initializer, optional
            Kernel weights initializer.
        kernel_regularizer : regularizer, optional
            Kernel weights regularizer.
        kernel_constraint : constraint, optional
            Kernel weights constraint.
        use_bias : bool, optional
            Whether the layers use bias vectors/matrices.
        **kwargs : dict
            Conv2D keyword arguments.
        """

        super().__init__(**kwargs)

        self.kernel_size = kernel_size
        self.kernel_initializer = kernel_initializer
        self.kernel_regularizer = kernel_regularizer
        self.kernel_constraint = kernel_constraint
        self.use_bias = use_bias

    def get_config(self):
        """
        Returns the config of the layer.

        Returns
        -------
        dict
            Configuration dictionary.
        """

        config = super().get_config()

        config.update({
            'kernel_size': self.kernel_size,
            'kernel_initializer': self.kernel_initializer,
            'kernel_regularizer': self.kernel_regularizer,
            'kernel_constraint': self.kernel_constraint,
            'use_bias': self.use_bias,
        })

        return config

    def build(self, input_shape):
        """
        Initializes layer weights.

        Parameters
        ----------
        input_shape : tuple
            Shape of the input tensors.
        """

        super().build(input_shape)

        self.sl_conv = tf.keras.layers.Conv2D(filters=input_shape[-1] * 2,
                                              kernel_size=self.kernel_size,
                                              padding='same',
                                              kernel_initializer=self.kernel_initializer,
                                              kernel_regularizer=self.kernel_regularizer,
                                              kernel_constraint=self.kernel_constraint,
                                              use_bias=self.use_bias)

    def call(self, inputs):
        """
        Apply dual gated convolution to the input.

        Parameters
        ----------
        inputs : tf.Tensor
            Input tensor.

        Returns
        -------
        tf.Tensor
            Tensor resulting from the dual gated convolution.
        """

        sl_conv = self.sl_conv(inputs)

        s_conv, l_conv = tf.split(sl_conv, num_or_size_splits=2, axis=-1)
        s_conv = tf.nn.sigmoid(s_conv)

        return s_conv * l_conv


class GatedResidualConv2D(tf.keras.layers.Layer):
    """
    Implements a Residual Gated Convolutional layer.
    """

    def __init__(self,
                 h=None,
                 kernel_size=(3, 3),
                 kernel_initializer='glorot_uniform',
                 kernel_regularizer=None,
                 kernel_constraint=None,
                 beta_initializer='zeros',
                 gamma_initializer='ones',
                 dropout=0.0,
                 use_bias=True,
                 **kwargs):
        """
        Initializes the layer.

        Parameters
        ----------
        h : int, optional
            Reduce the channels dimension to the value.
        kernel_size : int or tuple, optional
            Convolution window size.
        kernel_initializer : initializer, optional
            Kernel weights initializer.
        kernel_regularizer : regularizer, optional
            Kernel weights regularizer.
        kernel_constraint : constraint, optional
            Kernel weights constraint.
        beta_initializer : initializer, optional
            Beta weights initializer.
        gamma_initializer : initializer, optional
            Gamma weights initializer.
        dropout : float, optional
            Whether to apply dropout or not.
        use_bias : bool, optional
            Whether the layers use bias vectors/matrices.
        **kwargs : dict
            Conv2D keyword arguments.
        """

        super().__init__(**kwargs)

        self.h = h
        self.kernel_size = kernel_size
        self.kernel_initializer = kernel_initializer
        self.kernel_regularizer = kernel_regularizer
        self.kernel_constraint = kernel_constraint
        self.beta_initializer = beta_initializer
        self.gamma_initializer = gamma_initializer
        self.dropout = dropout
        self.use_bias = use_bias

    def get_config(self):
        """
        Returns the config of the layer.

        Returns
        -------
        dict
            Configuration dictionary.
        """

        config = super().get_config()

        config.update({
            'h': self.h,
            'kernel_size': self.kernel_size,
            'kernel_initializer': self.kernel_initializer,
            'kernel_regularizer': self.kernel_regularizer,
            'kernel_constraint': self.kernel_constraint,
            'beta_initializer': self.beta_initializer,
            'gamma_initializer': self.gamma_initializer,
            'dropout': self.dropout,
            'use_bias': self.use_bias,
        })

        return config

    def build(self, input_shape):
        """
        Initializes layer weights.

        Parameters
        ----------
        input_shape : tuple
            Shape of the input tensors.
        """

        super().build(input_shape)

        self.filters = input_shape[-1]
        self.h = self.h or self.filters

        self.s_conv = tf.keras.layers.Conv2D(filters=self.h,
                                             kernel_size=self.kernel_size,
                                             padding='same',
                                             kernel_initializer=self.kernel_initializer,
                                             kernel_regularizer=self.kernel_regularizer,
                                             kernel_constraint=self.kernel_constraint,
                                             use_bias=self.use_bias)

        if self.filters != self.h:
            self.o_conv = tf.keras.layers.Conv2D(filters=self.filters,
                                                 kernel_size=1,
                                                 padding='valid',
                                                 kernel_initializer=self.kernel_initializer,
                                                 kernel_regularizer=self.kernel_regularizer,
                                                 kernel_constraint=self.kernel_constraint,
                                                 use_bias=self.use_bias)

        self.beta = self.add_weight(name=f"{self.name}_beta",
                                    shape=(1,),
                                    initializer=self.beta_initializer,
                                    trainable=True)

        self.gamma = self.add_weight(name=f"{self.name}_gamma",
                                     shape=(1,),
                                     initializer=self.gamma_initializer,
                                     trainable=True)

        self.dropout_layer = tf.keras.layers.Dropout(rate=self.dropout)

    def call(self, inputs, training=False):
        """
        Apply residual gated convolution to the input.

        Parameters
        ----------
        inputs : tf.Tensor
            Input tensor.
        training : bool, optional
            Whether the call is for training or inference.

        Returns
        -------
        tf.Tensor
            Tensor resulting from the residual gated convolution.
        """

        s_conv = self.s_conv(inputs)
        g_conv = s_conv * tf.nn.sigmoid(s_conv * self.gamma)

        if training and self.dropout:
            g_conv = self.dropout_layer(g_conv)

        if self.filters != self.h:
            g_conv = self.o_conv(g_conv)

        return inputs + g_conv * self.beta


class OctaveConv2D(tf.keras.layers.Layer):
    """
    Implements octave convolutional layer.
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
        kernel_size : int or tuple, optional
            Convolution window size.
        strides : int or tuple, optional
            Stride of the convolution.
        padding : str, optional
            Padding mode: "valid" or "same".
        kernel_initializer : initializer, optional
            Kernel weights initializer.
        kernel_regularizer : regularizer, optional
            Kernel weights regularizer.
        kernel_constraint : constraint, optional
            Kernel weights constraint.
        **kwargs : dict
            Additional arguments.
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
            Configuration dictionary.
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
            Shape of the input tensors.
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
                                       padding=self.padding.upper())

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
            The Shape of the input tensors.

        Returns
        -------
        tf.TensorShape
            The computed shape of the output from the layer.
        """

        if not self.built:
            self.build(input_shape)

        output_shape = [(*input_shape[:3], self.high_channels),
                        (*input_shape[:3], self.low_channels)]

        return tf.TensorShape(output_shape)


class PositionEmbedding1D(tf.keras.layers.Layer):
    """
    1D Positional Embedding layer.
    Adds learnable positional embeddings to 1D input tensors.
    Typically used for sequence-like inputs: (batch_size, sequence_length, embedding_dim).

    References
    ----------
    BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding
        https://arxiv.org/abs/1810.04805
    """

    def __init__(self,
                 sequence_length,
                 embeddings_initializer='uniform',
                 embeddings_regularizer=None,
                 embeddings_constraint=None,
                 **kwargs):
        """
        Initializes the position embedding layer.

        Parameters
        ----------
        sequence_length : int
            Maximum length of the sequence.
        embeddings_initializer : str or initializer, optional
            Initializer for embeddings.
        embeddings_regularizer : str or regularizer, optional
            Regularizer for embeddings.
        embeddings_constraint : str or constraint, optional
            Constraint for embeddings.
        """
        super().__init__(**kwargs)

        self.sequence_length = sequence_length
        self.embeddings_initializer = tf.keras.initializers.get(embeddings_initializer)
        self.embeddings_regularizer = tf.keras.regularizers.get(embeddings_regularizer)
        self.embeddings_constraint = tf.keras.constraints.get(embeddings_constraint)

    def get_config(self):
        """
        Returns the config of the layer.

        Returns
        -------
        dict
            Configuration dictionary.
        """

        config = super().get_config()

        config.update({
            'sequence_length': self.sequence_length,
            'embeddings_initializer': tf.keras.initializers.serialize(self.embeddings_initializer),
            'embeddings_regularizer': tf.keras.regularizers.serialize(self.embeddings_regularizer),
            'embeddings_constraint': tf.keras.constraints.serialize(self.embeddings_constraint),
        })

        return config

    def build(self, input_shape):
        """
        Initializes layer weights.

        Parameters
        ----------
        input_shape : tuple
            Shape of the input tensors.
        """

        super().build(input_shape)

        self.pos_embeddings = self.add_weight(name=f"{self.name}_pos_embeddings",
                                              shape=(self.sequence_length, input_shape[-1]),
                                              initializer=self.embeddings_initializer,
                                              regularizer=self.embeddings_regularizer,
                                              constraint=self.embeddings_constraint,
                                              trainable=True)

    def call(self, inputs):
        """
        Adds 1D positional embeddings to the input.

        Parameters
        ----------
        inputs : tf.Tensor
            Tensor of shape (batch_size, sequence_length, embedding_dim)

        Returns
        -------
        tf.Tensor
            Tensor with 1D positional embeddings added.
        """

        input_shape = tf.shape(inputs)
        seq_len = input_shape[1]

        pos_emb = self.pos_embeddings[:seq_len, :]
        pos_emb = tf.expand_dims(pos_emb, axis=0)

        return inputs + pos_emb


class PositionEmbedding2D(tf.keras.layers.Layer):
    """
    2D Positional Embedding layer.
    Adds learnable row and column embeddings to 2D input tensors.
    Typically used for image-like inputs (batch_size, height, width, channels).

    References
    ----------
    BERT2D: Two Dimensional Positional Embeddings for Efficient Turkish NLP
        https://ieeexplore.ieee.org/document/10542953
    """

    def __init__(self,
                 height,
                 width,
                 embeddings_initializer='uniform',
                 embeddings_regularizer=None,
                 embeddings_constraint=None,
                 **kwargs):
        """
        Initializes the position embedding layer.

        Parameters
        ----------
        height : int
            Maximum height (number of rows).
        width : int
            Maximum width (number of columns).
        embeddings_initializer : str or initializer, optional
            Initializer for embeddings.
        embeddings_regularizer : str or regularizer, optional
            Regularizer for embeddings.
        embeddings_constraint : str or constraint, optional
            Constraint for embeddings.
        """

        super().__init__(**kwargs)

        self.height = height
        self.width = width
        self.embeddings_initializer = tf.keras.initializers.get(embeddings_initializer)
        self.embeddings_regularizer = tf.keras.regularizers.get(embeddings_regularizer)
        self.embeddings_constraint = tf.keras.constraints.get(embeddings_constraint)

    def get_config(self):
        """
        Returns the config of the layer.

        Returns
        -------
        dict
            Configuration dictionary.
        """

        config = super().get_config()

        config.update({
            'height': self.height,
            'width': self.width,
            'embeddings_initializer': tf.keras.initializers.serialize(self.embeddings_initializer),
            'embeddings_regularizer': tf.keras.regularizers.serialize(self.embeddings_regularizer),
            'embeddings_constraint': tf.keras.constraints.serialize(self.embeddings_constraint),
        })

        return config

    def build(self, input_shape):
        """
        Initializes layer weights.

        Parameters
        ----------
        input_shape : tuple
            Shape of the input tensors.
        """

        super().build(input_shape)

        self.row_embeddings = self.add_weight(name=f"{self.name}_row_embeddings",
                                              shape=(self.height, input_shape[-1]),
                                              initializer=self.embeddings_initializer,
                                              regularizer=self.embeddings_regularizer,
                                              constraint=self.embeddings_constraint,
                                              trainable=True)

        self.col_embeddings = self.add_weight(name=f"{self.name}_col_embeddings",
                                              shape=(self.width, input_shape[-1]),
                                              initializer=self.embeddings_initializer,
                                              regularizer=self.embeddings_regularizer,
                                              constraint=self.embeddings_constraint,
                                              trainable=True)

    def call(self, inputs):
        """
        Adds 2D positional embeddings to the input.

        Parameters
        ----------
        inputs : tf.Tensor
            Tensor of shape (batch_size, height, width, channels)

        Returns
        -------
        tf.Tensor
            Tensor with 2D positional embeddings added.
        """

        input_shape = tf.shape(inputs)
        h, w = input_shape[1], input_shape[2]

        row_emb = self.row_embeddings[:h, :]
        col_emb = self.col_embeddings[:w, :]

        row_emb = tf.reshape(row_emb, [1, h, 1, -1])
        col_emb = tf.reshape(col_emb, [1, 1, w, -1])

        pos_emb = row_emb + col_emb

        return inputs + pos_emb


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

        mu, logvar = inputs

        std = tf.exp(0.5 * logvar)
        eps = tf.random.normal(shape=tf.shape(mu))

        return eps * std + mu
