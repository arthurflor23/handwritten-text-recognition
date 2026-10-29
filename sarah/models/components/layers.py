import tensorflow as tf


class AdaptiveBatchNormalization(tf.keras.layers.Layer):
    """
    Adaptive Batch Normalization layer.
    Adjusts the normalized input features based on a conditional tensor.

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
            Additional keyword arguments.
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

        self.features = input_shape[0][-1]

        self.beta_dense = tf.keras.layers.Dense(units=self.features,
                                                kernel_initializer='zeros',
                                                use_bias=True)

        self.gamma_dense = tf.keras.layers.Dense(units=self.features,
                                                 kernel_initializer='zeros',
                                                 use_bias=True)

        self.norm_layer = tf.keras.layers.BatchNormalization(momentum=self.momentum,
                                                             scale=False,
                                                             center=False,
                                                             epsilon=self.epsilon)

    def call(self, inputs, training=False):
        """
        Forward pass of the layer.

        Parameters
        ----------
        inputs : tuple of tf.Tensor
            Input features and conditional tensor.
        training : bool, optional
            Whether the call is for training or inference.

        Returns
        -------
        tf.Tensor
            Adaptively normalized input tensor.
        """

        x, conditional = inputs

        beta = self.beta_dense(conditional)
        gamma = self.gamma_dense(conditional)

        beta = tf.reshape(beta, [-1, 1, 1, self.features])
        gamma = tf.reshape(gamma, [-1, 1, 1, self.features])

        normed = self.norm_layer(x, training=training)
        outputs = normed * gamma + beta

        return outputs


class AdaptiveInstanceNormalization(tf.keras.layers.Layer):
    """
    Adaptive Instance Normalization Layer.
    Adjusts the normalized input features based on a conditional tensor.

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

        self.features = input_shape[0][-1]

        self.beta_dense = tf.keras.layers.Dense(units=self.features,
                                                kernel_initializer='zeros',
                                                use_bias=True)

        self.gamma_dense = tf.keras.layers.Dense(units=self.features,
                                                 kernel_initializer='zeros',
                                                 use_bias=True)

        self.norm_layer = tf.keras.layers.GroupNormalization(groups=-1,
                                                             scale=False,
                                                             center=False,
                                                             epsilon=self.epsilon)

    def call(self, inputs, training=False):
        """
        Forward pass of the layer.

        Parameters
        ----------
        inputs : tuple of tf.Tensor
            Input features and conditional tensor.
        training : bool, optional
            Whether the call is for training or inference.

        Returns
        -------
        tf.Tensor
            Adaptively normalized input tensor.
        """

        x, conditional = inputs

        beta = self.beta_dense(conditional)
        gamma = self.gamma_dense(conditional)

        beta = tf.reshape(beta, [-1, 1, 1, self.features])
        gamma = tf.reshape(gamma, [-1, 1, 1, self.features])

        normed = self.norm_layer(x, training=training)
        outputs = normed * gamma + beta

        return outputs


class AdaptiveLayerNormalization(tf.keras.layers.Layer):
    """
    Adaptive Layer Normalization layer.
    Adjusts the normalized input features based on a conditional tensor.

    References
    ----------
    Scalable Diffusion Models with Transformers
        https://arxiv.org/abs/2212.09748
    """

    def __init__(self, epsilon=1e-3, **kwargs):
        """
        Initializes the adaptive layer normalization layer.

        Parameters
        ----------
        epsilon : float, optional
            Small float added to variance to avoid dividing by zero.
        **kwargs : dict
            Additional keyword arguments.
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

        self.features = input_shape[0][-1]
        self.broadcast_shape = [-1] + [1] * (len(input_shape[0]) - 2) + [self.features]

        self.beta_dense = tf.keras.layers.Dense(units=self.features,
                                                kernel_initializer='zeros',
                                                use_bias=True)

        self.gamma_dense = tf.keras.layers.Dense(units=self.features,
                                                 kernel_initializer='zeros',
                                                 use_bias=True)

        self.norm_layer = tf.keras.layers.LayerNormalization(axis=-1,
                                                             scale=False,
                                                             center=False,
                                                             epsilon=self.epsilon)

    def call(self, inputs, training=False):
        """
        Forward pass of the layer.

        Parameters
        ----------
        inputs : tuple of tf.Tensor
            Input features and conditional tensor.
        training : bool, optional
            Whether the call is for training or inference.

        Returns
        -------
        tf.Tensor
            Adaptively normalized input tensor.
        """

        x, conditional = inputs

        beta = self.beta_dense(conditional)
        gamma = self.gamma_dense(conditional)

        beta = tf.reshape(beta, self.broadcast_shape)
        gamma = tf.reshape(gamma, self.broadcast_shape)

        normed = self.norm_layer(x, training=training)
        outputs = normed * (1.0 + gamma) + beta

        return outputs


class ContentAlignment(tf.keras.layers.Layer):
    """
    Aligns the input content to match a target mask.
    """

    def __init__(self,
                 char_height_ratio,
                 char_width_ratio,
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

        mask_height = self.get_content_length(input_data=input_mask,
                                              pad_value=self.mask_padding_value,
                                              axis=1)

        mask_width = self.get_content_length(input_data=input_mask,
                                             pad_value=self.mask_padding_value,
                                             axis=2)

        def content_alignment(elems):
            img, text_h, text_w, mask_h, mask_w = elems
            image = tf.image.resize(img[:text_h, :text_w, :], size=(mask_h, mask_w), method=self.resize_method)

            if tf.shape(img)[1] > text_w and self.target_shape[2] > mask_w:
                size = [mask_h, self.target_shape[2] - mask_w]
                chunk = tf.image.resize(img[:text_h, text_w:, :], size=size, method=self.resize_method)
                image = tf.concat([image, chunk], axis=1)

            if self.target_shape[2] > tf.shape(image)[1]:
                size = [tf.shape(image)[0], self.target_shape[2] - tf.shape(image)[1], tf.shape(image)[-1]]
                chunk = tf.cast(tf.fill(size, value=-1), dtype=image.dtype)
                image = tf.concat([image, tf.stop_gradient(chunk)], axis=1)

            if tf.shape(img)[0] > text_h and self.target_shape[1] > mask_h:
                size = [self.target_shape[1] - mask_h, self.target_shape[2]]
                chunk = tf.image.resize(img[text_h:, :text_w, :], size=size, method=self.resize_method)
                image = tf.concat([image, chunk], axis=0)

            if self.target_shape[1] > tf.shape(image)[0]:
                size = [self.target_shape[1] - tf.shape(image)[0], tf.shape(image)[1], tf.shape(image)[-1]]
                chunk = tf.cast(tf.fill(size, value=-1), dtype=image.dtype)
                image = tf.concat([image, tf.stop_gradient(chunk)], axis=0)

            return image

        outputs = tf.map_fn(
            fn=content_alignment,
            elems=(tf.nn.tanh(input_data), text_height, text_width, mask_height, mask_width),
            fn_output_signature=tf.float32)

        outputs = (outputs * input_mask) + tf.clip_by_value(input_mask - 1, clip_value_min=-1, clip_value_max=0)

        return outputs

    def get_content_length(self,
                           input_data,
                           pad_value=0,
                           scale_by=None,
                           clip_by=None,
                           axis=1):
        """
        Computes content length along an axis.

        Parameters
        ----------
        input_data : tf.Tensor
            Input data tensor.
        pad_value : float, optional
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
        content = tf.cast(tf.not_equal(reduced, pad_value), dtype=tf.int32)

        lengths = tf.reduce_sum(content, axis=1)
        lengths = tf.stop_gradient(lengths)

        if scale_by is not None:
            lengths = lengths * scale_by

        if clip_by is not None:
            lengths = tf.clip_by_value(lengths, clip_value_min=0, clip_value_max=clip_by)

        return lengths

    def compute_output_shape(self, input_shape):
        """
        Computes the output shape of the layer.

        Parameters
        ----------
        input_shape : tuple
            Shape of the input tensors.

        Returns
        -------
        tf.TensorShape
            Output shape.
        """

        if not self.built:
            self.build(input_shape)

        input_shape, mask_shape = input_shape[0], input_shape[-1]
        output_shape = tf.TensorShape(mask_shape[:3] + input_shape[-1:])

        return output_shape


class GatedResidualConv1D(tf.keras.layers.Layer):
    """
    Implements a gated Conv1D residual branch.

    References
    ----------
    HTSR-Pollen: Handwritten Text Synthesis and Recognition System to Overcome Data Scarcity
        https://ieeexplore.ieee.org/document/11475723
    """

    def __init__(self,
                 filters=None,
                 kernel_size=3,
                 kernel_initializer='glorot_uniform',
                 kernel_regularizer=None,
                 kernel_constraint=None,
                 beta_initializer='zeros',
                 skip_projection=False,
                 dropout=0.0,
                 use_bias=True,
                 **kwargs):
        """
        Initializes the layer.

        Parameters
        ----------
        filters : int, optional
            Number of filters in the layer.
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
        skip_projection : bool, optional
            Whether to skip the projection.
        dropout : float, optional
            Dropout rate applied during training.
        use_bias : bool, optional
            Enables bias terms in the projection layer.
        **kwargs : dict
            Additional keyword arguments.
        """

        super().__init__(**kwargs)

        self.filters = filters
        self.kernel_size = kernel_size
        self.kernel_initializer = kernel_initializer
        self.kernel_regularizer = kernel_regularizer
        self.kernel_constraint = kernel_constraint
        self.beta_initializer = beta_initializer
        self.skip_projection = skip_projection
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
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'kernel_initializer': self.kernel_initializer,
            'kernel_regularizer': self.kernel_regularizer,
            'kernel_constraint': self.kernel_constraint,
            'beta_initializer': self.beta_initializer,
            'skip_projection': self.skip_projection,
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
            Shape of the input tensor.
        """

        super().build(input_shape)

        if not self.skip_projection:
            self.projection = tf.keras.layers.Conv1D(filters=self.filters or input_shape[-1],
                                                     kernel_size=self.kernel_size,
                                                     padding='same',
                                                     kernel_initializer=self.kernel_initializer,
                                                     kernel_regularizer=self.kernel_regularizer,
                                                     kernel_constraint=self.kernel_constraint,
                                                     use_bias=self.use_bias)

            self.gamma = self.add_weight(name=f'{self.name}_gamma',
                                         shape=(1,),
                                         initializer='ones',
                                         trainable=True)

        self.beta = self.add_weight(name=f'{self.name}_beta',
                                    shape=(1,),
                                    initializer=self.beta_initializer,
                                    trainable=True)

        if self.dropout:
            self.dropout_layer = tf.keras.layers.Dropout(rate=self.dropout)

    def call(self, inputs, training=False):
        """
        Forward pass of the layer.

        Parameters
        ----------
        inputs : tf.Tensor
            Input tensor.
        training : bool, optional
            Whether the call is for training or inference.

        Returns
        -------
        tf.Tensor
            Gated residual features.
        """

        if self.skip_projection:
            gated = tf.identity(inputs)
        else:
            projected = self.projection(inputs)
            gated = projected * tf.nn.sigmoid(projected * self.gamma)

        if self.dropout:
            gated = self.dropout_layer(gated, training=training)

        return gated * self.beta


class GatedResidualConv2D(tf.keras.layers.Layer):
    """
    Implements a gated Conv2D residual branch.

    References
    ----------
    HTSR-Pollen: Handwritten Text Synthesis and Recognition System to Overcome Data Scarcity
        https://ieeexplore.ieee.org/document/11475723
    """

    def __init__(self,
                 filters=None,
                 kernel_size=(3, 3),
                 kernel_initializer='glorot_uniform',
                 kernel_regularizer=None,
                 kernel_constraint=None,
                 beta_initializer='zeros',
                 skip_projection=False,
                 dropout=0.0,
                 use_bias=True,
                 **kwargs):
        """
        Initializes the layer.

        Parameters
        ----------
        filters : int, optional
            Number of filters in the layer.
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
        skip_projection : bool, optional
            Whether to skip the projection.
        dropout : float, optional
            Dropout rate applied during training.
        use_bias : bool, optional
            Enables bias terms in the projection layer.
        **kwargs : dict
            Additional keyword arguments.
        """

        super().__init__(**kwargs)

        self.filters = filters
        self.kernel_size = kernel_size
        self.kernel_initializer = kernel_initializer
        self.kernel_regularizer = kernel_regularizer
        self.kernel_constraint = kernel_constraint
        self.beta_initializer = beta_initializer
        self.skip_projection = skip_projection
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
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'kernel_initializer': self.kernel_initializer,
            'kernel_regularizer': self.kernel_regularizer,
            'kernel_constraint': self.kernel_constraint,
            'beta_initializer': self.beta_initializer,
            'skip_projection': self.skip_projection,
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
            Shape of the input tensor.
        """

        super().build(input_shape)

        if not self.skip_projection:
            self.projection = tf.keras.layers.Conv2D(filters=self.filters or input_shape[-1],
                                                     kernel_size=self.kernel_size,
                                                     padding='same',
                                                     kernel_initializer=self.kernel_initializer,
                                                     kernel_regularizer=self.kernel_regularizer,
                                                     kernel_constraint=self.kernel_constraint,
                                                     use_bias=self.use_bias)

            self.gamma = self.add_weight(name=f'{self.name}_gamma',
                                         shape=(1,),
                                         initializer='ones',
                                         trainable=True)

        self.beta = self.add_weight(name=f'{self.name}_beta',
                                    shape=(1,),
                                    initializer=self.beta_initializer,
                                    trainable=True)

        if self.dropout:
            self.dropout_layer = tf.keras.layers.Dropout(rate=self.dropout)

    def call(self, inputs, training=False):
        """
        Forward pass of the layer.

        Parameters
        ----------
        inputs : tf.Tensor
            Input tensor.
        training : bool, optional
            Whether the call is for training or inference.

        Returns
        -------
        tf.Tensor
            Gated residual features.
        """

        if self.skip_projection:
            gated = tf.identity(inputs)
        else:
            projected = self.projection(inputs)
            gated = projected * tf.nn.sigmoid(projected * self.gamma)

        if self.dropout:
            gated = self.dropout_layer(gated, training=training)

        return gated * self.beta


class GatedResidualDense(tf.keras.layers.Layer):
    """
    Implements a gated Dense residual branch.

    References
    ----------
    HTSR-Pollen: Handwritten Text Synthesis and Recognition System to Overcome Data Scarcity
        https://ieeexplore.ieee.org/document/11475723
    """

    def __init__(self,
                 units=None,
                 kernel_initializer='glorot_uniform',
                 kernel_regularizer=None,
                 kernel_constraint=None,
                 beta_initializer='zeros',
                 skip_projection=False,
                 dropout=0.0,
                 use_bias=True,
                 **kwargs):
        """
        Initializes the layer.

        Parameters
        ----------
        units : int, optional
            Number of units in the layer.
        kernel_initializer : initializer, optional
            Kernel weights initializer.
        kernel_regularizer : regularizer, optional
            Kernel weights regularizer.
        kernel_constraint : constraint, optional
            Kernel weights constraint.
        beta_initializer : initializer, optional
            Beta weights initializer.
        skip_projection : bool, optional
            Whether to skip the projection.
        dropout : float, optional
            Dropout rate applied during training.
        use_bias : bool, optional
            Enables bias terms in the projection layer.
        **kwargs : dict
            Additional keyword arguments.
        """

        super().__init__(**kwargs)

        self.units = units
        self.kernel_initializer = kernel_initializer
        self.kernel_regularizer = kernel_regularizer
        self.kernel_constraint = kernel_constraint
        self.beta_initializer = beta_initializer
        self.skip_projection = skip_projection
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
            'units': self.units,
            'kernel_initializer': self.kernel_initializer,
            'kernel_regularizer': self.kernel_regularizer,
            'kernel_constraint': self.kernel_constraint,
            'beta_initializer': self.beta_initializer,
            'skip_projection': self.skip_projection,
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
            Shape of the input tensor.
        """

        super().build(input_shape)

        if not self.skip_projection:
            self.projection = tf.keras.layers.Dense(units=self.units or input_shape[-1],
                                                    kernel_initializer=self.kernel_initializer,
                                                    kernel_regularizer=self.kernel_regularizer,
                                                    kernel_constraint=self.kernel_constraint,
                                                    use_bias=self.use_bias)

            self.gamma = self.add_weight(name=f'{self.name}_gamma',
                                         shape=(self.units or input_shape[-1], 1),
                                         initializer='ones',
                                         trainable=True)

        self.beta = self.add_weight(name=f'{self.name}_beta',
                                    shape=(1,),
                                    initializer=self.beta_initializer,
                                    trainable=True)

        if self.dropout:
            self.dropout_layer = tf.keras.layers.Dropout(rate=self.dropout)

    def call(self, inputs, training=False):
        """
        Forward pass of the layer.

        Parameters
        ----------
        inputs : tf.Tensor
            Input tensor.
        training : bool, optional
            Whether the call is for training or inference.

        Returns
        -------
        tf.Tensor
            Gated residual features.
        """

        if self.skip_projection:
            gated = tf.identity(inputs)
        else:
            projected = self.projection(inputs)
            gated = projected * tf.nn.sigmoid(projected @ self.gamma)

        if self.dropout:
            gated = self.dropout_layer(gated, training=training)

        return gated * self.beta


class GraphConv(tf.keras.layers.Layer):
    """
    Graph convolution layer over node and edge data.

    References
    ----------
    Semi-Supervised Classification with Graph Convolutional Networks
        https://arxiv.org/abs/1609.02907
    """

    def __init__(self,
                 units,
                 kernel_initializer='glorot_uniform',
                 kernel_regularizer=None,
                 kernel_constraint=None,
                 use_bias=True,
                 **kwargs):
        """
        Initializes the layer.

        Parameters
        ----------
        units : int
            Number of output features per node.
        kernel_initializer : initializer, optional
            Kernel weights initializer.
        kernel_regularizer : regularizer, optional
            Kernel weights regularizer.
        kernel_constraint : constraint, optional
            Kernel weights constraint.
        use_bias : bool, optional
            Whether to add bias terms.
        **kwargs : dict
            Additional keyword arguments.
        """

        super().__init__(**kwargs)

        self.units = units
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
            'units': self.units,
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

        self.kernel = tf.keras.layers.Dense(units=self.units,
                                            kernel_initializer=self.kernel_initializer,
                                            kernel_regularizer=self.kernel_regularizer,
                                            kernel_constraint=self.kernel_constraint,
                                            use_bias=self.use_bias)

    def call(self, inputs, masks=None):
        """
        Forward pass of the layer.

        Parameters
        ----------
        inputs : tuple or list of tf.Tensor
            Tuple of node features and edge indices.
        masks : tuple or list of tf.Tensor, optional
            Node and edge validity masks.

        Returns
        -------
        tf.Tensor
            Updated node features.
        """

        nodes, edges = inputs
        nodes_mask, edges_mask = masks if masks is not None else (None, None)

        if nodes_mask is None:
            nodes_mask = tf.ones(tf.shape(nodes)[:2], dtype=tf.bool)

        if edges_mask is None:
            edges_mask = tf.ones(tf.shape(edges[:, 0, :]), dtype=tf.bool)

        nodes = tf.where(nodes_mask[..., tf.newaxis], nodes, tf.zeros_like(nodes))

        batch_size = tf.shape(nodes)[0]
        num_nodes = tf.shape(nodes)[1]

        src = tf.cast(edges[:, 0, :], tf.int32)
        dst = tf.cast(edges[:, 1, :], tf.int32)

        src = tf.where(edges_mask, src, 0)
        dst = tf.where(edges_mask, dst, 0)

        edges_mask &= tf.gather(nodes_mask, src, batch_dims=1)
        edges_mask &= tf.gather(nodes_mask, dst, batch_dims=1)

        self_idx = tf.range(num_nodes, dtype=tf.int32)[tf.newaxis, :]
        self_idx = tf.tile(self_idx, [batch_size, 1])

        src = tf.concat([src, self_idx], axis=1)
        dst = tf.concat([dst, self_idx], axis=1)

        edges_mask = tf.concat([edges_mask, nodes_mask], axis=1)

        batch_offset = tf.range(batch_size, dtype=tf.int32)[:, tf.newaxis] * num_nodes
        dst_flat = tf.reshape(dst + batch_offset, [-1])
        total_segments = batch_size * num_nodes

        ones_flat = tf.reshape(tf.cast(edges_mask, nodes.dtype), [-1])
        degrees_flat = tf.math.unsorted_segment_sum(ones_flat, dst_flat, total_segments)

        degrees = tf.reshape(degrees_flat, [batch_size, num_nodes])
        deg_inv_sqrt = tf.math.rsqrt(tf.maximum(degrees, 1.0))

        src_norm = tf.gather(deg_inv_sqrt, src, batch_dims=1)
        dst_norm = tf.gather(deg_inv_sqrt, dst, batch_dims=1)

        norm = src_norm * dst_norm * tf.cast(edges_mask, nodes.dtype)
        messages = tf.gather(nodes, src, batch_dims=1) * norm[..., tf.newaxis]

        messages_flat = tf.reshape(messages, [-1, tf.shape(nodes)[-1]])
        aggregated_flat = tf.math.unsorted_segment_sum(messages_flat, dst_flat, total_segments)
        aggregated = tf.reshape(aggregated_flat, [batch_size, num_nodes, tf.shape(nodes)[-1]])

        outputs = self.kernel(aggregated)
        outputs = tf.where(nodes_mask[..., tf.newaxis], outputs, tf.zeros_like(outputs))

        return outputs


class GraphConv2D(tf.keras.layers.Layer):
    """
    Graph convolution over spatial features.

    References
    ----------
    Image Segmentation: Inducing graph-based learning
        https://arxiv.org/abs/2501.03765
    """

    def __init__(self,
                 filters,
                 kernel_size,
                 dilation=1,
                 kernel_initializer='glorot_uniform',
                 kernel_regularizer=None,
                 kernel_constraint=None,
                 use_bias=True,
                 **kwargs):
        """
        Initializes the layer.

        Parameters
        ----------
        filters : int
            Number of output features per node.
        kernel_size : int
            Number of graph neighbors.
        dilation : int, optional
            Step between neighbors in distance order.
        kernel_initializer : initializer, optional
            Kernel weights initializer.
        kernel_regularizer : regularizer, optional
            Kernel weights regularizer.
        kernel_constraint : constraint, optional
            Kernel weights constraint.
        use_bias : bool, optional
            Whether to add bias terms.
        **kwargs : dict
            Additional keyword arguments.
        """

        super().__init__(**kwargs)

        self.filters = filters
        self.kernel_size = kernel_size
        self.dilation = dilation
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
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'dilation': self.dilation,
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

        self.kernel = tf.keras.layers.Conv2D(filters=self.filters,
                                             kernel_size=1,
                                             kernel_initializer=self.kernel_initializer,
                                             kernel_regularizer=self.kernel_regularizer,
                                             kernel_constraint=self.kernel_constraint,
                                             use_bias=self.use_bias)

    def call(self, inputs):
        """
        Forward pass of the layer.

        Parameters
        ----------
        inputs : tf.Tensor
            Input feature map.

        Returns
        -------
        tf.Tensor
            Output feature map.
        """

        batch_size, height, width, channels = tf.unstack(tf.shape(inputs))
        num_nodes = height * width

        nodes = tf.reshape(inputs, [batch_size, num_nodes, channels])

        rows = tf.range(height, dtype=tf.float32)
        cols = tf.range(width, dtype=tf.float32)
        row_grid, col_grid = tf.meshgrid(rows, cols, indexing='ij')

        coords = tf.stack([row_grid, col_grid], axis=-1)
        offsets = tf.stack([tf.sin(row_grid), tf.cos(col_grid)], axis=-1)

        coordinates = tf.reshape(coords + offsets, [-1, 2])

        distances = coordinates[:, tf.newaxis, :] - coordinates[tf.newaxis, :, :]
        distances = tf.reduce_sum(tf.square(distances), axis=-1)

        neighbors = tf.math.top_k(-distances, k=self.kernel_size * self.dilation).indices
        neighbors = neighbors[:, ::self.dilation]

        aggregated = tf.reduce_sum(tf.gather(nodes, neighbors, axis=1), axis=2)
        aggregated = tf.reshape(aggregated, [batch_size, height, width, channels])
        aggregated.set_shape(inputs.shape)

        return self.kernel(aggregated)


class GraphExtraction(tf.keras.layers.Layer):
    """
    Extracts graphs from skeletonized handwriting using spatial block sampling.

    References
    ----------
    A fast parallel algorithm for thinning digital patterns
        https://doi.org/10.1145/357994.358023

    Combining graph edit distance and triplet networks for offline signature verification
        https://www.sciencedirect.com/science/article/pii/S0167865519301850

    Towards Visuo-Structural Handwriting Evaluation Based on Graph Matching
        https://link.springer.com/chapter/10.1007/978-3-031-45461-5_6
    """

    def __init__(self, block_size=1, relative=True, normalize=True, **kwargs):
        """
        Initializes the layer.

        Parameters
        ----------
        block_size : int, optional
            Skeleton sampling block size in pixels.
        relative : bool, optional
            Use relative node coordinates.
        normalize : bool, optional
            Normalize node coordinates.
        **kwargs : dict
            Additional keyword arguments.
        """

        super().__init__(**kwargs)

        self.block_size = block_size
        self.relative = relative
        self.normalize = normalize

        self._offsets = tf.range(self.block_size, dtype=tf.int32)

        self._lut0 = tf.constant(self._zhang_suen_lut(iteration=0), dtype=tf.bool)
        self._lut1 = tf.constant(self._zhang_suen_lut(iteration=1), dtype=tf.bool)

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
            'block_size': self.block_size,
            'relative': self.relative,
            'normalize': self.normalize,
        })

        return config

    def _zhang_suen_lut(self, iteration):
        """
        Builds one Zhang-Suen thinning lookup table.

        Parameters
        ----------
        iteration : int
            Sub-iteration selector.

        Returns
        -------
        list
            Boolean thinning lookup table.
        """

        lut = []

        for code in range(256):
            p9, p2, p3, p4, p5, p6, p7, p8 = [(code >> k) & 1 for k in range(8)]
            seq = [p2, p3, p4, p5, p6, p7, p8, p9, p2]

            b = sum(seq[:8])
            a = sum(seq[i] == 0 and seq[i + 1] == 1 for i in range(8))

            if iteration == 0:
                m1 = p2 * p4 * p6
                m2 = p4 * p6 * p8
            else:
                m1 = p2 * p4 * p8
                m2 = p2 * p6 * p8

            delete = 2 <= b <= 6 and a == 1 and m1 == 0 and m2 == 0
            lut.append(not delete)

        return lut

    def _thinning(self, ink):
        """
        Thins binary images to 1-pixel skeletons.

        Parameters
        ----------
        ink : tf.Tensor
            Batch of binary images.

        Returns
        -------
        tf.Tensor
            Binary skeleton masks.
        """

        def sub_iteration(img, lut):
            padded = tf.pad(img, [[0, 0], [1, 1], [1, 1]], constant_values=False)
            padded = tf.cast(padded, dtype=tf.int32)

            p9 = padded[:, :-2, :-2]
            p2 = padded[:, :-2, 1:-1]
            p3 = padded[:, :-2, 2:]
            p4 = padded[:, 1:-1, 2:]
            p5 = padded[:, 2:, 2:]
            p6 = padded[:, 2:, 1:-1]
            p7 = padded[:, 2:, :-2]
            p8 = padded[:, 1:-1, :-2]

            code = p9 + 2 * p2 + 4 * p3 + 8 * p4 + 16 * p5 + 32 * p6 + 64 * p7 + 128 * p8
            keep = tf.gather(lut, code)

            return img & (keep | border)

        def cond(img, changed):
            del img
            return changed

        def body(img, changed):
            del changed
            next_img = sub_iteration(img=img, lut=self._lut0)
            next_img = sub_iteration(img=next_img, lut=self._lut1)
            changed = tf.reduce_any(tf.not_equal(next_img, img))
            return next_img, changed

        image = tf.cast(ink, tf.bool)

        height = tf.shape(image)[1]
        width = tf.shape(image)[2]

        rows = tf.range(height, dtype=tf.int32)[:, tf.newaxis]
        cols = tf.range(width, dtype=tf.int32)[tf.newaxis, :]

        border = (
            (rows == 0)
            | (rows == height - 1)
            | (cols == 0)
            | (cols == width - 1)
        )[tf.newaxis, :, :]

        image, _ = tf.while_loop(cond=cond,
                                 body=body,
                                 loop_vars=(image, tf.constant(True)),
                                 maximum_iterations=100)

        return image

    def _cell_statistics(self, skeleton):
        """
        Computes occupancy and centroid statistics for spatial cells.

        Parameters
        ----------
        skeleton : tf.Tensor
            Batch of binary skeleton masks.

        Returns
        -------
        blocks : tf.Tensor
            Skeleton pixels grouped into spatial cells.
        counts : tf.Tensor
            Number of skeleton pixels in each cell.
        sum_x : tf.Tensor
            Sum of x coordinates in each cell.
        sum_y : tf.Tensor
            Sum of y coordinates in each cell.
        grid_h : tf.Tensor
            Number of cell rows.
        grid_w : tf.Tensor
            Number of cell columns.
        """

        batch_size = tf.shape(skeleton)[0]
        height = tf.shape(skeleton)[1]
        width = tf.shape(skeleton)[2]

        grid_h = tf.math.floordiv(height + self.block_size - 1, self.block_size)
        grid_w = tf.math.floordiv(width + self.block_size - 1, self.block_size)

        pad_h = grid_h * self.block_size - height
        pad_w = grid_w * self.block_size - width

        paddings = tf.stack([
            tf.constant([0, 0], dtype=tf.int32),
            tf.stack([tf.constant(0, tf.int32), pad_h]),
            tf.stack([tf.constant(0, tf.int32), pad_w]),
        ])

        padded = tf.pad(skeleton, paddings, constant_values=False)

        blocks = tf.reshape(padded, tf.stack([batch_size, grid_h, self.block_size, grid_w, self.block_size]))
        blocks_int = tf.cast(blocks, tf.int32)

        col_counts = tf.reduce_sum(blocks_int, axis=2)
        row_counts = tf.reduce_sum(blocks_int, axis=4)
        counts = tf.reduce_sum(col_counts, axis=3)

        local_x_sum = tf.reduce_sum(col_counts * self._offsets[tf.newaxis, tf.newaxis, tf.newaxis, :], axis=3)
        local_y_sum = tf.reduce_sum(row_counts * self._offsets[tf.newaxis, tf.newaxis, :, tf.newaxis], axis=2)

        grid_x = tf.range(grid_w, dtype=tf.int32)[tf.newaxis, tf.newaxis, :]
        grid_y = tf.range(grid_h, dtype=tf.int32)[tf.newaxis, :, tf.newaxis]

        sum_x = local_x_sum + counts * (grid_x * self.block_size)
        sum_y = local_y_sum + counts * (grid_y * self.block_size)

        return blocks, counts, sum_x, sum_y, grid_h, grid_w

    def _edge_slots(self, blocks):
        """
        Computes connectivity between neighboring spatial cells.

        Parameters
        ----------
        blocks : tf.Tensor
            Skeleton pixels grouped into spatial cells.

        Returns
        -------
        tf.Tensor
            East, south, southeast, and southwest connectivity.
        """

        left = blocks[:, :, :, :-1, -1]
        right = blocks[:, :, :, 1:, 0]

        east = tf.reduce_any(left & right, axis=2)
        east = east | tf.reduce_any(left[:, :, :-1, :] & right[:, :, 1:, :], axis=2)
        east = east | tf.reduce_any(left[:, :, 1:, :] & right[:, :, :-1, :], axis=2)

        top = blocks[:, :-1, -1, :, :]
        bottom = blocks[:, 1:, 0, :, :]

        south = tf.reduce_any(top & bottom, axis=3)
        south = south | tf.reduce_any(top[..., :-1] & bottom[..., 1:], axis=3)
        south = south | tf.reduce_any(top[..., 1:] & bottom[..., :-1], axis=3)

        southeast = blocks[:, :-1, -1, :-1, -1] & blocks[:, 1:, 0, 1:, 0]
        southwest = blocks[:, :-1, -1, 1:, 0] & blocks[:, 1:, 0, :-1, -1]

        east = tf.pad(east, [[0, 0], [0, 0], [0, 1]])
        south = tf.pad(south, [[0, 0], [0, 1], [0, 0]])
        southeast = tf.pad(southeast, [[0, 0], [0, 1], [0, 1]])
        southwest = tf.pad(southwest, [[0, 0], [0, 1], [1, 0]])

        return tf.stack([east, south, southeast, southwest], axis=-1)

    def _skeleton_to_graphs(self, skeleton):
        """
        Converts skeleton masks into padded graph tensors.

        Parameters
        ----------
        skeleton : tf.Tensor
            Batch of binary skeleton masks.

        Returns
        -------
        coordinates : tf.Tensor
            Original pixel coordinates.
        nodes : tf.Tensor
            Node features with configurable positions.
        edges : tf.Tensor
            Edge indices.
        nodes_mask : tf.Tensor
            Boolean validity mask for nodes.
        edges_mask : tf.Tensor
            Boolean validity mask for edges.
        """

        batch_size = tf.shape(skeleton)[0]

        blocks, counts, sum_x, sum_y, grid_h, grid_w = self._cell_statistics(skeleton=skeleton)
        cells_per_image = grid_h * grid_w

        counts = tf.reshape(tf.transpose(counts, [0, 2, 1]), tf.stack([batch_size, -1]))
        sum_x = tf.reshape(tf.transpose(sum_x, [0, 2, 1]), tf.stack([batch_size, -1]))
        sum_y = tf.reshape(tf.transpose(sum_y, [0, 2, 1]), tf.stack([batch_size, -1]))

        occupied = counts > 0
        occupied_int = tf.cast(occupied, tf.int32)

        cell_node = tf.cumsum(occupied_int, axis=1, exclusive=True)

        node_counts = tf.reduce_sum(occupied_int, axis=1)
        max_nodes = tf.reduce_max(node_counts)

        occupied_indices = tf.cast(tf.where(occupied), tf.int32)

        node_batch = occupied_indices[:, 0]
        node_local = tf.gather_nd(cell_node, occupied_indices)

        occupied_count = tf.cast(tf.gather_nd(counts, occupied_indices), tf.float32)

        x = tf.cast(tf.gather_nd(sum_x, occupied_indices), tf.float32) / occupied_count
        y = tf.cast(tf.gather_nd(sum_y, occupied_indices), tf.float32) / occupied_count

        xy = tf.stack([x, y], axis=1)
        node_xy = xy

        if self.relative:
            xy_min = tf.math.unsorted_segment_min(xy, node_batch, batch_size)
            node_xy = node_xy - tf.gather(xy_min, node_batch)

        if self.normalize:
            if self.relative:
                xy_max = tf.math.unsorted_segment_max(xy, node_batch, batch_size)
                graph_scale = tf.maximum(tf.reduce_max(xy_max - xy_min, axis=1), 1.0)
                scale = tf.gather(graph_scale, node_batch)
                node_xy = node_xy / scale[:, tf.newaxis]
            else:
                image_scale = tf.cast(tf.reduce_max(tf.shape(skeleton)[1:3]), tf.float32)
                node_xy = node_xy / tf.maximum(image_scale, 1.0)

        node_values = tf.pad(node_xy, [[0, 0], [0, 2]])
        node_indices = tf.stack([node_batch, node_local], axis=1)

        coordinates = tf.scatter_nd(indices=node_indices,
                                    updates=tf.cast(tf.round(xy), tf.int32),
                                    shape=tf.stack([batch_size, max_nodes, 2]))

        nodes = tf.scatter_nd(indices=node_indices,
                              updates=node_values,
                              shape=tf.stack([batch_size, max_nodes, 4]))

        edge_slots = self._edge_slots(blocks=blocks)
        edge_slots = tf.transpose(edge_slots, [0, 2, 1, 3])

        forward_counts = tf.reduce_sum(tf.cast(edge_slots, tf.int32), axis=(1, 2, 3))
        active_slots = tf.cast(tf.where(tf.reshape(edge_slots, [-1]))[:, 0], tf.int32)

        source_cell = active_slots // 4
        direction = active_slots % 4
        source_batch = source_cell // cells_per_image

        target_delta = tf.gather(tf.stack([grid_h, 1, grid_h + 1, 1 - grid_h]), direction)
        target_cell = source_cell + target_delta

        cell_node_flat = tf.reshape(cell_node, [-1])

        source_local = tf.gather(cell_node_flat, source_cell)
        target_local = tf.gather(cell_node_flat, target_cell)

        forward = tf.stack([source_local, target_local], axis=1)
        reverse = tf.stack([target_local, source_local], axis=1)

        edge_values = tf.reshape(tf.stack([forward, reverse], axis=1), [-1, 2])

        forward_offsets = tf.cumsum(forward_counts, exclusive=True)
        forward_local = tf.range(tf.shape(active_slots)[0], dtype=tf.int32) - tf.gather(forward_offsets, source_batch)

        edge_batch = tf.reshape(tf.stack([source_batch, source_batch], axis=1), [-1])
        edge_local = tf.reshape(tf.stack([2 * forward_local, 2 * forward_local + 1], axis=1), [-1])

        max_edges = 2 * tf.reduce_max(forward_counts)

        edges = tf.transpose(tf.scatter_nd(indices=tf.stack([edge_batch, edge_local], axis=1),
                                           updates=edge_values,
                                           shape=tf.stack([batch_size, max_edges, 2])), [0, 2, 1])

        nodes_mask = tf.sequence_mask(node_counts, maxlen=max_nodes)
        edges_mask = tf.sequence_mask(2 * forward_counts, maxlen=max_edges)

        return nodes, edges, nodes_mask, edges_mask, coordinates

    def _build_inputs(self, inputs):
        """
        Prepares images and computes a binary ink mask.

        Parameters
        ----------
        inputs : tf.Tensor
            Batch of padded grayscale images.

        Returns
        -------
        tf.Tensor
            Binary ink masks.
        """

        image = tf.cast(inputs[..., 0], tf.float32)

        minimum = tf.reduce_min(image, axis=(1, 2), keepdims=True)
        maximum = tf.reduce_max(image, axis=(1, 2), keepdims=True)

        content = tf.not_equal(image, minimum)

        heights = tf.reduce_sum(tf.cast(tf.reduce_any(content, axis=2), tf.int32), axis=1)
        widths = tf.reduce_sum(tf.cast(tf.reduce_any(content, axis=1), tf.int32), axis=1)

        height = tf.shape(image)[1]
        width = tf.shape(image)[2]

        rows = tf.range(height, dtype=tf.int32)[tf.newaxis, :, tf.newaxis]
        cols = tf.range(width, dtype=tf.int32)[tf.newaxis, :]

        valid = (
            (rows < heights[:, tf.newaxis, tf.newaxis])
            & (cols < widths[:, tf.newaxis, tf.newaxis])
        )

        image = tf.where(valid, image, maximum)
        image = image[..., tf.newaxis]

        mean = tf.nn.avg_pool2d(image, ksize=31, strides=1, padding='SAME')
        mean_sq = tf.nn.avg_pool2d(tf.square(image), ksize=31, strides=1, padding='SAME')

        variance = tf.maximum(mean_sq - tf.square(mean), 0.0)
        stddev = tf.sqrt(variance)

        threshold = mean * (1.0 + 0.1 * (stddev / 128.0 - 1.0))

        return image[..., 0] <= threshold[..., 0]

    def call(self, inputs):
        """
        Forward pass of the layer.

        Parameters
        ----------
        inputs : tf.Tensor
            Batch of padded grayscale images.

        Returns
        -------
        coordinates : tf.Tensor
            Original pixel coordinates.
        nodes : tf.Tensor
            Node features with configurable positions.
        edges : tf.Tensor
            Edge indices.
        nodes_mask : tf.Tensor
            Boolean validity mask for nodes.
        edges_mask : tf.Tensor
            Boolean validity mask for edges.
        """

        ink = self._build_inputs(inputs=inputs)
        skeleton = self._thinning(ink=ink)

        outputs = self._skeleton_to_graphs(skeleton=skeleton)
        outputs = tf.nest.map_structure(tf.stop_gradient, outputs)

        return outputs


class GraphNormalization(tf.keras.layers.Layer):
    """
    Graph normalization layer for stabilizing training of graph neural networks.

    References
    ----------
    GraphNorm: A Principled Approach to Accelerating Graph Neural Network Training
        https://arxiv.org/abs/2009.03294
    """

    def __init__(self, epsilon=1e-3, **kwargs):
        """
        Initializes the graph normalization layer.

        Parameters
        ----------
        epsilon : float, optional
            Small float added to variance to avoid dividing by zero.
        **kwargs : dict
            Additional keyword arguments.
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
            Shape of the input tensor.
        """

        super().build(input_shape)

        self.features = input_shape[-1]

        self.alpha = self.add_weight(name=f'{self.name}_alpha',
                                     shape=(self.features,),
                                     initializer='ones',
                                     trainable=True)

        self.gamma = self.add_weight(name=f'{self.name}_gamma',
                                     shape=(self.features,),
                                     initializer='ones',
                                     trainable=True)

        self.beta = self.add_weight(name=f'{self.name}_beta',
                                    shape=(self.features,),
                                    initializer='zeros',
                                    trainable=True)

    def call(self, inputs, nodes_mask=None):
        """
        Forward pass of the layer.

        Parameters
        ----------
        inputs : tf.Tensor
            Features tensor.
        nodes_mask : tf.Tensor, optional
            Validity mask.

        Returns
        -------
        tf.Tensor
            Normalized feature tensor.
        """

        mask = tf.identity(nodes_mask)

        if mask is None:
            mask = tf.ones(tf.shape(inputs)[:2], dtype=tf.bool)

        mask = mask[..., tf.newaxis]
        inputs = tf.where(mask, inputs, tf.zeros_like(inputs))

        count = tf.reduce_sum(tf.cast(mask, inputs.dtype), axis=1, keepdims=True)
        count = tf.maximum(count, 1.0)

        mean = tf.reduce_sum(inputs, axis=1, keepdims=True) / count

        centered = inputs - self.alpha * mean
        centered = tf.where(mask, centered, tf.zeros_like(centered))
        variance = tf.reduce_sum(tf.square(centered), axis=1, keepdims=True) / count

        normed = centered / tf.sqrt(variance + self.epsilon)

        outputs = normed * self.gamma + self.beta
        outputs = tf.where(mask, outputs, tf.zeros_like(outputs))

        return outputs


class PositionEmbedding1D(tf.keras.layers.Layer):
    """
    1D Positional Embedding layer.
    Adds learnable positional embeddings to 1D input tensors.

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
        self.embeddings_initializer = embeddings_initializer
        self.embeddings_regularizer = embeddings_regularizer
        self.embeddings_constraint = embeddings_constraint

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
            'embeddings_initializer': self.embeddings_initializer,
            'embeddings_regularizer': self.embeddings_regularizer,
            'embeddings_constraint': self.embeddings_constraint,
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
            Tensor of features.

        Returns
        -------
        tf.Tensor
            Tensor with 1D positional embeddings added.
        """

        input_shape = tf.shape(inputs)
        seq_len = input_shape[1]

        pos_emb = self.pos_embeddings[:seq_len, :]
        pos_emb = tf.expand_dims(pos_emb, axis=0)

        return pos_emb


class PositionEmbedding2D(tf.keras.layers.Layer):
    """
    2D Positional Embedding layer.
    Adds learnable row and column embeddings to 2D input tensors.

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
        self.embeddings_initializer = embeddings_initializer
        self.embeddings_regularizer = embeddings_regularizer
        self.embeddings_constraint = embeddings_constraint

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
            'embeddings_initializer': self.embeddings_initializer,
            'embeddings_regularizer': self.embeddings_regularizer,
            'embeddings_constraint': self.embeddings_constraint,
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
            Tensor of features.

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

        return pos_emb


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
