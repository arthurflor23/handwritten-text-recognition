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

        self.features = input_shape[0][-1]

        self.beta_dense = tf.keras.layers.Dense(self.features, use_bias=False)
        self.gamma_dense = tf.keras.layers.Dense(self.features, use_bias=False)

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

        beta = tf.reshape(beta, [-1, 1, 1, self.features])
        gamma = tf.reshape(gamma, [-1, 1, 1, self.features])

        normed = self.norm_layer(x, training=training)
        outputs = normed * gamma + beta

        return outputs


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

        self.beta_dense = tf.keras.layers.Dense(self.features, use_bias=False)
        self.gamma_dense = tf.keras.layers.Dense(self.features, use_bias=False)

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

        beta = tf.reshape(beta, [-1, 1, 1, self.features])
        gamma = tf.reshape(gamma, [-1, 1, 1, self.features])

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
        output_shape = tf.TensorShape(input_shape[-1:] + mask_shape[:3])

        return output_shape


class GatedResidualConv2D(tf.keras.layers.Layer):
    """
    Implements a Gated Residual Convolutional layer.

    References
    ----------
    HTSR-Pollen: Handwritten Text Synthesis and Recognition System to Overcome Data Scarcity
        https://ieeexplore.ieee.org/document/11475723
    """

    def __init__(self,
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
            Dropout rate applied during training.
        use_bias : bool, optional
            Enables bias terms in the projection layers.
        **kwargs : dict
            Additional keyword arguments.
        """

        super().__init__(**kwargs)

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

        self.s_conv = tf.keras.layers.Conv2D(filters=input_shape[-1],
                                             kernel_size=self.kernel_size,
                                             padding='same',
                                             kernel_initializer=self.kernel_initializer,
                                             kernel_regularizer=self.kernel_regularizer,
                                             kernel_constraint=self.kernel_constraint,
                                             use_bias=self.use_bias)

        self.beta = self.add_weight(name=f'{self.name}_beta',
                                    shape=(1,),
                                    initializer=self.beta_initializer,
                                    trainable=True)

        self.gamma = self.add_weight(name=f'{self.name}_gamma',
                                     shape=(1,),
                                     initializer=self.gamma_initializer,
                                     trainable=True)

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
            Output tensor.
        """

        s_conv = self.s_conv(inputs)
        g_conv = s_conv * tf.nn.sigmoid(s_conv * self.gamma)

        if training and self.dropout:
            g_conv = tf.nn.dropout(g_conv, rate=self.dropout)

        return inputs + g_conv * self.beta


class GatedResidualDense(tf.keras.layers.Layer):
    """
    Implements a Gated Residual Dense layer.

    References
    ----------
    HTSR-Pollen: Handwritten Text Synthesis and Recognition System to Overcome Data Scarcity
        https://ieeexplore.ieee.org/document/11475723
    """

    def __init__(self,
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
            Dropout rate applied during training.
        use_bias : bool, optional
            Enables bias terms in the projection layers.
        **kwargs : dict
            Additional keyword arguments.
        """

        super().__init__(**kwargs)

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

        self.s_dense = tf.keras.layers.Dense(units=input_shape[-1],
                                             kernel_initializer=self.kernel_initializer,
                                             kernel_regularizer=self.kernel_regularizer,
                                             kernel_constraint=self.kernel_constraint,
                                             use_bias=self.use_bias)

        self.beta = self.add_weight(name=f'{self.name}_beta',
                                    shape=(1,),
                                    initializer=self.beta_initializer,
                                    trainable=True)

        self.gamma = self.add_weight(name=f'{self.name}_gamma',
                                     shape=(input_shape[-1], 1),
                                     initializer=self.gamma_initializer,
                                     trainable=True)

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
            Output tensor.
        """

        s_dense = self.s_dense(inputs)
        g_dense = s_dense * tf.nn.sigmoid(tf.linalg.matmul(s_dense, self.gamma))

        if training and self.dropout:
            stddev = tf.sqrt(self.dropout / (1.0 - self.dropout))
            g_dense *= tf.random.normal(tf.shape(g_dense), 1.0, stddev, g_dense.dtype)

        return inputs + g_dense * self.beta


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
                                            use_bias=False)

        if self.use_bias:
            self.bias = self.add_weight(name=f'{self.name}_bias',
                                        shape=(self.units,),
                                        initializer='zeros',
                                        trainable=True)

    def call(self, inputs):
        """
        Forward pass of the layer.

        Parameters
        ----------
        inputs : tuple of tf.Tensor
            Tuple of node features and edge indices.

        Returns
        -------
        tf.Tensor
            Updated node features.
        """

        nodes, edges = inputs

        batch_size = tf.shape(nodes)[0]
        num_nodes = tf.shape(nodes)[1]

        src = tf.cast(edges[:, 0, :], tf.int32)
        dst = tf.cast(edges[:, 1, :], tf.int32)

        self_idx = tf.range(num_nodes, dtype=tf.int32)[tf.newaxis, :]
        self_idx = tf.tile(self_idx, [batch_size, 1])

        src = tf.concat([src, self_idx], axis=1)
        dst = tf.concat([dst, self_idx], axis=1)

        transformed = self.kernel(nodes)

        batch_offset = tf.range(batch_size, dtype=tf.int32)[:, tf.newaxis] * num_nodes
        dst_flat = tf.reshape(dst + batch_offset, [-1])
        total_segments = batch_size * num_nodes

        ones_flat = tf.ones(tf.shape(dst_flat), dtype=transformed.dtype)
        degrees_flat = tf.math.unsorted_segment_sum(ones_flat, dst_flat, total_segments)
        degrees = tf.reshape(degrees_flat, [batch_size, num_nodes])

        deg_inv_sqrt = tf.math.rsqrt(tf.maximum(degrees, 1.0))

        src_norm = tf.gather(deg_inv_sqrt, src, batch_dims=1)
        dst_norm = tf.gather(deg_inv_sqrt, dst, batch_dims=1)
        norm = src_norm * dst_norm

        messages = tf.gather(transformed, src, batch_dims=1) * norm[..., tf.newaxis]
        messages_flat = tf.reshape(messages, [-1, tf.shape(messages)[-1]])

        outputs_flat = tf.math.unsorted_segment_sum(messages_flat, dst_flat, total_segments)
        outputs = tf.reshape(outputs_flat, [batch_size, num_nodes, self.units])

        if self.use_bias:
            outputs = outputs + self.bias

        return outputs


class GraphConv2D(tf.keras.layers.Layer):
    """
    Graph convolution layer over spatial features for modeling non-local dependencies.

    References
    ----------
    Image Segmentation: Inducing graph-based learning
        https://arxiv.org/abs/2501.03765
    """

    def __init__(self,
                 units,
                 k_neighbors=1,
                 num_pos_scales=1,
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
        k_neighbors : int, optional
            Number of neighbors used to build the graph.
        num_pos_scales : int, optional
            Number of sine-cosine positional scales used in the spatial offsets.
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
        self.k_neighbors = k_neighbors
        self.num_pos_scales = num_pos_scales
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
            'k_neighbors': self.k_neighbors,
            'num_pos_scales': self.num_pos_scales,
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

        self.height = input_shape[1]
        self.width = input_shape[2]
        self.channels = input_shape[3]

        self.kernel = tf.keras.layers.Dense(units=self.units,
                                            kernel_initializer=self.kernel_initializer,
                                            kernel_regularizer=self.kernel_regularizer,
                                            kernel_constraint=self.kernel_constraint,
                                            use_bias=False)

        if self.use_bias:
            self.bias = self.add_weight(name=f'{self.name}_bias',
                                        shape=(self.units,),
                                        initializer='zeros',
                                        trainable=True)

    def _build_base_coordinates(self):
        """
        Builds the base spatial coordinates.

        Returns
        -------
        tf.Tensor
            Coordinate tensor.
        """

        row_positions = tf.linspace(0.0, 1.0, self.height)
        col_positions = tf.linspace(0.0, 1.0, self.width)

        row_grid, col_grid = tf.meshgrid(row_positions, col_positions, indexing='ij')

        return tf.stack([row_grid, col_grid], axis=-1)

    def _build_positional_offsets(self):
        """
        Builds fixed sine-cosine positional offsets.

        Returns
        -------
        tf.Tensor
            Offset tensor.
        """

        coords = self._build_base_coordinates()
        two_pi = tf.cast(2.0 * tf.experimental.numpy.pi, tf.float32)

        offset_x = tf.zeros([self.height, self.width], dtype=tf.float32)
        offset_y = tf.zeros([self.height, self.width], dtype=tf.float32)

        for i in range(self.num_pos_scales):
            freq = tf.cast(2 ** i, tf.float32)

            offset_x += tf.sin(two_pi * freq * coords[..., 0])
            offset_y += tf.cos(two_pi * freq * coords[..., 1])

        offset_x = offset_x / tf.cast(self.num_pos_scales, tf.float32)
        offset_y = offset_y / tf.cast(self.num_pos_scales, tf.float32)

        return tf.stack([offset_x, offset_y], axis=-1)

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

        batch_size = tf.shape(inputs)[0]
        num_nodes = self.height * self.width

        nodes = tf.reshape(inputs, [-1, num_nodes, self.channels])

        base_coords = self._build_base_coordinates()
        offsets = self._build_positional_offsets()

        warped_coords = tf.reshape(base_coords + offsets, [num_nodes, 2])

        diff = warped_coords[:, None, :] - warped_coords[None, :, :]
        dist = tf.reduce_sum(tf.square(diff), axis=-1)
        dist = dist + tf.eye(num_nodes, dtype=dist.dtype) * 1e12

        knn = tf.math.top_k(-dist, k=self.k_neighbors).indices
        dst = tf.repeat(tf.range(num_nodes), repeats=self.k_neighbors)
        src = tf.reshape(knn, [-1])

        edges = tf.stack([dst, src], axis=-1)
        self_loops = tf.stack([tf.range(num_nodes), tf.range(num_nodes)], axis=-1)

        edge_index = tf.concat([edges, self_loops], axis=0)

        dst = edge_index[:, 0]
        src = edge_index[:, 1]

        transformed = self.kernel(tf.gather(nodes, src, axis=1))

        batch_offset = tf.range(batch_size, dtype=tf.int32)[:, tf.newaxis] * num_nodes
        dst_flat = tf.reshape(dst[tf.newaxis, :] + batch_offset, [-1])
        total_segments = batch_size * num_nodes

        neighbor_features_flat = tf.reshape(transformed, [-1, tf.shape(transformed)[-1]])

        outputs_flat = tf.math.unsorted_segment_sum(neighbor_features_flat, dst_flat, total_segments)
        outputs = tf.reshape(outputs_flat, [batch_size, self.height, self.width, self.units])

        if self.use_bias:
            outputs = outputs + self.bias

        return outputs


class GraphExtraction(tf.keras.layers.Layer):
    """
    Extracts graph representations from handwriting images via skeletonization.

    References
    ----------
    Combining graph edit distance and triplet networks for offline signature verification
        https://www.sciencedirect.com/science/article/pii/S0167865519301850

    Graph Embedding for Offline Handwritten Signature Verification
        https://dl.acm.org/doi/10.1145/3345336.3345346

    Towards Visuo-Structural Handwriting Evaluation Based on Graph Matching
        https://link.springer.com/chapter/10.1007/978-3-031-45461-5_6
    """

    def __init__(self, k_length=1, maximum_iterations=100, **kwargs):
        """
        Initializes the layer.

        Parameters
        ----------
        k_length : int, optional
            Distance between sampled keypoints along the skeleton.
        maximum_iterations : int, optional
            Maximum number of thinning iterations.
        **kwargs : dict
            Additional keyword arguments.
        """

        super().__init__(**kwargs)

        self.k_length = k_length
        self.maximum_iterations = maximum_iterations

        self._offsets = tf.range(self.k_length, dtype=tf.int32)

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
            'k_length': self.k_length,
            'maximum_iterations': self.maximum_iterations,
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
                                 maximum_iterations=self.maximum_iterations)

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

        grid_h = tf.math.floordiv(height + self.k_length - 1, self.k_length)
        grid_w = tf.math.floordiv(width + self.k_length - 1, self.k_length)

        pad_h = grid_h * self.k_length - height
        pad_w = grid_w * self.k_length - width

        paddings = tf.stack([
            tf.constant([0, 0], dtype=tf.int32),
            tf.stack([tf.constant(0, tf.int32), pad_h]),
            tf.stack([tf.constant(0, tf.int32), pad_w]),
        ])

        padded = tf.pad(skeleton, paddings, constant_values=False)

        blocks = tf.reshape(padded, tf.stack([batch_size, grid_h, self.k_length, grid_w, self.k_length]))
        blocks_int = tf.cast(blocks, tf.int32)

        col_counts = tf.reduce_sum(blocks_int, axis=2)
        row_counts = tf.reduce_sum(blocks_int, axis=4)
        counts = tf.reduce_sum(col_counts, axis=3)

        local_x_sum = tf.reduce_sum(col_counts * self._offsets[tf.newaxis, tf.newaxis, tf.newaxis, :], axis=3)
        local_y_sum = tf.reduce_sum(row_counts * self._offsets[tf.newaxis, tf.newaxis, :, tf.newaxis], axis=2)

        grid_x = tf.range(grid_w, dtype=tf.int32)[tf.newaxis, tf.newaxis, :]
        grid_y = tf.range(grid_h, dtype=tf.int32)[tf.newaxis, :, tf.newaxis]

        sum_x = local_x_sum + counts * (grid_x * self.k_length)
        sum_y = local_y_sum + counts * (grid_y * self.k_length)

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
            Zero-padded pixel coordinates.
        nodes : tf.Tensor
            Zero-padded node features.
        edges : tf.Tensor
            Zero-padded edge indices.
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

        xy_min = tf.math.unsorted_segment_min(xy, node_batch, batch_size)
        xy_max = tf.math.unsorted_segment_max(xy, node_batch, batch_size)

        graph_scale = tf.maximum(tf.reduce_max(xy_max - xy_min, axis=1), 1.0)
        graph_min = tf.gather(xy_min, node_batch)
        scale = tf.gather(graph_scale, node_batch)

        normalized_xy = (xy - graph_min) / scale[:, tf.newaxis]
        node_values = tf.pad(normalized_xy, [[0, 0], [0, 2]])

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

        edges = tf.scatter_nd(indices=tf.stack([edge_batch, edge_local], axis=1),
                              updates=edge_values,
                              shape=tf.stack([batch_size, max_edges, 2]))

        return coordinates, nodes, tf.transpose(edges, [0, 2, 1])

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
        cols = tf.range(width, dtype=tf.int32)[tf.newaxis, tf.newaxis, :]

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
            Zero-padded pixel coordinates.
        nodes : tf.Tensor
            Zero-padded node features.
        edges : tf.Tensor
            Zero-padded edge indices.
        """

        ink = self._build_inputs(inputs=inputs)
        skeleton = self._thinning(ink=ink)
        coordinates, nodes, edges = self._skeleton_to_graphs(skeleton=skeleton)

        coordinates = tf.stop_gradient(coordinates)
        nodes = tf.stop_gradient(nodes)
        edges = tf.stop_gradient(edges)

        return coordinates, nodes, edges

    def compute_output_shape(self, input_shape):
        """
        Computes the output shape of the layer.

        Parameters
        ----------
        input_shape : tuple
            Shape of the input tensors.

        Returns
        -------
        tuple of tf.TensorShape
            Output shape.
        """

        if not self.built:
            self.build(input_shape)

        coordinates_shape = tf.TensorShape([input_shape[0], None, 2])
        nodes_shape = tf.TensorShape([input_shape[0], None, 4])
        edges_shape = tf.TensorShape([input_shape[0], 2, None])

        return coordinates_shape, nodes_shape, edges_shape


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

        self.alpha = self.add_weight(name='alpha',
                                     shape=(self.features,),
                                     initializer='ones',
                                     trainable=True)

        self.gamma = self.add_weight(name='gamma',
                                     shape=(self.features,),
                                     initializer='ones',
                                     trainable=True)

        self.beta = self.add_weight(name='beta',
                                    shape=(self.features,),
                                    initializer='zeros',
                                    trainable=True)

    def call(self, inputs):
        """
        Forward pass of the layer.

        Parameters
        ----------
        inputs : tf.Tensor
            Node feature tensor.

        Returns
        -------
        tf.Tensor
            Normalized node feature tensor.
        """

        mean = tf.reduce_mean(inputs, axis=1, keepdims=True)

        centered = inputs - self.alpha * mean
        variance = tf.reduce_mean(tf.square(centered), axis=1, keepdims=True)

        normed = centered / tf.sqrt(variance + self.epsilon)
        outputs = normed * self.gamma + self.beta

        return outputs


class GraphPooling(tf.keras.layers.Layer):
    """
    Graph pooling layer for reducing the number of graph nodes.

    References
    ----------
    Graph U-Nets
        https://arxiv.org/abs/1905.05178

    Self-Attention Graph Pooling
        https://arxiv.org/abs/1904.08082
    """

    def __init__(self,
                 strides=1,
                 kernel_initializer='glorot_uniform',
                 kernel_regularizer=None,
                 kernel_constraint=None,
                 use_bias=True,
                 **kwargs):
        """
        Initializes the layer.

        Parameters
        ----------
        strides : int, optional
            Graph reduction factor.
        kernel_initializer : initializer, optional
            Kernel weights initializer for the score projection.
        kernel_regularizer : regularizer, optional
            Kernel weights regularizer for the score projection.
        kernel_constraint : constraint, optional
            Kernel weights constraint for the score projection.
        use_bias : bool, optional
            Enables bias term in the score projection layer.
        **kwargs : dict
            Additional keyword arguments.
        """

        super().__init__(**kwargs)

        self.strides = strides
        self.kernel_initializer = kernel_initializer
        self.kernel_regularizer = kernel_regularizer
        self.kernel_constraint = kernel_constraint
        self.use_bias = use_bias

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
            'strides': self.strides,
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
        input_shape : TensorShape
            Shape of the input tensor.
        """

        super().build(input_shape)

        self.gate_dense = tf.keras.layers.Dense(units=1,
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
            Node feature tensor.

        Returns
        -------
        tf.Tensor
            Reduced node feature tensor.
        """

        scores = self.gate_dense(inputs)
        scores = tf.squeeze(scores, axis=-1)

        nodes = tf.math.ceil(tf.cast(tf.shape(inputs)[1], tf.float32) / self.strides)
        _, indices = tf.math.top_k(scores, k=tf.cast(nodes, tf.int32), sorted=False)

        indices = tf.sort(indices, axis=1)

        outputs = tf.gather(inputs, indices, axis=1, batch_dims=1)
        scores = tf.gather(scores, indices, axis=1, batch_dims=1)

        outputs = outputs * tf.nn.sigmoid(scores)[..., tf.newaxis]

        return outputs


class GlobalGraphPooling(tf.keras.layers.Layer):
    """
    Global Graph pooling layer for aggregating node features into embeddings.

    References
    ----------
    Gated Graph Sequence Neural Networks
        https://arxiv.org/abs/1511.05493

    Self-Attention Graph Pooling
        https://arxiv.org/pdf/1904.08082
    """

    def __init__(self,
                 kernel_initializer='glorot_uniform',
                 kernel_regularizer=None,
                 kernel_constraint=None,
                 use_bias=True,
                 **kwargs):
        """
        Initializes the layer.

        Parameters
        ----------
        kernel_initializer : initializer, optional
            Kernel weights initializer for the gate projection.
        kernel_regularizer : regularizer, optional
            Kernel weights regularizer for the gate projection.
        kernel_constraint : constraint, optional
            Kernel weights constraint for the gate projection.
        use_bias : bool, optional
            Enables bias term in the gate projection layer.
        **kwargs : dict
            Additional keyword arguments.
        """

        super().__init__(**kwargs)

        self.kernel_initializer = kernel_initializer
        self.kernel_regularizer = kernel_regularizer
        self.kernel_constraint = kernel_constraint
        self.use_bias = use_bias

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
        input_shape : TensorShape
            Shape of the input tensor.
        """

        super().build(input_shape)

        self.gate_dense = tf.keras.layers.Dense(units=1,
                                                kernel_initializer=self.kernel_initializer,
                                                kernel_regularizer=self.kernel_regularizer,
                                                kernel_constraint=self.kernel_constraint,
                                                use_bias=self.use_bias)

        self.softmax = tf.keras.layers.Softmax(axis=1)

    def call(self, inputs):
        """
        Forward pass of the layer.

        Parameters
        ----------
        inputs : tf.Tensor
            Node feature tensor.

        Returns
        -------
        tf.Tensor
            Graph-level feature vector.
        """

        attn_scores = self.gate_dense(inputs)
        attn_weights = self.softmax(attn_scores)

        outputs = inputs * attn_weights
        outputs = tf.reduce_sum(outputs, axis=1)

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
