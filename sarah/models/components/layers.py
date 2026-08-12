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

        self.kernel = self.add_weight(name=f'{self.name}_kernel',
                                      shape=(input_shape[0][-1], self.units),
                                      initializer=self.kernel_initializer,
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint,
                                      trainable=True)

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

        transformed = tf.linalg.matmul(nodes, self.kernel)

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
        outputs = tf.reshape(outputs_flat, [batch_size, num_nodes, -1])

        if self.use_bias:
            outputs = outputs + self.bias

        return outputs

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

        nodes_shape = input_shape[0]
        output_shape = tf.TensorShape(nodes_shape[:-1] + (self.units,))

        return output_shape


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

        self.kernel = self.add_weight(name=f'{self.name}_kernel',
                                      shape=(input_shape[-1], self.units),
                                      initializer=self.kernel_initializer,
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint,
                                      trainable=True)

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

        transformed = tf.linalg.matmul(tf.gather(nodes, src, axis=1), self.kernel)

        batch_offset = tf.range(batch_size, dtype=tf.int32)[:, tf.newaxis] * num_nodes
        dst_flat = tf.reshape(dst[tf.newaxis, :] + batch_offset, [-1])
        total_segments = batch_size * num_nodes

        neighbor_features_flat = tf.reshape(transformed, [-1, tf.shape(transformed)[-1]])
        outputs_flat = tf.math.unsorted_segment_sum(neighbor_features_flat, dst_flat, total_segments)
        outputs = tf.reshape(outputs_flat, [batch_size, num_nodes, -1])

        if self.use_bias:
            outputs = outputs + self.bias

        return tf.reshape(outputs, [-1, self.height, self.width, self.units])

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

        output_shape = [input_shape[0], self.height, self.width, self.units]
        output_shape = tf.TensorShape(output_shape)

        return output_shape


class GraphEncoder(tf.keras.layers.Layer):
    """
    Encodes graph representations from handwriting images via skeletonization.

    References
    ----------
    Combining graph edit distance and triplet networks for offline signature verification
        https://www.sciencedirect.com/science/article/pii/S0167865519301850

    Graph Embedding for Offline Handwritten Signature Verification
        https://dl.acm.org/doi/10.1145/3345336.3345346

    Towards Visuo-Structural Handwriting Evaluation Based on Graph Matching
        https://link.springer.com/chapter/10.1007/978-3-031-45461-5_6
    """

    def __init__(self,
                 filters=None,
                 units=None,
                 dropout=0.0,
                 activation=None,
                 kernel_initializer='glorot_uniform',
                 kernel_regularizer=None,
                 kernel_constraint=None,
                 beta_initializer='zeros',
                 gamma_initializer='ones',
                 use_residual=False,
                 use_pooling=False,
                 use_gate=False,
                 use_bias=True,
                 **kwargs):
        """
        Initializes the layer.

        Parameters
        ----------
        filters : int or list, optional
            Output units for each graph convolution layer.
        units : int or list, optional
            Output units for each dense projection layer.
        dropout : float, optional
            Dropout rate applied during training.
        activation : str, optional
            Activation applied after each layer.
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
        use_residual : bool, optional
            Enables residual connections.
        use_pooling : bool, optional
            Enables downsampling in projection stage.
        use_gate : bool, optional
            Enables gated residual connections.
        use_bias : bool, optional
            Enables bias terms in the layers.
        **kwargs : dict
            Additional keyword arguments.
        """

        super().__init__(**kwargs)

        self.filters = [filters] if isinstance(filters, int) else (filters or [])
        self.units = [units] if isinstance(units, int) else (units or [])
        self.dropout = dropout
        self.activation = activation
        self.kernel_initializer = kernel_initializer
        self.kernel_regularizer = kernel_regularizer
        self.kernel_constraint = kernel_constraint
        self.beta_initializer = beta_initializer
        self.gamma_initializer = gamma_initializer
        self.use_residual = use_residual
        self.use_pooling = use_pooling
        self.use_gate = use_gate
        self.use_bias = use_bias

        self._offset_dy = tf.constant([-1, -1, -1, 0, 1, 1, 1, 0], dtype=tf.int32)
        self._offset_dx = tf.constant([-1, 0, 1, 1, 1, 0, -1, -1], dtype=tf.int32)
        self._offset_dist = tf.constant([2 ** 0.5, 1.0] * 4, dtype=tf.float32)

        self._lut0 = tf.constant(self._zhang_suen_lut(iteration=0), dtype=tf.float32)
        self._lut1 = tf.constant(self._zhang_suen_lut(iteration=1), dtype=tf.float32)

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
            'units': self.units,
            'dropout': self.dropout,
            'activation': tf.keras.activations.serialize(self.activation),
            'kernel_initializer': self.kernel_initializer,
            'kernel_regularizer': self.kernel_regularizer,
            'kernel_constraint': self.kernel_constraint,
            'beta_initializer': self.beta_initializer,
            'gamma_initializer': self.gamma_initializer,
            'use_residual': self.use_residual,
            'use_pooling': self.use_pooling,
            'use_gate': self.use_gate,
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

        if not self.filters:
            return

        self.activation = tf.keras.activations.get(self.activation)

        self.kernels = [
            self.add_weight(name=f'{self.name}_kernel_{i}',
                            shape=(4 if i == 0 else self.filters[i - 1], units),
                            initializer=self.kernel_initializer,
                            regularizer=self.kernel_regularizer,
                            constraint=self.kernel_constraint,
                            trainable=True)
            for i, units in enumerate(self.filters)
        ]

        if self.use_residual:
            self.residual_dense_layers = [
                tf.keras.layers.Dense(units=units,
                                      kernel_initializer=self.kernel_initializer,
                                      kernel_regularizer=self.kernel_regularizer,
                                      kernel_constraint=self.kernel_constraint,
                                      use_bias=self.use_bias)
                for units in self.filters
            ]

        self.projection_layers = [
            tf.keras.layers.Dense(units=units,
                                  kernel_initializer=self.kernel_initializer,
                                  kernel_regularizer=self.kernel_regularizer,
                                  kernel_constraint=self.kernel_constraint,
                                  use_bias=self.use_bias)
            for units in self.units
        ]

        if self.use_gate:
            self.gate_projection_layers = [
                tf.keras.layers.Dense(units=units,
                                      kernel_initializer=self.kernel_initializer,
                                      kernel_regularizer=self.kernel_regularizer,
                                      kernel_constraint=self.kernel_constraint,
                                      use_bias=self.use_bias)
                for units in self.units
            ]

            self.gate_betas = [
                self.add_weight(name=f'{self.name}_gate_beta_{i}',
                                shape=(1,),
                                initializer=self.beta_initializer,
                                trainable=True)
                for i in range(len(self.units))
            ]

            self.gate_gammas = [
                self.add_weight(name=f'{self.name}_gate_gamma_{i}',
                                shape=(units, 1),
                                initializer=self.gamma_initializer,
                                trainable=True)
                for i, units in enumerate(self.units)
            ]

        if self.use_bias:
            self.biases = [
                self.add_weight(name=f'{self.name}_bias_{i}',
                                shape=(units,),
                                initializer='zeros',
                                trainable=True)
                for i, units in enumerate(self.filters)
            ]

        if self.use_pooling:
            self.pooling = tf.keras.layers.AveragePooling1D(pool_size=2, strides=2)

        self.global_pooling = tf.keras.layers.GlobalAveragePooling1D()

    def _zhang_suen_lut(self, iteration):
        """
        Builds one Zhang-Suen thinning deletion lookup table.

        Parameters
        ----------
        iteration : int
            Sub-iteration selector.

        Returns
        -------
        list
            Deletion lookup table.
        """

        lut = []

        for code in range(256):
            p9, p2, p3, p4, p5, p6, p7, p8 = [(code >> k) & 1 for k in range(8)]
            seq = [p2, p3, p4, p5, p6, p7, p8, p9, p2]

            b = sum(seq[:8])
            a = sum(seq[i] == 0 and seq[i + 1] == 1 for i in range(8))
            m1, m2 = (p2 * p4 * p6, p4 * p6 * p8) if iteration == 0 else (p2 * p4 * p8, p2 * p6 * p8)

            delete = 2 <= b <= 6 and a == 1 and m1 == 0 and m2 == 0
            lut.append(0 if delete else 1)

        return lut

    def _thinning(self, ink_mask):
        """
        Thins a binary ink mask to a 1-pixel skeleton via Zhang-Suen.

        Parameters
        ----------
        ink_mask : tf.Tensor
            Binary ink mask.

        Returns
        -------
        tf.Tensor
            Binary skeleton mask.
        """

        def sub_iteration(img, lut):
            patches = tf.image.extract_patches(images=img,
                                               sizes=[1, 3, 3, 1],
                                               strides=[1, 1, 1, 1],
                                               rates=[1, 1, 1, 1],
                                               padding='SAME')

            nw, n, ne, w, center, e, sw, s, se = tf.unstack(patches, axis=-1)
            p = [n, ne, e, se, s, sw, w, nw]

            index = tf.add_n([p[k] * weights[k] for k in range(8)])
            index = tf.cast(index, tf.int32)
            delete = (1.0 - tf.gather(lut, index)) * center

            return img * (1.0 - delete[..., tf.newaxis] * interior)

        def cond(img, changed, i):
            return tf.logical_and(changed, i < 200)

        def body(img, changed, i):
            next_img = sub_iteration(img=sub_iteration(img=img, lut=self._lut0), lut=self._lut1)
            return next_img, tf.reduce_any(tf.not_equal(next_img, img)), i + 1

        weights = tf.constant([2, 4, 8, 16, 32, 64, 128, 1], dtype=tf.float32)
        height, width = ink_mask.shape[0], ink_mask.shape[1]

        interior = tf.ones([height - 2, width - 2], dtype=tf.float32)
        interior = tf.pad(interior, paddings=[[1, 1], [1, 1]], )[tf.newaxis, :, :, tf.newaxis]

        image = tf.cast(ink_mask, tf.float32)[tf.newaxis, :, :, tf.newaxis]

        image, _, _ = tf.while_loop(cond=cond,
                                    body=body,
                                    loop_vars=[image, tf.constant(True), tf.constant(0)])

        return image[0, :, :, 0]

    def _connected_components(self, mask, connectivity=8):
        """
        Labels 8- or 4-connected components of a binary mask.

        Parameters
        ----------
        mask : tf.Tensor
            Binary mask.
        connectivity : int, optional
            Pixel connectivity.

        Returns
        -------
        num_labels : tf.Tensor
            Number of labels, including background.
        labels : tf.Tensor
            Labeled image.
        """

        def cond(label, changed, i):
            return tf.logical_and(changed, i < (height * width))

        def body(label, changed, i):
            padded = tf.pad(tensor=label, paddings=[[1, 1], [1, 1]], constant_values=sentinel)
            cands = [label] + [padded[1 + dy:1 + dy + height, 1 + dx:1 + dx + width] for dy, dx in offsets]
            next_label = tf.where(mask_bool, tf.reduce_min(tf.stack(cands), axis=0), sentinel)
            return next_label, tf.reduce_any(tf.not_equal(next_label, label)), i + 1

        height, width = mask.shape[0], mask.shape[1]
        mask_bool = tf.cast(mask, tf.bool)
        sentinel = tf.constant(height * width, dtype=tf.int32)

        rows = tf.range(height, dtype=tf.int32)[:, tf.newaxis]
        cols = tf.range(width, dtype=tf.int32)[tf.newaxis, :]
        label = tf.where(mask_bool, rows * width + cols, sentinel)

        if connectivity == 8:
            offsets = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        else:
            offsets = [(-1, 0), (0, -1), (0, 1), (1, 0)]

        label, _, _ = tf.while_loop(cond=cond,
                                    body=body,
                                    loop_vars=[label, tf.constant(True), tf.constant(0)])

        unique_labels, fg_inverse = tf.unique(tf.boolean_mask(label, mask_bool))

        compact = tf.tensor_scatter_nd_update(tensor=tf.zeros([height, width], dtype=tf.int32),
                                              indices=tf.where(mask_bool),
                                              updates=tf.cast(fg_inverse + 1, tf.int32))

        return tf.shape(unique_labels)[0] + 1, compact

    def _build_endpoints(self, skeleton, image_shape):
        """
        Detects stroke endpoints and junction candidates in a skeleton.

        Parameters
        ----------
        skeleton : tf.Tensor
            Flattened skeleton image.
        image_shape : tf.TensorShape
            Height and width of the image.

        Returns
        -------
        endpoints : tf.Tensor
            Stroke endpoint positions.
        possible_junctions : tf.Tensor
            Raw junction candidates.
        """

        mask = 1.0 - tf.reshape(skeleton, image_shape[:2])
        kernel = tf.constant([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=tf.float32)

        neighbor_sum = tf.nn.conv2d(input=mask[tf.newaxis, :, :, tf.newaxis],
                                    filters=kernel[:, :, tf.newaxis, tf.newaxis],
                                    strides=1,
                                    padding='SAME')[0, :, :, 0]

        weighted_count = tf.reshape(mask * neighbor_sum, [-1])

        endpoints = tf.cast(tf.reshape(tf.where(tf.equal(weighted_count, 1.0)), [-1]), tf.int32)
        possible_junctions = tf.cast(tf.reshape(tf.where(weighted_count > 2.0), [-1]), tf.int32)

        return endpoints, possible_junctions

    def _build_paths(self, skeleton, start_row, start_col, image_shape):
        """
        Extracts evenly spaced keypoints along the skeleton.

        Parameters
        ----------
        skeleton : tf.Tensor
            Flattened skeleton image.
        start_row, start_col : tf.Tensor
            Starting pixel position.
        image_shape : tf.TensorShape
            Height and width of the image.

        Returns
        -------
        keypoints : tf.Tensor
            Sampled positions along the path.
        skeleton : tf.Tensor
            Updated skeleton with visited pixels consumed.
        """

        def cond(skel, row, col, last_row, last_col, dist, kp, kp_n, step, active):
            return tf.logical_and(active, step < (height * width))

        def body(skel, row, col, last_row, last_col, dist, kp, kp_n, step, active):
            next_row, next_col = row + self._offset_dy, col + self._offset_dx

            valid = (next_row >= -height) & (next_row < height) & (next_col >= -width) & (next_col < width)
            wrapped = tf.math.floormod(next_row, height) * width + tf.math.floormod(next_col, width)

            free = valid & tf.equal(tf.gather(skel, wrapped), 0.0)
            pick = tf.argmax(tf.cast(free, tf.int32), output_type=tf.int32)

            picked_row, picked_col = next_row[pick], next_col[pick]

            def step_forward():
                new_skel = tf.tensor_scatter_nd_update(tensor=skel, indices=[[wrapped[pick]]], updates=[1.0])
                new_dist = dist + self._offset_dist[pick]
                record = new_dist >= 5

                new_kp = tf.cond(
                    pred=record,
                    true_fn=lambda: kp.write(kp_n, tf.stack([picked_row, picked_col])),
                    false_fn=lambda: kp,
                )

                return (
                    new_skel,
                    picked_row,
                    picked_col,
                    tf.where(record, picked_row, last_row),
                    tf.where(record, picked_col, last_col),
                    tf.where(record, 0.0, new_dist),
                    new_kp,
                    kp_n + tf.cast(record, tf.int32),
                    step + 1,
                    True,
                )

            def stop():
                return skel, row, col, last_row, last_col, dist, kp, kp_n, step + 1, False

            return tf.cond(pred=tf.reduce_any(free), true_fn=step_forward, false_fn=stop)

        height, width = image_shape[:2]

        skeleton = tf.tensor_scatter_nd_update(tensor=skeleton,
                                               indices=[[start_row * width + start_col]],
                                               updates=[1.0])

        keypoints = tf.TensorArray(dtype=tf.int32, size=1, dynamic_size=True, element_shape=[2])
        keypoints = keypoints.write(0, tf.stack([start_row, start_col]))

        skeleton, row, col, last_row, last_col, _, keypoints, kp_n, _, _ = tf.while_loop(
            cond=cond,
            body=body,
            loop_vars=[
                skeleton, start_row, start_col, start_row, start_col, tf.constant(0.0),
                keypoints, tf.constant(1), tf.constant(0), tf.constant(True),
            ],
            shape_invariants=[
                tf.TensorShape([height * width]),
                tf.TensorShape([]), tf.TensorShape([]),
                tf.TensorShape([]), tf.TensorShape([]),
                tf.TensorShape([]),
                tf.TensorShape(None),
                tf.TensorShape([]),
                tf.TensorShape([]),
                tf.TensorShape([]),
            ],
        )

        keypoints = tf.cond(
            pred=tf.logical_or(tf.not_equal(last_row, row), tf.not_equal(last_col, col)),
            true_fn=lambda: keypoints.write(kp_n, tf.stack([row, col])),
            false_fn=lambda: keypoints,
        )

        return keypoints.stack(), skeleton

    def _convert_skeleton_to_paths(self, skeleton, image_shape):
        """
        Converts a skeleton to ordered paths, handling complex topologies.

        Parameters
        ----------
        skeleton : tf.Tensor
            Flattened skeleton image.
        image_shape : tf.TensorShape
            Height and width of the image.

        Returns
        -------
        all_rows, all_cols : tf.Tensor
            Positions of every path point.
        path_id_of_point : tf.Tensor
            Path membership of every point.
        """

        def by_col_then_row(idx):
            r, c = idx // width, idx % width
            order = tf.argsort(c * height + r)
            return tf.gather(r, order), tf.gather(c, order)

        def gather_safe(flat, rows, cols, default):
            in_range = (rows >= 0) & (rows < height) & (cols >= 0) & (cols < width)
            idx = tf.clip_by_value(rows, 0, height - 1) * width + tf.clip_by_value(cols, 0, width - 1)
            return tf.where(in_range, tf.gather(flat, idx), default)

        def walk_cond(i, skel, rows, cols, pids, pos, path_id):
            return i < tf.shape(start_rows)[0]

        def walk_body(i, skel, rows, cols, pids, pos, path_id):
            keypoints, skel = self._build_paths(skeleton=skel,
                                                start_row=start_rows[i],
                                                start_col=start_cols[i],
                                                image_shape=image_shape)

            kp_row, kp_col = keypoints[:, 0], keypoints[:, 1]

            head_anchored = gather_safe(flat=anchor_mask, rows=kp_row[:1], cols=kp_col[:1], default=False)
            tail_anchored = gather_safe(flat=anchor_mask, rows=kp_row[-1:], cols=kp_col[-1:], default=False)

            zero = tf.zeros([0], tf.int32)

            prefix_row = tf.cond(
                pred=head_anchored[0],
                true_fn=lambda: gather_safe(flat=anchor_row, rows=kp_row[:1], cols=kp_col[:1], default=0),
                false_fn=lambda: zero,
            )

            prefix_col = tf.cond(
                pred=head_anchored[0],
                true_fn=lambda: gather_safe(flat=anchor_col, rows=kp_row[:1], cols=kp_col[:1], default=0),
                false_fn=lambda: zero,
            )

            suffix_row = tf.cond(
                pred=tail_anchored[0],
                true_fn=lambda: gather_safe(flat=anchor_row, rows=kp_row[-1:], cols=kp_col[-1:], default=0),
                false_fn=lambda: zero,
            )

            suffix_col = tf.cond(
                pred=tail_anchored[0],
                true_fn=lambda: gather_safe(flat=anchor_col, rows=kp_row[-1:], cols=kp_col[-1:], default=0),
                false_fn=lambda: zero,
            )

            full_row = tf.concat([prefix_row, kp_row, suffix_row], axis=0)
            full_col = tf.concat([prefix_col, kp_col, suffix_col], axis=0)

            n = tf.shape(full_row)[0]
            span = pos + tf.range(n)

            return (
                i + 1,
                skel,
                rows.scatter(span, full_row),
                cols.scatter(span, full_col),
                pids.scatter(span, tf.fill([n], path_id)),
                pos + n,
                path_id + 1,
            )

        def loop_cond(i, skel, rows, cols, pids, pos, path_id):
            return i < n_loops

        def loop_body(i, skel, rows, cols, pids, pos, path_id):
            keypoints, skel = self._build_paths(skeleton=skel,
                                                start_row=loop_start_row[i],
                                                start_col=loop_start_col[i],
                                                image_shape=image_shape)

            n_kp = tf.shape(keypoints)[0]
            closed_row = tf.concat([keypoints[:n_kp - 1, 0], keypoints[0:1, 0]], axis=0)
            closed_col = tf.concat([keypoints[:n_kp - 1, 1], keypoints[0:1, 1]], axis=0)
            n = tf.shape(closed_row)[0]
            span = pos + tf.range(n)

            return (
                i + 1,
                skel,
                rows.scatter(span, closed_row),
                cols.scatter(span, closed_col),
                pids.scatter(span, tf.fill([n], path_id)),
                pos + n,
                path_id + 1,
            )

        height, width = image_shape[:2]
        size = height * width

        _, junctions = self._build_endpoints(skeleton=skeleton, image_shape=image_shape)
        ones = tf.ones_like(junctions, dtype=tf.float32)

        skeleton_no_junctions = tf.tensor_scatter_nd_update(tensor=skeleton,
                                                            indices=junctions[:, None],
                                                            updates=ones)

        junction_map = tf.tensor_scatter_nd_update(tensor=tf.zeros([size], dtype=tf.float32),
                                                   indices=junctions[:, None],
                                                   updates=ones)

        num_labels, junction_2d = self._connected_components(tf.reshape(junction_map, [height, width]))

        junction = tf.reshape(junction_2d, [-1])
        n_junctions = tf.maximum(num_labels - 1, 1)

        row_grid = tf.repeat(tf.range(height, dtype=tf.int32), width)
        col_grid = tf.tile(tf.range(width, dtype=tf.int32), [height])

        in_junction = junction > 0
        segment = tf.boolean_mask(junction, in_junction) - 1

        centroid_row = tf.cast(tf.round(tf.math.unsorted_segment_mean(
            tf.cast(tf.boolean_mask(row_grid, in_junction), tf.float32), segment, n_junctions)), tf.int32)

        centroid_col = tf.cast(tf.round(tf.math.unsorted_segment_mean(
            tf.cast(tf.boolean_mask(col_grid, in_junction), tf.float32), segment, n_junctions)), tf.int32)

        endpoints, _ = self._build_endpoints(skeleton=skeleton_no_junctions, image_shape=image_shape)
        e_row, e_col = endpoints // width, endpoints % width

        near_row = e_row[:, None] + self._offset_dy[None, :]
        near_col = e_col[:, None] + self._offset_dx[None, :]

        in_range = (near_row >= -1) & (near_row < height) & (near_col >= -1) & (near_col < width)
        near_idx = tf.math.floormod(near_row, height) * width + tf.math.floormod(near_col, width)

        neighbor_labels = tf.where(in_range, tf.gather(junction, near_idx), tf.zeros_like(near_row))
        present = tf.not_equal(neighbor_labels, 0)
        rank = tf.cumsum(tf.cast(present, tf.int32), axis=1)

        zeros = tf.zeros_like(neighbor_labels)
        count = tf.reduce_sum(tf.cast(present, tf.int32), axis=1)

        label_a = tf.reduce_sum(tf.where(present & tf.equal(rank, 1), neighbor_labels, zeros), axis=1)
        label_b = tf.reduce_sum(tf.where(present & tf.equal(rank, 2), neighbor_labels, zeros), axis=1)

        is_anchor, is_bridge = tf.equal(count, 1), tf.equal(count, 2)

        idx_a = tf.clip_by_value(label_a - 1, 0, n_junctions - 1)
        idx_b = tf.clip_by_value(label_b - 1, 0, n_junctions - 1)

        centroid_a_row, centroid_a_col = tf.gather(centroid_row, idx_a), tf.gather(centroid_col, idx_a)
        centroid_b_row, centroid_b_col = tf.gather(centroid_row, idx_b), tf.gather(centroid_col, idx_b)

        anchors = tf.boolean_mask(endpoints, is_anchor)

        anchor_mask = tf.tensor_scatter_nd_update(tensor=tf.zeros([size], dtype=tf.bool),
                                                  indices=anchors[:, None],
                                                  updates=tf.ones([tf.shape(anchors)[0]], dtype=tf.bool))

        anchor_row = tf.tensor_scatter_nd_update(tensor=tf.zeros([size], dtype=tf.int32),
                                                 indices=anchors[:, None],
                                                 updates=tf.boolean_mask(centroid_a_row, is_anchor))

        anchor_col = tf.tensor_scatter_nd_update(tensor=tf.zeros([size], dtype=tf.int32),
                                                 indices=anchors[:, None],
                                                 updates=tf.boolean_mask(centroid_a_col, is_anchor))

        anchor_row_sorted, anchor_col_sorted = by_col_then_row(idx=anchors)
        isolated = tf.boolean_mask(endpoints, ~(is_anchor | is_bridge))
        isolated_row_sorted, isolated_col_sorted = by_col_then_row(idx=isolated)

        start_rows = tf.concat([anchor_row_sorted, isolated_row_sorted], axis=0)
        start_cols = tf.concat([anchor_col_sorted, isolated_col_sorted], axis=0)

        all_rows = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True)
        all_cols = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True)
        path_id_of_point = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True)

        i0, pos0, path_id0 = tf.constant(0), tf.constant(0), tf.constant(0)

        loop_shapes = [
            tf.TensorShape([]), tf.TensorShape([size]), tf.TensorShape(None),
            tf.TensorShape(None), tf.TensorShape(None), tf.TensorShape([]), tf.TensorShape([]),
        ]

        _, working_skeleton, all_rows, all_cols, path_id_of_point, pos, path_id = tf.while_loop(
            cond=walk_cond,
            body=walk_body,
            loop_vars=[i0, skeleton_no_junctions, all_rows, all_cols, path_id_of_point, pos0, path_id0],
            shape_invariants=loop_shapes,
        )

        n_bridges = tf.reduce_sum(tf.cast(is_bridge, tf.int32))

        bridge_rows = tf.reshape(tf.stack(
            [tf.boolean_mask(centroid_a_row, is_bridge), tf.boolean_mask(centroid_b_row, is_bridge)], axis=1), [-1])

        bridge_cols = tf.reshape(tf.stack(
            [tf.boolean_mask(centroid_a_col, is_bridge), tf.boolean_mask(centroid_b_col, is_bridge)], axis=1), [-1])

        bridge_ids = tf.repeat(tf.range(n_bridges) + path_id, 2)
        span = pos + tf.range(2 * n_bridges)

        all_rows = all_rows.scatter(span, bridge_rows)
        all_cols = all_cols.scatter(span, bridge_cols)

        path_id_of_point = path_id_of_point.scatter(span, bridge_ids)
        pos, path_id = pos + 2 * n_bridges, path_id + n_bridges

        mask = tf.reshape(tf.cast((1.0 - working_skeleton) > 0.0, tf.float32), [height, width])
        num_loops, loop_labels_2d = self._connected_components(mask=mask)

        loop_labels = tf.reshape(loop_labels_2d, [-1])

        n_loops = num_loops - 1
        n_loops_safe = tf.maximum(n_loops, 1)

        in_loop = loop_labels > 0
        loop_segment = tf.boolean_mask(loop_labels, in_loop) - 1

        loop_key = tf.boolean_mask(col_grid * height + row_grid, in_loop)
        min_key = tf.math.unsorted_segment_min(loop_key, loop_segment, n_loops_safe)

        loop_start_row = (min_key % height)[:n_loops]
        loop_start_col = (min_key // height)[:n_loops]

        _, _, all_rows, all_cols, path_id_of_point, _, _ = tf.while_loop(
            cond=loop_cond,
            body=loop_body,
            loop_vars=[tf.constant(0), working_skeleton, all_rows, all_cols, path_id_of_point, pos, path_id],
            shape_invariants=loop_shapes,
        )

        all_rows, all_cols = all_rows.stack(), all_cols.stack()
        keep = ~gather_safe(flat=anchor_mask, rows=all_rows, cols=all_cols, default=False)

        return (
            tf.boolean_mask(all_rows, keep),
            tf.boolean_mask(all_cols, keep),
            tf.boolean_mask(path_id_of_point.stack(), keep),
        )

    def _convert_images_to_graphs(self, binary_image_tensor):
        """
        Extracts a graph for a single image via skeletonization.

        Parameters
        ----------
        binary_image_tensor : tf.Tensor
            Single binary image.

        Returns
        -------
        coordinates : tf.Tensor
            Pixel coordinates for the image.
        nodes : tf.Tensor
            Node features for the image.
        edges : tf.Tensor
            Edge pairs for the image.
        """

        def empty():
            return tf.zeros([0, 2], tf.int32), tf.zeros([0, 4], tf.float32), tf.zeros([0, 2], tf.int32)

        def nonempty():
            span = 2 * width
            key = (all_rows + height) * span + (all_cols + width)

            unique_key, point_id = tf.unique(key)
            row = unique_key // span - height
            col = unique_key % span - width

            order = tf.argsort(col * (2 * height) + (row + height))
            rank = tf.argsort(order)

            node_id = tf.gather(rank, point_id)
            num_nodes = tf.shape(unique_key)[0]

            x = tf.cast(tf.gather(col, order), tf.float32)
            y = tf.cast(tf.gather(row, order), tf.float32)

            x_min, y_min = tf.reduce_min(x), tf.reduce_min(y)
            scale = tf.maximum(tf.reduce_max(x) - x_min, tf.reduce_max(y) - y_min)
            scale = tf.where(tf.equal(scale, 0.0), 1.0, scale)

            coordinates = tf.cast(tf.stack([x, y], axis=1), tf.int32)

            nodes = tf.stack([
                (x - x_min) / scale, (y - y_min) / scale,
                tf.zeros(num_nodes),
                tf.zeros(num_nodes)
            ], axis=1)

            a, b = node_id[:-1], node_id[1:]
            linked = tf.equal(path_id_of_point[:-1], path_id_of_point[1:]) & tf.not_equal(a, b)

            lo = tf.minimum(tf.boolean_mask(a, linked), tf.boolean_mask(b, linked))
            hi = tf.maximum(tf.boolean_mask(a, linked), tf.boolean_mask(b, linked))

            adjacency = tf.tensor_scatter_nd_update(tensor=tf.zeros([num_nodes, num_nodes], dtype=tf.bool),
                                                    indices=tf.stack([lo, hi], axis=1),
                                                    updates=tf.ones_like(lo, dtype=tf.bool))

            pairs = tf.where(adjacency)
            lo, hi = tf.cast(pairs[:, 0], tf.int32), tf.cast(pairs[:, 1], tf.int32)
            edges = tf.stack([tf.concat([lo, hi], 0), tf.concat([hi, lo], 0)], axis=1)

            return coordinates, nodes, edges

        image_shape = binary_image_tensor.shape[:2]
        height, width = image_shape[:2]

        ink_mask = tf.cast(binary_image_tensor > 0, tf.float32)
        skeleton = tf.reshape(1.0 - self._thinning(ink_mask=ink_mask), [-1])

        all_rows, all_cols, path_id_of_point = self._convert_skeleton_to_paths(skeleton=skeleton,
                                                                               image_shape=image_shape)

        coordinates, nodes, edges = tf.cond(pred=tf.shape(all_rows)[0] > 0,
                                            true_fn=nonempty,
                                            false_fn=empty)

        coordinates.set_shape([None, 2])
        nodes.set_shape([None, 4])
        edges.set_shape([None, 2])

        return coordinates, nodes, edges

    def _graph_conv(self, nodes, edges, kernel, bias):
        """
        Applies graph convolution step with symmetric degree normalization.

        Parameters
        ----------
        nodes : tf.Tensor
            Node features.
        edges : tf.Tensor
            Edge index pairs.
        kernel : tf.Tensor
            Kernel applied before message passing.
        bias : tf.Variable, optional
            Bias added once after aggregation.

        Returns
        -------
        tf.Tensor
            Updated node features.
        """

        num_nodes = tf.shape(nodes)[1]
        batch_size = tf.shape(nodes)[0]

        src = tf.cast(edges[:, 0, :], tf.int32)
        dst = tf.cast(edges[:, 1, :], tf.int32)

        self_idx = tf.range(num_nodes, dtype=tf.int32)[tf.newaxis, :]
        self_idx = tf.tile(self_idx, [batch_size, 1])

        src = tf.concat([src, self_idx], axis=1)
        dst = tf.concat([dst, self_idx], axis=1)

        transformed = tf.linalg.matmul(nodes, kernel)

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
        outputs = tf.reshape(outputs_flat, [batch_size, num_nodes, -1])

        if self.use_bias:
            outputs = outputs + bias

        return outputs

    def call(self, inputs, training=False):
        """
        Forward pass of the layer.

        Parameters
        ----------
        inputs : tf.Tensor
            Batch of grayscale images.
        training : bool, optional
            Whether the call is for training or inference.

        Returns
        -------
        if filters is set
            tf.Tensor
                Latent space representation.
        else
            coordinates : tf.Tensor
                Zero-padded pixel coordinates.
            nodes : tf.Tensor
                Zero-padded node features.
            edges : tf.Tensor
                Zero-padded edge indices.
        """

        def cond(i, all_coordinates, all_nodes, all_edges, node_row_ids, edge_row_ids, node_pos, edge_pos):
            return i < batch_size

        def body(i, all_coordinates, all_nodes, all_edges, node_row_ids, edge_row_ids, node_pos, edge_pos):
            coords, nodes, edges = self._convert_images_to_graphs(binary_image_tensor=binary_images[i])

            n_nodes = tf.shape(nodes)[0]
            n_edges = tf.shape(edges)[0]

            node_span = node_pos + tf.range(n_nodes)
            edge_span = edge_pos + tf.range(n_edges)

            all_coordinates = all_coordinates.scatter(node_span, coords)
            all_nodes = all_nodes.scatter(node_span, nodes)
            all_edges = all_edges.scatter(edge_span, edges)
            node_row_ids = node_row_ids.scatter(node_span, tf.fill([n_nodes], i))
            edge_row_ids = edge_row_ids.scatter(edge_span, tf.fill([n_edges], i))

            return (
                i + 1,
                all_coordinates,
                all_nodes,
                all_edges,
                node_row_ids,
                edge_row_ids,
                node_pos + n_nodes,
                edge_pos + n_edges,
            )

        images = (inputs + 1.0) * 127.5

        mean = tf.nn.avg_pool2d(images, ksize=32, strides=1, padding='SAME')
        mean_sq = tf.nn.avg_pool2d(tf.square(images), ksize=32, strides=1, padding='SAME')

        stddev = tf.sqrt(tf.maximum(mean_sq - tf.square(mean), 0.0))
        threshold = mean * (1.0 + 0.1 * (stddev / 128.0 - 1.0))

        binary_images = tf.cast(tf.squeeze(images <= threshold, axis=-1), tf.uint8) * 255
        batch_size = tf.shape(binary_images)[0]

        all_coordinates = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True, element_shape=[2])
        all_nodes = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True, element_shape=[4])
        all_edges = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True, element_shape=[2])
        node_row_ids = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True)
        edge_row_ids = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True)

        _, all_coordinates, all_nodes, all_edges, node_row_ids, edge_row_ids, _, _ = tf.while_loop(
            cond=cond,
            body=body,
            loop_vars=[
                tf.constant(0),
                all_coordinates,
                all_nodes,
                all_edges,
                node_row_ids,
                edge_row_ids,
                tf.constant(0),
                tf.constant(0),
            ],
            shape_invariants=[
                tf.TensorShape([]),
                tf.TensorShape(None),
                tf.TensorShape(None),
                tf.TensorShape(None),
                tf.TensorShape(None),
                tf.TensorShape(None),
                tf.TensorShape([]),
                tf.TensorShape([]),
            ],
        )

        node_row_ids = node_row_ids.stack()
        edge_row_ids = edge_row_ids.stack()

        coordinates = tf.RaggedTensor.from_value_rowids(all_coordinates.stack(), node_row_ids, nrows=batch_size)
        nodes = tf.RaggedTensor.from_value_rowids(all_nodes.stack(), node_row_ids, nrows=batch_size)
        edges = tf.RaggedTensor.from_value_rowids(all_edges.stack(), edge_row_ids, nrows=batch_size)

        coordinates = coordinates.to_tensor(default_value=0)
        nodes = nodes.to_tensor(default_value=0.0)
        edges = tf.transpose(edges.to_tensor(default_value=0), perm=[0, 2, 1])

        coordinates = tf.reshape(coordinates, [batch_size, tf.shape(coordinates)[1], 2])
        nodes = tf.reshape(nodes, [batch_size, tf.shape(nodes)[1], 4])
        edges = tf.reshape(edges, [batch_size, 2, tf.shape(edges)[2]])

        coordinates = tf.stop_gradient(coordinates)
        nodes = tf.stop_gradient(nodes)
        edges = tf.stop_gradient(edges)

        if not self.filters:
            return coordinates, nodes, edges

        latent = tf.identity(nodes)

        for i, kernel in enumerate(self.kernels):
            bias = self.biases[i] if self.use_bias else None
            conv_out = self._graph_conv(nodes=latent, edges=edges, kernel=kernel, bias=bias)
            conv_out = self.activation(conv_out)

            if self.use_residual:
                skip_source = nodes if i == 0 else latent
                latent = conv_out + self.residual_dense_layers[i](skip_source)
            else:
                latent = tf.identity(conv_out)

        if training and self.dropout:
            latent = tf.nn.dropout(latent, rate=self.dropout)

        for i, projection_layer in enumerate(self.projection_layers):
            latent = projection_layer(latent)
            latent = self.activation(latent)

            if self.use_pooling:
                latent = self.pooling(latent)

            if self.use_gate:
                s_dense = self.gate_projection_layers[i](latent)
                gated = s_dense * tf.nn.sigmoid(tf.linalg.matmul(s_dense, self.gate_gammas[i]))
                latent = latent + gated * self.gate_betas[i]

        return self.global_pooling(latent)

    def compute_output_shape(self, input_shape):
        """
        Computes the output shape of the layer.

        Parameters
        ----------
        input_shape : tuple
            Shape of the input tensors.

        Returns
        -------
        tf.TensorShape or tuple of tf.TensorShape
            Output shape.
        """

        if not self.built:
            self.build(input_shape)

        if not self.filters:
            coordinates_shape = tf.TensorShape([input_shape[0], None, 2])
            nodes_shape = tf.TensorShape([input_shape[0], None, 4])
            edges_shape = tf.TensorShape([input_shape[0], 2, None])

            return coordinates_shape, nodes_shape, edges_shape

        output_units = self.units[-1] if self.units else self.filters[-1]
        output_shape = tf.TensorShape([input_shape[0], output_units])

        return output_shape


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
        variance = tf.math.reduce_variance(inputs, axis=1, keepdims=True)

        normed = (inputs - self.alpha * mean) / tf.sqrt(variance + self.epsilon)
        outputs = normed * self.gamma + self.beta

        return outputs


class GraphPooling(tf.keras.layers.Layer):
    """
    Graph pooling layer for aggregating node features into embeddings.

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


class OctaveConv2D(tf.keras.layers.Layer):
    """
    Implements octave convolutional layer.

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
            Additional keyword arguments.
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

        self.low_features = int(self.filters * self.alpha)
        self.high_features = self.filters - self.low_features

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
                                                   shape=(*self.kernel_size, high_in, self.high_features),
                                                   initializer=self.kernel_initializer,
                                                   regularizer=self.kernel_regularizer,
                                                   constraint=self.kernel_constraint)

        self.high_to_low_kernel = self.add_weight(name='high_to_low_kernel',
                                                  shape=(*self.kernel_size, high_in, self.low_features),
                                                  initializer=self.kernel_initializer,
                                                  regularizer=self.kernel_regularizer,
                                                  constraint=self.kernel_constraint)

        self.low_to_high_kernel = self.add_weight(name='low_to_high_kernel',
                                                  shape=(*self.kernel_size, low_in, self.high_features),
                                                  initializer=self.kernel_initializer,
                                                  regularizer=self.kernel_regularizer,
                                                  constraint=self.kernel_constraint)

        self.low_to_low_kernel = self.add_weight(name='low_to_low_kernel',
                                                 shape=(*self.kernel_size, low_in, self.low_features),
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

        high_shape = tf.TensorShape([*input_shape[:3], self.high_features])
        low_shape = tf.TensorShape([*input_shape[:3], self.low_features])

        return high_shape, low_shape


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


class ResidualGatedConv2D(tf.keras.layers.Layer):
    """
    Implements a Residual Gated Convolutional layer.

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


class ResidualGatedDense(tf.keras.layers.Layer):
    """
    Implements a Residual Gated Dense layer.

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
            g_dense = tf.nn.dropout(g_dense, rate=self.dropout)

        return inputs + g_dense * self.beta


class SpatialSelfAttention(tf.keras.layers.Layer):
    """
    Attention layer for capturing long-range dependencies in 1D or 2D spatial data.

    References
    ----------
    Attention Is All You Need
        https://arxiv.org/abs/1706.03762

    Self-Attention Generative Adversarial Networks
        https://arxiv.org/abs/1805.08318
    """

    def __init__(self,
                 k=1/8,
                 h=1/2,
                 kernel_initializer='glorot_uniform',
                 kernel_regularizer=None,
                 kernel_constraint=None,
                 beta_initializer='zeros',
                 dropout=0.0,
                 use_pooling=True,
                 use_bias=True,
                 **kwargs):
        """
        Initialize the attention layer.

        Parameters
        ----------
        k : float, optional
            Projection factor for query and key features.
        h : float, optional
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
            Dropout rate for attention weights.
        use_pooling : bool, optional
            Enables spatial reduction with max pooling.
        use_bias : bool, optional
            Enables bias terms in the projection layers.
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
        self.use_pooling = use_pooling
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
            'use_pooling': self.use_pooling,
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
        value_shape : TensorShape, optional
            Shape of the Value input tensor.
        key_shape : TensorShape, optional
            Shape of the Key input tensor.
        """

        super().build(query_shape)

        if value_shape is None:
            value_shape = query_shape

        if key_shape is None:
            key_shape = value_shape

        if len(query_shape) == 3:
            conv_layer = tf.keras.layers.Conv1D
            pooling_layer = tf.keras.layers.MaxPooling1D
            pool_size = strides = 2 if query_shape[-2] > 1 else 1

        elif len(query_shape) == 4:
            conv_layer = tf.keras.layers.Conv2D
            pooling_layer = tf.keras.layers.MaxPooling2D
            pool_size = strides = (2 if query_shape[-3] > 1 else 1,
                                   2 if query_shape[-2] > 1 else 1)

        else:
            raise ValueError("Unsupported input shape: must be 1D or 2D")

        self.filters = query_shape[-1]
        self.k_filters = int(self.filters * self.k)
        self.h_filters = int(self.filters * self.h)

        if self.use_pooling:
            self.k_pooling = pooling_layer(pool_size=pool_size, strides=strides)
            self.v_pooling = pooling_layer(pool_size=pool_size, strides=strides)

        self.k_conv = conv_layer(filters=self.k_filters,
                                 kernel_size=1,
                                 padding='valid',
                                 kernel_initializer=self.kernel_initializer,
                                 kernel_regularizer=self.kernel_regularizer,
                                 kernel_constraint=self.kernel_constraint,
                                 use_bias=self.use_bias)

        self.q_conv = conv_layer(filters=self.k_filters,
                                 kernel_size=1,
                                 padding='valid',
                                 kernel_initializer=self.kernel_initializer,
                                 kernel_regularizer=self.kernel_regularizer,
                                 kernel_constraint=self.kernel_constraint,
                                 use_bias=self.use_bias)

        self.v_conv = conv_layer(filters=self.h_filters,
                                 kernel_size=1,
                                 padding='valid',
                                 kernel_initializer=self.kernel_initializer,
                                 kernel_regularizer=self.kernel_regularizer,
                                 kernel_constraint=self.kernel_constraint,
                                 use_bias=self.use_bias)

        if self.h != 1:
            self.o_conv = conv_layer(filters=self.filters,
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
        k = self.k_conv(key)

        if self.use_pooling:
            k = self.k_pooling(k)

        k = tf.reshape(k, shape=[shape[0], -1, k.shape[-1]])

        q = self.q_conv(query)
        q = tf.reshape(q, shape=[shape[0], -1, q.shape[-1]])

        s = tf.matmul(q, k, transpose_b=True)
        s = tf.nn.softmax(s, axis=-1)

        if training and self.dropout:
            s = tf.nn.dropout(s, rate=self.dropout)
            s = tf.math.divide_no_nan(s, tf.reduce_sum(s, axis=-1, keepdims=True))

        v = self.v_conv(value)

        if self.use_pooling:
            v = self.v_pooling(v)

        v = tf.reshape(v, shape=[shape[0], -1, v.shape[-1]])

        o = tf.matmul(s, v)
        o = tf.reshape(o, shape=shape[:-1] + [self.h_filters])

        if self.h != 1:
            o = self.o_conv(o)

        return o * self.beta
