import cv2
import numpy as np
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

        return tf.TensorShape(input_shape[-1][:3] + input_shape[0][-1:])


class GraphConv(tf.keras.layers.Layer):
    """
    Graph convolution layer over explicit node and edge data.

    References
    ----------
    Semi-Supervised Classification with Graph Convolutional Networks
        https://arxiv.org/abs/1609.02907
    """

    def __init__(self,
                 units,
                 kernel_initializer='glorot_uniform',
                 use_bias=True,
                 **kwargs):
        """
        Initializes the layer.

        Parameters
        ----------
        units : int
            Number of output features per node.
        kernel_initializer : str or initializer, optional
            Initializer for the kernel weights.
        use_bias : bool, optional
            Whether to add bias terms.
        **kwargs : dict
            Additional keyword arguments.
        """

        super().__init__(**kwargs)

        self.units = units
        self.kernel_initializer = kernel_initializer
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

        self.num_nodes = input_shape[0][1]

        self.kernel = self.add_weight(name=f'{self.name}_kernel',
                                      shape=(input_shape[0][2], self.units),
                                      initializer=self.kernel_initializer,
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

        src = tf.cast(edges[:, 0, :], tf.int32)
        dst = tf.cast(edges[:, 1, :], tf.int32)

        self_idx = tf.expand_dims(tf.range(self.num_nodes, dtype=tf.int32), axis=0)
        self_idx = tf.tile(self_idx, [tf.shape(nodes)[0], 1])

        src = tf.concat([src, self_idx], axis=1)
        dst = tf.concat([dst, self_idx], axis=1)

        transformed = tf.linalg.matmul(nodes, self.kernel)
        ones = tf.ones(tf.shape(src), dtype=transformed.dtype)

        degrees = tf.map_fn(
            fn=lambda x: tf.math.unsorted_segment_sum(x[0], x[1], self.num_nodes),
            elems=(ones, dst),
            fn_output_signature=tf.float32)

        deg_inv_sqrt = tf.math.rsqrt(tf.maximum(degrees, 1.0))

        src_norm = tf.gather(deg_inv_sqrt, src, batch_dims=1)
        dst_norm = tf.gather(deg_inv_sqrt, dst, batch_dims=1)
        norm = src_norm * dst_norm

        messages = tf.gather(transformed, src, batch_dims=1)
        messages = messages * tf.expand_dims(norm, axis=-1)

        outputs = tf.map_fn(
            fn=lambda x: tf.math.unsorted_segment_sum(x[0], x[1], self.num_nodes),
            elems=(messages, dst),
            fn_output_signature=tf.float32)

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

        return tf.TensorShape([input_shape[0][0], self.num_nodes, self.units])


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
        kernel_initializer : str or initializer, optional
            Initializer for the kernel weights.
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
        self.num_nodes = self.height * self.width

        self.kernel = self.add_weight(name=f'{self.name}_kernel',
                                      shape=(input_shape[3], self.units),
                                      initializer=self.kernel_initializer,
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

    def _build_edge_index(self):
        """
        Builds the graph edges using k-neighbors.

        Returns
        -------
        tf.Tensor
            Edge tensor, formatted as [dst, src].
        """

        base_coords = self._build_base_coordinates()
        offsets = self._build_positional_offsets()

        warped_coords = tf.reshape(base_coords + offsets, [self.num_nodes, 2])

        diff = warped_coords[:, None, :] - warped_coords[None, :, :]
        dist = tf.reduce_sum(tf.square(diff), axis=-1)
        dist = dist + tf.eye(self.num_nodes, dtype=dist.dtype) * 1e12

        knn = tf.math.top_k(-dist, k=self.k_neighbors).indices
        dst = tf.repeat(tf.range(self.num_nodes), repeats=self.k_neighbors)
        src = tf.reshape(knn, [-1])

        edges = tf.stack([dst, src], axis=-1)
        self_loops = tf.stack([tf.range(self.num_nodes), tf.range(self.num_nodes)], axis=-1)

        return tf.concat([edges, self_loops], axis=0)

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

        nodes = tf.reshape(inputs, [batch_size, self.num_nodes, -1])
        edge_index = self._build_edge_index()

        dst = edge_index[:, 0]
        src = edge_index[:, 1]

        neighbor_features = tf.linalg.matmul(tf.gather(nodes, src, axis=1), self.kernel)

        outputs = tf.map_fn(
            fn=lambda x: tf.math.unsorted_segment_sum(x, dst, self.num_nodes),
            elems=neighbor_features,
            fn_output_signature=tf.float32)

        if self.use_bias:
            outputs = outputs + self.bias

        return tf.reshape(outputs, [batch_size, self.height, self.width, self.units])

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

        return tf.TensorShape([input_shape[0], self.height, self.width, self.units])


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

    def __init__(self, step_length=5, **kwargs):
        """
        Initializes the layer.

        Parameters
        ----------
        step_length : int, optional
            Arc-length distance between keypoint pixels.
        **kwargs : dict
            Additional keyword arguments.
        """

        super().__init__(**kwargs)

        self.step_length = step_length

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
            'step_length': self.step_length,
        })

        return config

    def _build_endpoints(self, skeleton):
        """
        Detects stroke endpoints and junction points in a skeleton image.

        Parameters
        ----------
        skeleton : np.ndarray
            Binary skeleton image.

        Returns
        -------
        endpoints : list
            Stroke endpoint positions.
        junctions : list
            Deduplicated junction positions.
        possible_junctions : list
            Raw junction candidates before deduplication.
        """

        mask = (1 - skeleton).astype(np.float32)
        kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.float32)

        weighted_count = mask * cv2.filter2D(src=mask,
                                             ddepth=-1,
                                             kernel=kernel,
                                             borderType=cv2.BORDER_CONSTANT)

        rows_j, cols_j = np.nonzero(weighted_count > 2)
        possible_junctions = list(zip(rows_j.tolist(), cols_j.tolist()))

        rows_e, cols_e = np.nonzero(weighted_count == 1)
        endpoints = list(zip(rows_e.tolist(), cols_e.tolist()))

        junctions = []

        for point in possible_junctions:
            has_stronger_neighbor = False

            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if not (dy or dx):
                        continue

                    neighbor = (point[0] + dy, point[1] + dx)

                    if neighbor in junctions:
                        if weighted_count[neighbor] <= weighted_count[point]:
                            junctions.remove(neighbor)
                        else:
                            has_stronger_neighbor = True

            if not has_stronger_neighbor:
                junctions.append(point)

        return endpoints, junctions, possible_junctions

    def _build_paths(self, skeleton, start_point, step_length):
        """
        Extracts evenly spaced keypoints along the skeleton.

        Parameters
        ----------
        skeleton : np.ndarray
            Binary skeleton image.
        start_point : tuple
            Starting pixel position.
        step_length : float
            Arc-length distance between keypoints.

        Returns
        -------
        keypoints : list
            Sampled positions along the path.
        skeleton : np.ndarray
            Updated skeleton with visited pixels consumed.
        """

        keypoints = [start_point]
        accumulated_distance = 0.0
        current_point = start_point
        skeleton[current_point] = 1

        neighbor_offsets = [
            ((-1, -1), np.sqrt(2)), ((-1, 0), 1.0), ((-1, 1), np.sqrt(2)),
            ((0,   1), 1.0),        ((1,  1), np.sqrt(2)), ((1,  0), 1.0),
            ((1,  -1), np.sqrt(2)), ((0, -1), 1.0),
        ]

        while True:
            next_point = None
            step_distance = 0.0

            for (dy, dx), dist in neighbor_offsets:
                neighbor = (current_point[0] + dy, current_point[1] + dx)

                try:
                    if skeleton[neighbor] == 0:
                        next_point = neighbor
                        step_distance = dist
                        break
                except IndexError:
                    pass

            if next_point is None:
                break

            accumulated_distance += step_distance
            skeleton[next_point] = 1
            current_point = next_point

            if accumulated_distance >= step_length:
                keypoints.append(current_point)
                accumulated_distance = 0.0

        if current_point not in keypoints:
            keypoints.append(current_point)

        return keypoints, skeleton

    def _convert_skeleton_to_paths(self, skeleton, step_length):
        """
        Converts skeletons to ordered paths, handling complex topologies.

        Parameters
        ----------
        skeleton : np.ndarray
            Binary skeleton image.
        step_length : float
            Arc-length distance between keypoints.

        Returns
        -------
        list of list of tuple
            Keypoint sequences, one per stroke segment.
        """

        _, _, possible_junctions = self._build_endpoints(skeleton)

        skeleton_no_junctions = np.copy(skeleton)
        junction_pixel_map = np.zeros(skeleton.shape, dtype=np.uint8)

        for junction in possible_junctions:
            skeleton_no_junctions[junction] = 1
            junction_pixel_map[junction] = 1

        num_labels, junction_labels = cv2.connectedComponents(junction_pixel_map, connectivity=8)

        junction_groups = [[] for _ in range(num_labels - 1)]
        rows_lbl, cols_lbl = np.nonzero(junction_labels)

        for row, col in zip(rows_lbl.tolist(), cols_lbl.tolist()):
            junction_groups[junction_labels[row, col] - 1].append((row, col))

        centroids = [tuple(int(round(sum(y) / len(y))) for y in zip(*x)) for x in junction_groups]
        endpoints_on_arc, _, _ = self._build_endpoints(skeleton_no_junctions)

        nearest_junction = {}
        ignore_endpoints = []
        junction_bridge_paths = []
        endpoints_isolated = []

        offset = [(-1, -1), (-1, 0), (-1, 1), (0, 1),
                  (1,   1),  (1, 0), (1,  -1), (0, -1)]

        for point in endpoints_on_arc:
            neighbor_labels = []

            for dy, dx in offset:
                neighbor = (point[0] + dy, point[1] + dx)

                try:
                    lv = junction_labels[neighbor]
                    if lv:
                        neighbor_labels.append(lv)
                except IndexError:
                    pass

            if len(neighbor_labels) == 1:
                nearest_junction[point] = centroids[neighbor_labels[0] - 1]

            elif len(neighbor_labels) == 2:
                ignore_endpoints.append(point)
                junction_bridge_paths.append([centroids[neighbor_labels[0] - 1],
                                              centroids[neighbor_labels[1] - 1]])
            else:
                endpoints_isolated.append(point)

        endpoints_near_junction = sorted(nearest_junction.keys(), key=lambda x: (x[1], x[0]))

        endpoints_isolated = sorted(endpoints_isolated, key=lambda x: (x[1], x[0]))
        ignore_endpoints_set = set(ignore_endpoints)
        endpoints_near_junction_set = set(endpoints_near_junction)

        working_skeleton = np.copy(skeleton_no_junctions)
        paths = []

        for endpoint in endpoints_near_junction + endpoints_isolated:
            if endpoint in ignore_endpoints_set:
                working_skeleton[endpoint] = 1
            else:
                path, working_skeleton = self._build_paths(working_skeleton, endpoint, step_length)

                if path[0] in nearest_junction:
                    path.insert(0, nearest_junction[path[0]])

                if path[-1] in nearest_junction:
                    path.append(nearest_junction[path[-1]])

                paths.append(path)

        paths.extend(junction_bridge_paths)

        pixel_map = np.uint8((1 - working_skeleton) > 0)
        num_labels, loop_labels = cv2.connectedComponents(pixel_map, connectivity=8)

        circle_start_points = [None] * (num_labels - 1)
        rows_loop, cols_loop = np.nonzero(loop_labels)

        for row, col in zip(rows_loop.tolist(), cols_loop.tolist()):
            lv = loop_labels[row, col]
            cur = circle_start_points[lv - 1]

            if cur is None or col < cur[1] or (col == cur[1] and row < cur[0]):
                circle_start_points[lv - 1] = (row, col)

        for circle_start in circle_start_points:
            path, working_skeleton = self._build_paths(working_skeleton, circle_start, step_length)
            paths.append(path[:-1] + path[0:1])

        return [[x for x in path if x not in endpoints_near_junction_set] for path in paths]

    def _convert_images_to_graphs(self, binary_images_tensor):
        """
        Extracts graphs for a full batch and returns zero-padded arrays.

        Parameters
        ----------
        binary_images_tensor : tf.Tensor
            Batch of binary images.

        Returns
        -------
        coordinates : tf.RaggedTensor
            Pixel coordinates per image.
        nodes : tf.RaggedTensor
            Node features per image.
        edges : tf.RaggedTensor
            Edge pairs per image.
        """

        def fn(image):
            thinned = cv2.ximgproc.thinning(image.numpy())
            skeleton = np.where(thinned == 255, 0.0, 1.0).astype(np.float32)
            paths = self._convert_skeleton_to_paths(skeleton.copy(), self.step_length)

            if not any(paths):
                return np.zeros((0, 4), dtype=np.float32), np.zeros((0, 2), dtype=np.int32)

            unique_points = sorted({point for p in paths for point in p}, key=lambda x: (x[1], x[0]))
            node_id_of = {x: idx for idx, x in enumerate(unique_points)}
            num_nodes = len(unique_points)

            coords = np.array(unique_points, dtype=np.float32)
            y_pixels, x_pixels = coords[:, 0], coords[:, 1]
            scale = max(x_pixels.max() - x_pixels.min(), y_pixels.max() - y_pixels.min()) or 1.0

            nodes = np.stack([
                (x_pixels - x_pixels.min()) / scale,
                (y_pixels - y_pixels.min()) / scale,
                np.zeros(num_nodes, dtype=np.float32),
                np.zeros(num_nodes, dtype=np.float32),
            ], axis=1)

            coordinates = np.stack([x_pixels, y_pixels], axis=1).astype(np.int32)
            edge_pairs = set()

            for path in paths:
                for pa, pb in zip(path, path[1:]):
                    a, b = node_id_of[pa], node_id_of[pb]

                    if a != b:
                        edge_pairs.add((min(a, b), max(a, b)))

            if edge_pairs:
                edge_src = np.array([e[0] for e in edge_pairs], dtype=np.int32)
                edge_dst = np.array([e[1] for e in edge_pairs], dtype=np.int32)

                edges = np.stack([np.concatenate([edge_src, edge_dst]),
                                  np.concatenate([edge_dst, edge_src])], axis=0).T
            else:
                edges = np.zeros((0, 2), dtype=np.int32)

            return coordinates, nodes, edges

        coordinates, nodes, edges = tf.py_function(func=fn,
                                                   inp=[binary_images_tensor],
                                                   Tout=[tf.int32, tf.float32, tf.int32])

        coordinates.set_shape([None, 2])
        nodes.set_shape([None, 4])
        edges.set_shape([None, 2])

        coordinates = tf.RaggedTensor.from_uniform_row_length(tf.reshape(coordinates, [-1]), 2)
        nodes = tf.RaggedTensor.from_uniform_row_length(tf.reshape(nodes, [-1]), 4)
        edges = tf.RaggedTensor.from_uniform_row_length(tf.reshape(edges, [-1]), 2)

        return coordinates, nodes, edges

    def call(self, inputs):
        """
        Forward pass of the layer.

        Parameters
        ----------
        inputs : tf.Tensor
            Batch of grayscale images.

        Returns
        -------
        nodes : tf.Tensor
            Zero-padded node features.
        edges : tf.Tensor
            Zero-padded edge indices.
        """

        images = (inputs + 1.0) * 127.5

        mean = tf.nn.avg_pool2d(images, ksize=32, strides=1, padding='SAME')
        mean_sq = tf.nn.avg_pool2d(tf.square(images), ksize=32, strides=1, padding='SAME')

        stddev = tf.sqrt(tf.maximum(mean_sq - tf.square(mean), 0.0))
        threshold = mean * (1.0 + 0.1 * (stddev / 128.0 - 1.0))

        binary_images = tf.cast(tf.squeeze(images <= threshold, axis=-1), tf.uint8) * 255

        coordinates, nodes, edges = tf.map_fn(
            fn=self._convert_images_to_graphs,
            elems=binary_images,
            fn_output_signature=(
                tf.RaggedTensorSpec(shape=[None, 2], dtype=tf.int32, ragged_rank=1),
                tf.RaggedTensorSpec(shape=[None, 4], dtype=tf.float32, ragged_rank=1),
                tf.RaggedTensorSpec(shape=[None, 2], dtype=tf.int32, ragged_rank=1),
            )
        )

        self.coordinates = tf.stop_gradient(coordinates.to_tensor(default_value=0))
        nodes = tf.stop_gradient(nodes.to_tensor(default_value=0.0))
        edges = tf.stop_gradient(tf.transpose(edges.to_tensor(default_value=0), perm=[0, 2, 1]))

        return nodes, edges

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

        nodes_shape = tf.TensorShape([input_shape[0], None, 4])
        edges_shape = tf.TensorShape([input_shape[0], 2, None])

        return nodes_shape, edges_shape


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

        return pos_emb


class PositionEmbedding2D(tf.keras.layers.Layer):
    """
    2D Positional Embedding layer.
    Adds learnable row and column embeddings to 2D input tensors.
    Typically used for image-like inputs (batch_size, height, width, features).

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
            Tensor of shape (batch_size, height, width, features)

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
    Implements a Residual Gated Dense (fully-connected) layer.

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
                                     shape=(input_shape[-1],),
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
        g_dense = s_dense * tf.nn.sigmoid(s_dense * self.gamma)

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
