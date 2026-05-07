import tensorflow as tf


class EditDistance(tf.keras.metrics.Metric):
    """
    Edit distance for sequence comparison tasks.

    References
    ----------
    A Guided Tour to Approximate String Matching
        https://dl.acm.org/doi/10.1145/375360.375365

    Binary Codes Capable of Correcting Deletions, Insertions, and Reversals
        https://nymity.ch/sybilhunting/pdf/Levenshtein1966a.pdf
    """

    def __init__(self, beam_width=1, name='dist', **kwargs):
        """
        Initialize the EditDistance instance.

        Parameters
        ----------
        beam_width : int, optional
            The width of the beam for CTC beam search decoder.
        name : str, optional
            A name for the instance.
        **kwargs : dict
            Additional arguments.
        """

        super().__init__(name=name, **kwargs)

        self.beam_width = beam_width
        self.tracker = tf.keras.metrics.Mean()

    def update_state(self, y_true, y_pred):
        """
        Compute the edit distance between true and predicted sequences.

        Parameters
        ----------
        y_true : tf.Tensor
            Tensor of true labels.
        y_pred : tf.Tensor
            Tensor of predicted labels.
        """

        y_true = tf.reshape(y_true, shape=(tf.shape(y_true)[0], -1))
        y_pred = tf.reshape(y_pred, shape=(tf.shape(y_pred)[0], -1, tf.shape(y_pred)[-1]))

        labels = tf.sparse.from_dense(y_true)
        sequence_length = tf.fill([tf.shape(y_pred)[0]], value=tf.shape(y_pred)[1])

        decoded, _ = tf.nn.ctc_beam_search_decoder(inputs=tf.transpose(y_pred, perm=[1, 0, 2]),
                                                   sequence_length=sequence_length,
                                                   beam_width=self.beam_width,
                                                   top_paths=1)

        edit_distance = tf.edit_distance(hypothesis=decoded[0], truth=labels, normalize=True)
        edit_distance = tf.reduce_mean(edit_distance)

        self.tracker.update_state(edit_distance)

    def result(self):
        """
        Return the current value.

        Returns
        -------
        float
            The current value.
        """

        return self.tracker.result()

    def reset_state(self):
        """
        Reset the state of the tracker.
        """

        self.tracker.reset_state()


class FrechetInceptionDistance(tf.keras.metrics.Metric):
    """
    Fréchet Inception Distance (FID) for assessing image generation quality.

    References
    ----------
    GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium
        https://arxiv.org/abs/1706.08500
    """

    def __init__(self, image_shape, name='fid', **kwargs):
        """
        Initialize the FID metric instance.

        Parameters
        ----------
        image_shape : list or tuple
            Input image shape.
        name : str, optional
            A name for the instance.
        **kwargs : dict
            Additional arguments.
        """

        super().__init__(name=name, **kwargs)

        self.image_shape = image_shape
        self.inception_image_shape = (299, 299, 3)
        self.tracker = tf.keras.metrics.Mean()

        height_patches = max(1, round(self.image_shape[0] / self.inception_image_shape[0]))
        width_patches = max(1, round(self.image_shape[1] / self.inception_image_shape[1]))

        height = self.inception_image_shape[0] * height_patches
        width = self.inception_image_shape[1] * width_patches

        self.inception_encoder = tf.keras.Sequential([
            tf.keras.layers.InputLayer(shape=(None, None, 1)),
            tf.keras.layers.Resizing(height=height, width=width, interpolation='bilinear'),
            tf.keras.layers.Lambda(lambda x: tf.image.extract_patches(
                images=x,
                sizes=[1, *self.inception_image_shape[:2], 1],
                strides=[1, *self.inception_image_shape[:2], 1],
                rates=[1, 1, 1, 1],
                padding='VALID',
            )),
            tf.keras.layers.Lambda(lambda x: tf.reshape(x, shape=[-1, *self.inception_image_shape[:2], 1])),
            tf.keras.layers.Lambda(lambda x: tf.repeat(x, repeats=3, axis=-1)),
            tf.keras.applications.InceptionV3(
                include_top=False,
                weights='imagenet',
                pooling='avg',
                input_shape=self.inception_image_shape,
            ),
        ], name='inception_encoder')

    def update_state(self, y_true, y_pred):
        """
        Compute the FID between true and predicted images.

        Parameters
        ----------
        y_true : tf.Tensor
            Batch of real images.
        y_pred : tf.Tensor
            Batch of generated images.
        """

        real_features = self.inception_encoder(y_true, training=False)
        generated_features = self.inception_encoder(y_pred, training=False)

        mean_real = tf.reduce_mean(real_features, axis=0)
        mean_generated = tf.reduce_mean(generated_features, axis=0)

        mean_diff = mean_real - mean_generated
        mean_term = tf.reduce_sum(tf.square(mean_diff))

        centered_real = real_features - mean_real
        centered_generated = generated_features - mean_generated

        batch_size = tf.cast(tf.shape(real_features)[0], dtype=tf.float32)
        denom = tf.maximum(batch_size - 1.0, 1.0)

        cov_term = (tf.reduce_sum(tf.square(centered_real)) +
                    tf.reduce_sum(tf.square(centered_generated))) / denom

        cov_sqrt_term = 2.0 * self.matrix_sqrt_trace(centered_real, centered_generated, denom)

        fid = mean_term + cov_term - cov_sqrt_term

        self.tracker.update_state(fid)

    @tf.function
    def matrix_sqrt_trace(self, centered_a, centered_b, denom):
        """
        Compute trace of the matrix square root using eigendecomposition.

        Parameters
        ----------
        centered_a : tf.Tensor
            Centered features from distribution A.
        centered_b : tf.Tensor
            Centered features from distribution B.
        denom : tf.Tensor
            Scalar normalization factor.

        Returns
        -------
        tf.Tensor
            Trace of the matrix square root of the covariance product.
        """

        m_ab = tf.matmul(centered_a, centered_b, transpose_b=True)
        m_ba = tf.matmul(centered_b, centered_a, transpose_b=True)
        product = tf.matmul(m_ab, m_ba) / (denom * denom)

        eigenvalues = tf.linalg.eigvals(product)
        eigenvalues = tf.math.real(eigenvalues)
        eigenvalues = tf.maximum(eigenvalues, 0.0)

        return tf.reduce_sum(tf.sqrt(tf.sqrt(eigenvalues)))

    def result(self):
        """
        Return the current value.

        Returns
        -------
        float
            The current value.
        """

        return self.tracker.result()

    def reset_state(self):
        """
        Reset the state of the tracker.
        """

        self.tracker.reset_state()


class KernelInceptionDistance(tf.keras.metrics.Metric):
    """
    Kernel Inception Distance (KID) for assessing image generation quality.

    References
    ----------
    Demystifying MMD GANs
        https://arxiv.org/abs/1801.01401
    """

    def __init__(self, image_shape, name='kid', **kwargs):
        """
        Initialize the KID metric instance.

        Parameters
        ----------
        image_shape : list or tuple
            Input image shape.
        name : str, optional
            A name for the instance.
        **kwargs : dict
            Additional arguments.
        """

        super().__init__(name=name, **kwargs)

        self.image_shape = image_shape
        self.inception_image_shape = (299, 299, 3)
        self.tracker = tf.keras.metrics.Mean()

        height_patches = max(1, round(self.image_shape[0] / self.inception_image_shape[0]))
        width_patches = max(1, round(self.image_shape[1] / self.inception_image_shape[1]))

        height = self.inception_image_shape[0] * height_patches
        width = self.inception_image_shape[1] * width_patches

        self.inception_encoder = tf.keras.Sequential([
            tf.keras.layers.InputLayer(shape=(None, None, 1)),
            tf.keras.layers.Resizing(height=height, width=width, interpolation='bilinear'),
            tf.keras.layers.Lambda(lambda x: tf.image.extract_patches(
                images=x,
                sizes=[1, *self.inception_image_shape[:2], 1],
                strides=[1, *self.inception_image_shape[:2], 1],
                rates=[1, 1, 1, 1],
                padding='VALID',
            )),
            tf.keras.layers.Lambda(lambda x: tf.reshape(x, shape=[-1, *self.inception_image_shape[:2], 1])),
            tf.keras.layers.Lambda(lambda x: tf.repeat(x, repeats=3, axis=-1)),
            tf.keras.applications.InceptionV3(
                include_top=False,
                weights='imagenet',
                pooling='avg',
                input_shape=self.inception_image_shape,
            ),
        ], name='inception_encoder')

    def update_state(self, y_true, y_pred):
        """
        Compute the KID between true and predicted images.

        Parameters
        ----------
        y_true : tf.Tensor
            Batch of real images.
        y_pred : tf.Tensor
            Batch of generated images.
        """

        real_features = self.inception_encoder(y_true, training=False)
        generated_features = self.inception_encoder(y_pred, training=False)

        kernel_real = self.polynomial_kernel(real_features, real_features)
        kernel_generated = self.polynomial_kernel(generated_features, generated_features)
        kernel_cross = self.polynomial_kernel(real_features, generated_features)

        batch_size = tf.cast(tf.shape(real_features)[0], dtype=tf.int32)
        batch_factor = tf.cast(batch_size * (batch_size - 1), dtype=tf.float32)

        sum_kernel_real = tf.reduce_sum(kernel_real * (1.0 - tf.eye(batch_size)))
        mean_kernel_real = tf.math.divide_no_nan(sum_kernel_real, batch_factor)

        sum_kernel_generated = tf.reduce_sum(kernel_generated * (1.0 - tf.eye(batch_size)))
        mean_kernel_generated = tf.math.divide_no_nan(sum_kernel_generated, batch_factor)

        kid = mean_kernel_real + mean_kernel_generated - 2.0 * tf.reduce_mean(kernel_cross)

        self.tracker.update_state(kid)

    @tf.function
    def polynomial_kernel(self, features_1, features_2):
        """
        Compute the polynomial kernel between two feature sets.

        Parameters
        ----------
        features_1 : tf.Tensor
            First set of features.
        features_2 : tf.Tensor
            Second set of features.

        Returns
        -------
        tf.Tensor
            Polynomial kernel between the two feature sets.
        """

        feature_dimensions = tf.cast(tf.shape(features_1)[1], dtype=tf.float32)
        features = features_1 @ tf.transpose(features_2)

        return (tf.math.divide_no_nan(features, feature_dimensions) + 1.0) ** 3.0

    def result(self):
        """
        Return the current value.

        Returns
        -------
        float
            The current value.
        """

        return self.tracker.result()

    def reset_state(self):
        """
        Reset the state of the tracker.
        """

        self.tracker.reset_state()
