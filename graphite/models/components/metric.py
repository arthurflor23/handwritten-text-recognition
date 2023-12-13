import tensorflow as tf


class KID(tf.keras.metrics.Metric):
    """
    Kernel Inception Distance (KID) metric class.

    Kernel Inception Distance (KID) was proposed as a replacement for the popular
        Frechet Inception Distance (FID) metric for measuring image generation quality.
    Both metrics measure the difference in the generated and training distributions
        in the representation space of an InceptionV3 network pretrained on ImageNet.

    According to the paper, KID was proposed because FID has no unbiased estimator, its
        expected value is higher when it is measured on fewer images. KID is more suitable for
        small datasets because its expected value does not depend on the number of samples it is
        measured on. It is also computationally lighter, numerically more stable, and simpler to
        implement because it can be estimated in a per-batch manner.

    References
    ----------
    Demystifying MMD GANs
        https://arxiv.org/abs/1801.01401

    Rethinking the Inception Architecture for Computer Vision
        https://arxiv.org/abs/1512.00567

    InceptionV3
        https://keras.io/api/applications/inceptionv3/

    ImageNet
        https://www.tensorflow.org/datasets/catalog/imagenet2012
    """

    def __init__(self, name='kernel_inception_distance', **kwargs):
        """
        Initialize the KID metric instance.

        Parameters
        ----------
        name : str, optional
            Name of the metric instance.
        **kwargs : dict
            Additional keyword arguments.
        """

        super().__init__(name=name, **kwargs)

        self.tracker = tf.keras.metrics.Mean()
        self.kid_image_size = (299, 299, 3)

        self.encoder = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(None, None, 1)),
            tf.keras.layers.Lambda(lambda x: tf.tile(x, [1, 1, 1, 3])),
            tf.keras.layers.Rescaling(255.0),
            tf.keras.layers.Resizing(height=self.kid_image_size[0], width=self.kid_image_size[1]),
            tf.keras.layers.Lambda(tf.keras.applications.inception_v3.preprocess_input),
            tf.keras.applications.InceptionV3(include_top=False, input_shape=self.kid_image_size, weights='imagenet'),
            tf.keras.layers.GlobalAveragePooling2D(),
        ], name='inception_encoder')

    def polynomial_kernel(self, features_1, features_2):
        """
        Compute the polynomial kernel between two sets of features.

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
        return (features_1 @ tf.transpose(features_2) / feature_dimensions + 1.0) ** 3.0

    def update_state(self, y_true, y_pred, sample_weight=None):
        """
        Update the metric state with new data.

        Parameters
        ----------
        y_true : tf.Tensor
            Batch of real images.
        y_pred : tf.Tensor
            Batch of generated images.
        sample_weight : tf.Tensor, optional
            Sample weights.
        """

        real_features = self.encoder(y_true, training=False)
        generated_features = self.encoder(y_pred, training=False)

        kernel_real = self.polynomial_kernel(real_features, real_features)
        kernel_generated = self.polynomial_kernel(generated_features, generated_features)
        kernel_cross = self.polynomial_kernel(real_features, generated_features)

        batch_size = tf.cast(tf.shape(real_features)[0], dtype=tf.float32)

        sum_kernel_real = tf.reduce_sum(kernel_real * (1.0 - tf.eye(batch_size)))
        mean_kernel_real = sum_kernel_real / (batch_size * (batch_size - 1.0))

        sum_kernel_generated = tf.reduce_sum(kernel_generated * (1.0 - tf.eye(batch_size)))
        mean_kernel_generated = sum_kernel_generated / (batch_size * (batch_size - 1.0))
        mean_kernel_cross = tf.reduce_mean(kernel_cross)

        value = mean_kernel_real + mean_kernel_generated - 2.0 * mean_kernel_cross

        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, dtype=tf.float32)
            sample_weight /= tf.reduce_sum(sample_weight) / batch_size
            value = tf.reduce_sum(value * sample_weight)

        self.tracker.update_state(value)

    def result(self):
        """
        Return the current result of the metric.

        Returns
        -------
        float
            The current value.
        """

        return self.tracker.result()

    def reset_state(self):
        """
        Reset the state of the metric.
        """

        self.tracker.reset_state()


class EditDistance(tf.keras.metrics.Metric):
    """
    Metric that calculates the normalized edit distance between sequences.

    References
    ----------
    Binary Codes Capable of Correcting Deletions, Insertions and Reversals
        https://mi.mathnet.ru/dan31411
    """

    def __init__(self, beam_width=10, name='edit_distance', **kwargs):
        """
        Initialize the EditDistance metric instance.

        Parameters
        ----------
        beam_width : int, optional
            The width of the beam for CTC beam search decoder.
        name : str, optional
            The name of the metric.
        **kwargs : dict
            Additional keyword arguments.
        """

        super().__init__(name=name, **kwargs)

        self.beam_width = beam_width
        self.tracker = tf.keras.metrics.Mean()

    def update_state(self, y_true, y_pred, sample_weight=None):
        """
        Update the metric state with new data.

        Parameters
        ----------
        y_true : tf.Tensor
            Ground truth label sequences.
        y_pred : tf.Tensor
            Predicted sequences, typically the logits from a model.
        sample_weight : tf.Tensor, optional
            Sample weights.
        """

        y_true = tf.reshape(y_true, [tf.shape(y_true)[0], -1])
        y_pred = tf.reshape(y_pred, [tf.shape(y_pred)[0], -1, tf.shape(y_pred)[-1]])

        inputs = tf.transpose(y_pred, [1, 0, 2])
        sequence_length = tf.fill([tf.shape(y_pred)[0]], tf.shape(y_pred)[1])

        decoded, _ = tf.nn.ctc_beam_search_decoder(inputs=inputs,
                                                   sequence_length=sequence_length,
                                                   beam_width=self.beam_width,
                                                   top_paths=1)

        y_true_indices = tf.where(y_true != 0)
        y_true_values = tf.gather_nd(y_true, y_true_indices)
        y_true_shape = tf.shape(y_true, out_type=tf.int64)
        y_true_sparse = tf.SparseTensor(y_true_indices, y_true_values, y_true_shape)

        edit_distance = tf.edit_distance(decoded[0], y_true_sparse, normalize=True)
        value = tf.reduce_mean(edit_distance)

        if sample_weight is not None:
            batch_size = tf.cast(tf.shape(y_pred)[0], dtype=tf.float32)

            sample_weight = tf.cast(sample_weight, dtype=tf.float32)
            sample_weight /= tf.reduce_sum(sample_weight) / batch_size
            value = tf.reduce_sum(value * sample_weight)

        self.tracker.update_state(value)

    def result(self):
        """
        Return the current result of the metric.

        Returns
        -------
        float
            The current value.
        """

        return self.tracker.result()

    def reset_state(self):
        """
        Reset the state of the metric.
        """

        self.tracker.reset_state()
