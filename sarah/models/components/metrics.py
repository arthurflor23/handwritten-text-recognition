import tensorflow as tf


class EditDistance(tf.keras.metrics.Metric):
    """
    Metric that calculates the normalized edit distance between sequences.

    References
    ----------
    A Guided Tour to Approximate String Matching
        https://dl.acm.org/doi/10.1145/375360.375365

    A Novel Connectionist System for Unconstrained Handwriting Recognition
        https://ieeexplore.ieee.org/document/4531750

    Character-Level Incremental Speech Recognition with Recurrent Neural Networks
        https://arxiv.org/abs/1601.06581
    """

    def __init__(self, beam_width=1, name='dist', **kwargs):
        """
        Initialize the EditDistance metric instance.

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

        self.tracker = tf.keras.metrics.Mean()

        self.beam_width = beam_width

    def update_state(self, y_true, y_pred):
        """
        Update the metric state with new data.

        Parameters
        ----------
        y_true : tf.Tensor
            Tensor of true labels.
        y_pred : tf.Tensor
            Tensor of predicted labels.
        """

        y_true = tf.reshape(y_true, (tf.shape(y_true)[0], -1))
        y_pred = tf.reshape(y_pred, (tf.shape(y_pred)[0], -1, tf.shape(y_pred)[-1]))

        labels = tf.sparse.from_dense(y_true)
        logits = tf.transpose(tf.math.log(y_pred + 1e-8), perm=[1, 0, 2])

        sequence_length = tf.fill([tf.shape(y_pred)[0]], tf.shape(y_pred)[1])

        decoded, _ = tf.nn.ctc_beam_search_decoder(inputs=tf.cast(logits, dtype=tf.float32),
                                                   sequence_length=tf.cast(sequence_length, dtype=tf.int32),
                                                   beam_width=self.beam_width,
                                                   top_paths=1)

        edit_distance = tf.edit_distance(hypothesis=decoded[0], truth=labels, normalize=True)
        edit_distance = tf.reduce_mean(edit_distance)

        self.tracker.update_state(edit_distance)

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


class KernelInceptionDistance(tf.keras.metrics.Metric):
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

    def __init__(self, scale=1.0, offset=0.0, name='kid', **kwargs):
        """
        Initialize the KID metric instance.

        Parameters
        ----------
        scale : float, optional
            Scaling factor for preprocessing.
        offset : float, optional
            Offset value for preprocessing.
        name : str, optional
            A name for the instance.
        **kwargs : dict
            Additional arguments.
        """

        super().__init__(name=name, **kwargs)

        self.tracker = tf.keras.metrics.Mean()

        self.scale = scale
        self.offset = offset
        self.kid_image_size = (299, 299, 3)

        self.encoder = tf.keras.Sequential([
            tf.keras.layers.InputLayer(shape=(None, None, 1)),
            tf.keras.layers.Lambda(lambda x: tf.tile(x, [1, 1, 1, 3])),
            tf.keras.layers.Rescaling(scale=self.scale, offset=self.offset),
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
        return (features_1 @ tf.transpose(features_2) / (feature_dimensions + 1e-8) + 1.0) ** 3.0

    def update_state(self, y_true, y_pred):
        """
        Update the metric state with new data.

        Parameters
        ----------
        y_true : tf.Tensor
            Batch of real images.
        y_pred : tf.Tensor
            Batch of generated images.
        """

        real_features = self.encoder(y_true, training=False)
        generated_features = self.encoder(y_pred, training=False)

        kernel_real = self.polynomial_kernel(real_features, real_features)
        kernel_generated = self.polynomial_kernel(generated_features, generated_features)
        kernel_cross = self.polynomial_kernel(real_features, generated_features)

        batch_size = tf.cast(tf.shape(real_features)[0], dtype=tf.float32)

        sum_kernel_real = tf.reduce_sum(kernel_real * (1.0 - tf.eye(batch_size)))
        mean_kernel_real = sum_kernel_real / ((batch_size * (batch_size - 1.0)) + 1e-8)

        sum_kernel_generated = tf.reduce_sum(kernel_generated * (1.0 - tf.eye(batch_size)))
        mean_kernel_generated = sum_kernel_generated / ((batch_size * (batch_size - 1.0)) + 1e-8)
        mean_kernel_cross = tf.reduce_mean(kernel_cross)

        value = mean_kernel_real + mean_kernel_generated - 2.0 * mean_kernel_cross

        tf.cond(pred=tf.math.is_nan(value),
                true_fn=lambda: (self.tracker.update_state(self.result())),
                false_fn=lambda: (self.tracker.update_state(value)))

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


class MetricsTracker:
    """
    A metrics tracker and manager over time.
    """

    def __init__(self, metrics=None):
        """
        Initialize the tracker instance.

        Parameters
        ----------
        metrics : list, optional
            List of metric names.
        """

        self.metrics = {}

        if metrics is not None:
            self.add(metrics)

    def add(self, metrics):
        """
        Add new metrics to the tracker.

        Parameters
        ----------
        metrics : list
            List of metric names to add.
        """

        for name in metrics:
            if name not in self.metrics:
                self.metrics[name] = tf.keras.metrics.Mean()

    def update(self, metrics):
        """
        Update the metrics with new values.

        Parameters
        ----------
        metrics : dict
            Dictionary with metric names as keys and their new values.
        """

        def _skip_update():
            return 0

        def _update_metric():
            if name not in self.metrics:
                self.add([name])

            self.metrics[name].update_state(value)
            return 0

        for name, value in metrics.items():
            nan_check = tf.reduce_any(tf.math.is_nan(value))
            tf.cond(nan_check, _skip_update, _update_metric)

    def result(self):
        """
        Return the current average results of all metrics.

        Returns
        -------
        dict
            Dictionary containing the current average of each metric.
        """

        return {name: metric.result() for name, metric in self.metrics.items()}

    def reset(self):
        """
        Reset the state of the metrics.
        """

        for metric in self.metrics.values():
            metric.reset_states()
