import tensorflow as tf

from sarah.models.components.layers import ExtractPatches


class EditDistance(tf.keras.metrics.Metric):
    """
    Edit distance for sequence comparison tasks.

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

        sequence_length = tf.fill([tf.shape(y_pred)[0]], value=tf.shape(y_pred)[1])

        decoded, _ = tf.keras.ops.ctc_decode(inputs=tf.cast(y_pred, dtype=tf.float32),
                                             sequence_lengths=tf.cast(sequence_length, dtype=tf.int32),
                                             strategy='beam_search',
                                             beam_width=self.beam_width,
                                             top_paths=1,
                                             merge_repeated=True,
                                             mask_index=0)

        decoded = tf.where(decoded[0] == -1, 0, decoded[0])
        decoded = tf.sparse.from_dense(decoded)

        labels = tf.cast(y_true, dtype=tf.int32)
        labels = tf.sparse.from_dense(labels)

        edit_distance = tf.edit_distance(hypothesis=decoded, truth=labels, normalize=True)
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


class KernelInceptionDistance(tf.keras.metrics.Metric):
    """
    Kernel Inception Distance (KID) for assessing image generation quality.
    KID is an unbiased alternative to FID, suitable for per-batch estimation.

    References
    ----------
    Demystifying MMD GANs
        https://arxiv.org/abs/1801.01401

    Rethinking the Inception Architecture for Computer Vision
        https://arxiv.org/abs/1512.00567
    """

    def __init__(self,
                 image_shape,
                 epsilon=1e-5,
                 name='kid',
                 **kwargs):
        """
        Initialize the KID metric instance.

        Parameters
        ----------
        image_shape : list or tuple
            Input image shape.
        epsilon : float, optional
            Small constant for numerical stability.
        name : str, optional
            A name for the instance.
        **kwargs : dict
            Additional arguments.
        """

        super().__init__(name=name, **kwargs)

        self.image_shape = image_shape
        self.epsilon = epsilon

        self.kid_image_size = (299, 299, 3)
        self.tracker = tf.keras.metrics.Mean()

        self.height = int(tf.math.ceil(self.image_shape[0] / self.kid_image_size[0]) * self.kid_image_size[0])
        self.width = int(tf.math.ceil(self.image_shape[1] / self.kid_image_size[1]) * self.kid_image_size[1])

        self.inception_encoder = tf.keras.Sequential([
            tf.keras.layers.InputLayer(shape=(None, None, 1)),
            tf.keras.layers.Resizing(height=self.height, width=self.width, interpolation='nearest'),
            ExtractPatches(patch_shape=self.kid_image_size[:2], strides=(1, 1), padding='valid'),
            tf.keras.layers.Lambda(lambda x: tf.tile(x, [1, 1, 1, 3])),
            tf.keras.applications.InceptionV3(include_top=False, input_shape=self.kid_image_size, weights='imagenet'),
            tf.keras.layers.GlobalAveragePooling2D(),
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

        batch_size = tf.cast(tf.shape(real_features)[0], dtype=tf.float32)

        sum_kernel_real = tf.reduce_sum(kernel_real * (1.0 - tf.eye(batch_size)))
        mean_kernel_real = sum_kernel_real / ((batch_size * (batch_size - 1.0)) + self.epsilon)

        sum_kernel_generated = tf.reduce_sum(kernel_generated * (1.0 - tf.eye(batch_size)))
        mean_kernel_generated = sum_kernel_generated / ((batch_size * (batch_size - 1.0)) + self.epsilon)
        mean_kernel_cross = tf.reduce_mean(kernel_cross)

        kid = mean_kernel_real + mean_kernel_generated - 2.0 * mean_kernel_cross

        self.tracker.update_state(kid)

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
        return (features_1 @ tf.transpose(features_2) / (feature_dimensions + self.epsilon) + 1.0) ** 3.0

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
