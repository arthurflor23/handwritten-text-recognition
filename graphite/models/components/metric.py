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

    Reference:
        [Demystifying MMD GANs - Kernel Inception Distance (KID)](https://arxiv.org/abs/1801.01401).
        [Rethinking the Inception Architecture for Computer Vision](https://arxiv.org/abs/1512.00567).
        [InceptionV3](https://keras.io/api/applications/inceptionv3/).
        [ImageNet](https://www.tensorflow.org/datasets/catalog/imagenet2012).
    """

    def __init__(self, name='kid', **kwargs):
        """
        Initialize the KID metric.

        Args:
            name (str, optional):
                Name of the metric.
            **kwargs:
                Additional keyword arguments.
        """

        super().__init__(name=name, **kwargs)

        self.kid_tracker = tf.keras.metrics.Mean()
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

        Args:
            features_1 (tensor):
                First set of features.
            features_2 (tensor):
                Second set of features.

        Returns:
            Polynomial kernel between the two feature sets.
        """

        feature_dimensions = tf.cast(tf.shape(features_1)[1], dtype=tf.float32)
        return (features_1 @ tf.transpose(features_2) / feature_dimensions + 1.0) ** 3.0

    def update_state(self, real_images, generated_images, sample_weight=None):
        """
        Update the state of the metric with new data.

        Args:
            real_images (tensor):
                Batch of real images.
            generated_images (tensor):
                Batch of generated images.
            sample_weight (tensor, optional):
                Sample weights.
        """

        real_features = self.encoder(real_images, training=False)
        generated_features = self.encoder(generated_images, training=False)

        kernel_real = self.polynomial_kernel(real_features, real_features)
        kernel_generated = self.polynomial_kernel(generated_features, generated_features)
        kernel_cross = self.polynomial_kernel(real_features, generated_features)

        batch_size = tf.shape(real_features)[0]
        batch_size_f = tf.cast(batch_size, dtype=tf.float32)

        sum_kernel_real = tf.reduce_sum(kernel_real * (1.0 - tf.eye(batch_size)))
        mean_kernel_real = sum_kernel_real / (batch_size_f * (batch_size_f - 1.0))

        sum_kernel_generated = tf.reduce_sum(kernel_generated * (1.0 - tf.eye(batch_size)))
        mean_kernel_generated = sum_kernel_generated / (batch_size_f * (batch_size_f - 1.0))

        mean_kernel_cross = tf.reduce_mean(kernel_cross)

        kid = mean_kernel_real + mean_kernel_generated - 2.0 * mean_kernel_cross
        self.kid_tracker.update_state(kid)

    def result(self):
        """
        Return the current result of the metric.

        Returns:
            The current KID value.
        """

        return self.kid_tracker.result()

    def reset_state(self):
        """
        Reset the state of the metric.
        """

        self.kid_tracker.reset_state()
