import tensorflow as tf


class BetaVAELoss(tf.keras.losses.Loss):
    """
    Loss function for beta-VAE, combining normalized reconstruction error and KL divergence.

    References
    ----------
    Auto-Encoding Variational Bayes
        https://arxiv.org/abs/1312.6114

    beta-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework
        https://openreview.net/forum?id=Sy2fzU9gl

    Understanding disentangling in β-VAE
        https://arxiv.org/abs/1804.03599
    """

    def __init__(self, beta=1e-4, name='beta_vae_loss', **kwargs):
        """
        Initialize the BetaVAELoss instance.

        Parameters
        ----------
        beta : float, optional
            Weight for the KL divergence term.
        name : str, optional
            A name for the instance.
        **kwargs : dict
            Additional arguments.
        """

        super().__init__(name=name, **kwargs)

        self.beta = beta

    def call(self, y_true, y_pred):
        """
        Compute the loss given the original and reconstructed images.

        Parameters
        ----------
        y_true : tf.Tensor
            Original input images.
        y_pred : tuple
            Tuple containing generated images, latent z, mean (mu), and log-variance (logvar).

        Returns
        -------
        tf.Tensor
            Total loss combining reconstruction error and KL divergence.
        """

        generated_images, z, mu, logvar = y_pred

        N = tf.cast(tf.reduce_prod(tf.shape(y_true)[1:]), tf.float32)
        b = tf.cast(tf.shape(z)[0], tf.float32)
        m = tf.cast(tf.shape(z)[1], tf.float32)

        rec_loss = tf.reduce_sum(tf.square(y_true - generated_images))
        rec_loss = rec_loss / N
        rec_loss = rec_loss / b

        kl_loss = -0.5 * tf.reduce_sum(1 + logvar - tf.square(mu) - tf.exp(logvar), axis=1)
        kl_loss = tf.reduce_sum(kl_loss) / m
        kl_loss = kl_loss / b

        return rec_loss + (self.beta * kl_loss)


class CTCLoss(tf.keras.losses.Loss):
    """
    Connectionist Temporal Classification (CTC) loss for sequence recognition task.

    References
    ----------
    A Novel Connectionist System for Unconstrained Handwriting Recognition
        https://ieeexplore.ieee.org/document/4531750

    Connectionist Temporal Classification: Labelling Unsegmented Sequence Data with Recurrent Neural Networks
        https://dl.acm.org/doi/10.1145/1143844.1143891
    """

    def __init__(self, name='ctc_loss', **kwargs):
        """
        Initialize the CTCLoss instance.

        Parameters
        ----------
        name : str, optional
            A name for the instance.
        **kwargs : dict
            Additional arguments.
        """

        super().__init__(name=name, **kwargs)

    def call(self, y_true, y_pred):
        """
        Compute the CTC loss between the true labels and predicted labels.

        Parameters
        ----------
        y_true : tf.Tensor
            Tensor of true labels.
        y_pred : tf.Tensor
            Tensor of predicted labels.

        Returns
        -------
        tf.Tensor
            The computed CTC loss.
        """

        y_true = tf.reshape(y_true, shape=(tf.shape(y_true)[0], -1))
        y_pred = tf.reshape(y_pred, shape=(tf.shape(y_pred)[0], -1, tf.shape(y_pred)[-1]))

        labels = tf.sparse.from_dense(y_true)
        logits = tf.transpose(tf.math.log(y_pred + 1e-8), perm=[1, 0, 2])

        logit_length = tf.fill([tf.shape(y_pred)[0]], tf.shape(y_pred)[1])

        ctc_loss = tf.nn.ctc_loss(labels=tf.cast(labels, dtype=tf.int32),
                                  logits=tf.cast(logits, dtype=tf.float32),
                                  label_length=None,
                                  logit_length=tf.cast(logit_length, dtype=tf.int32),
                                  logits_time_major=True,
                                  blank_index=-1)

        ctc_loss = tf.reduce_mean(ctc_loss)

        return ctc_loss


class CTXLoss(tf.keras.losses.Loss):
    """
    Contextual loss for comparing high-level features between two tensors.

    Useful in tasks like style transfer and feature alignment. This loss
        function measures the feature similarities between two tensors.

    References
    ----------
    Maintaining Natural Image Statistics with the Contextual Loss
        https://arxiv.org/abs/1803.04626

    The Contextual Loss for Image Transformation with Non-Aligned Data
        https://arxiv.org/abs/1803.02077
    """

    def __init__(self,
                 sigma=0.5,
                 name='ctx_loss',
                 **kwargs):
        """
        Initialize the CTXLoss instance.

        Parameters
        ----------
        sigma : float, optional
            Sharpness parameter of the similarity function.
        name : str, optional
            A name for the instance.
        **kwargs : dict
            Additional arguments.
        """

        super().__init__(name=name, **kwargs)

        self.sigma = sigma

    def call(self, y_true, y_pred):
        """
        Compute the contextual loss between the true and predicted tensors.

        Parameters
        ----------
        y_true : tf.Tensor
            The tensor of true values.
        y_pred : tf.Tensor
            The tensor of predicted values.

        Returns
        -------
        tf.Tensor
            The computed contextual loss.
        """

        ref_norm = tf.nn.l2_normalize(y_true, axis=-1)
        gen_norm = tf.nn.l2_normalize(y_pred, axis=-1)

        cosine_dist = tf.matmul(ref_norm, gen_norm, transpose_b=True)
        cosine_dist = 1.0 - cosine_dist

        d_min = tf.math.reduce_min(cosine_dist, axis=1, keepdims=True)
        d_tilde = cosine_dist / (d_min + 1e-5)

        w = tf.math.exp((1 - d_tilde) / self.sigma)
        ctx_ij = w / tf.math.reduce_sum(w, axis=1, keepdims=True)

        ctx = tf.reduce_mean(tf.reduce_max(ctx_ij, axis=1), axis=1)
        ctx_loss = tf.math.reduce_mean(-tf.math.log(ctx + 1e-8))

        return ctx_loss


class EditDistance(tf.keras.losses.Loss):
    """
    Edit distance for sequence comparison tasks, calculating the normalized edit distance between sequences.

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

    def call(self, y_true, y_pred):
        """
        Compute the edit distance between true and predicted sequences.

        Parameters
        ----------
        y_true : tf.Tensor
            Tensor of true labels.
        y_pred : tf.Tensor
            Tensor of predicted labels.

        Returns
        -------
        tf.Tensor
            The computed edit distance.
        """

        y_true = tf.reshape(y_true, shape=(tf.shape(y_true)[0], -1))
        y_pred = tf.reshape(y_pred, shape=(tf.shape(y_pred)[0], -1, tf.shape(y_pred)[-1]))

        labels = tf.cast(tf.sparse.from_dense(y_true), dtype=tf.int64)
        logits = tf.transpose(tf.math.log(y_pred + 1e-8), perm=[1, 0, 2])

        sequence_length = tf.fill([tf.shape(y_pred)[0]], tf.shape(y_pred)[1])

        decoded, _ = tf.nn.ctc_beam_search_decoder(inputs=tf.cast(logits, dtype=tf.float32),
                                                   sequence_length=tf.cast(sequence_length, dtype=tf.int32),
                                                   beam_width=self.beam_width,
                                                   top_paths=1)

        edit_distance = tf.edit_distance(hypothesis=decoded[0], truth=labels, normalize=True)
        edit_distance = tf.reduce_mean(edit_distance)

        return edit_distance


class KernelInceptionDistance(tf.keras.losses.Loss):
    """
    Kernel Inception Distance (KID) for assessing image generation quality.
    KID is an unbiased alternative to FID, suitable for per-batch estimation.

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

    def __init__(self, scale, offset=0.0, name='kid', **kwargs):
        """
        Initialize the KID instance.

        Parameters
        ----------
        scale : float
            Scaling factor for preprocessing.
        offset : float, optional
            Offset value for preprocessing.
        name : str, optional
            Name for the instance.
        **kwargs : dict
            Additional arguments.
        """
        super().__init__(name=name, **kwargs)

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

    def call(self, y_true, y_pred):
        """
        Compute the KID between true and predicted images.

        Parameters
        ----------
        y_true : tf.Tensor
            Real images.
        y_pred : tf.Tensor
            Generated images.

        Returns
        -------
        tf.Tensor
            Computed KID.
        """

        real_features = self.encoder(y_true, training=False)
        pred_features = self.encoder(y_pred, training=False)

        kernel_real = self.polynomial_kernel(real_features, real_features)
        kernel_generated = self.polynomial_kernel(pred_features, pred_features)
        kernel_cross = self.polynomial_kernel(real_features, pred_features)

        batch_size = tf.cast(tf.shape(real_features)[0], dtype=tf.float32)

        sum_kernel_real = tf.reduce_sum(kernel_real * (1.0 - tf.eye(batch_size)))
        mean_kernel_real = sum_kernel_real / ((batch_size * (batch_size - 1.0)) + 1e-8)

        sum_kernel_generated = tf.reduce_sum(kernel_generated * (1.0 - tf.eye(batch_size)))
        mean_kernel_generated = sum_kernel_generated / ((batch_size * (batch_size - 1.0)) + 1e-8)

        mean_kernel_cross = tf.reduce_mean(kernel_cross)

        kid = mean_kernel_real + mean_kernel_generated - 2.0 * mean_kernel_cross

        return kid

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
            Polynomial kernel result.
        """

        feature_dim = tf.cast(tf.shape(features_1)[1], dtype=tf.float32)
        return (features_1 @ tf.transpose(features_2) / (feature_dim + 1e-8) + 1.0) ** 3.0


class LossTracker():
    """
    Tracks and adapts weighted losses during training.

    References
    ----------
    Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics
        https://arxiv.org/abs/1705.07115
    """

    def __init__(self):
        """
        Initializes trackers for loss values.
        """

        self.mean_tracker = {}
        self.loss_history = {}
        self.loss_weights = {}

    def add(self, name):
        """
        Adds a new loss to track if not already present.

        Parameters
        ----------
        name : str
            Name of the loss to track.
        """

        if name not in self.loss_history:
            self.mean_tracker[name] = tf.keras.metrics.Mean()

            self.loss_history[name] = []
            self.loss_weights[name] = tf.keras.Variable(name=f"weight_{name}",
                                                        initializer=1.0,
                                                        dtype=tf.float32,
                                                        trainable=True)

    def result(self, val_loss=False, reduction=None):
        """
        Returns the current loss values.

        Parameters
        ----------
        val_loss : bool, optional
            Whether to return validation losses only.
        reduction : str, optional
            Specifies reduction type: 'mean' or None.

        Returns
        -------
        dict
            Dictionary with average or latest loss values.
        """

        loss_results = {}

        for name, values in self.loss_history.items():
            if val_loss == name.startswith('val_'):
                r_name = name.lstrip('val_')

                if reduction == 'mean':
                    loss_results[r_name] = self.mean_tracker[name].result()
                else:
                    loss_results[r_name] = values[-1]

        return loss_results

    def update(self, losses):
        """
        Updates the tracked losses with new values.

        Parameters
        ----------
        losses : dict of tf.Tensor
            Dictionary of loss names and values.
        """

        def _update(name, value):
            self.mean_tracker[name].update_state(value)
            self.loss_history[name].append(value)

        for name, loss in losses.items():
            if name not in self.loss_history:
                self.add(name)

            tf.cond(pred=tf.reduce_any(tf.math.is_nan(loss)),
                    true_fn=lambda: None,
                    false_fn=lambda: _update(name, loss))

    def weight(self, losses):
        """
        Calculates weighted losses with adaptive regularization.

        Parameters
        ----------
        losses : dict of tf.Tensor
            Dictionary of loss names and their current values.

        Returns
        -------
        tuple
            Weighted loss dictionary and list of trainable weights.
        """

        weighted_losses = {}
        loss_trainable_weights = []

        for name, loss in losses.items():
            if name not in self.loss_history:
                self.add(name)

            weighted_loss = 0.5 / (self.loss_weights[name] ** 2) * loss
            regularization = tf.math.log(1 + self.loss_weights[name] ** 2)

            weighted_losses[name] = weighted_loss + regularization
            loss_trainable_weights.append(self.loss_weights[name])

        return weighted_losses, loss_trainable_weights
