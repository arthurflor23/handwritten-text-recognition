import tensorflow as tf


class CTCLoss(tf.keras.losses.Loss):
    """
    Connectionist Temporal Classification (CTC) loss for sequence recognition task.

    References
    ----------
    Connectionist Temporal Classification: Labelling Unsegmented Sequence Data with Recurrent Neural Networks
        https://dl.acm.org/doi/10.1145/1143844.1143891
    """

    def __init__(self, epsilon=1e-7, name='ctc_loss', **kwargs):
        """
        Initialize the CTCLoss instance.

        Parameters
        ----------
        epsilon : float, optional
            Small constant to avoid log of zero.
        name : str, optional
            A name for the instance.
        **kwargs : dict
            Additional keyword arguments for the loss function.
        """

        super().__init__(name=name, **kwargs)

        self.epsilon = epsilon

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

        y_true = tf.reshape(y_true, (tf.shape(y_true)[0], -1))
        y_pred = tf.reshape(y_pred, (tf.shape(y_pred)[0], -1, tf.shape(y_pred)[-1]))

        labels = tf.sparse.from_dense(y_true)
        logits = tf.transpose(tf.math.log(y_pred + self.epsilon), perm=[1, 0, 2])

        logit_length = tf.reduce_sum(tf.reduce_sum(y_pred, axis=-1), axis=-1)
        # label_length = tf.math.count_nonzero(y_true, axis=-1)

        # ctc_loss = tf.nn.ctc_loss(labels=tf.cast(labels, dtype=tf.int32),
        #                           logits=tf.cast(logits, dtype=tf.float32),
        #                           label_length=tf.cast(label_length, dtype=tf.int32),
        #                           logit_length=tf.cast(logit_length, dtype=tf.int32),
        #                           logits_time_major=True,
        #                           blank_index=0)

        ctc_loss = tf.compat.v1.nn.ctc_loss(labels=tf.cast(labels, dtype=tf.int32),
                                            inputs=tf.cast(logits, dtype=tf.float32),
                                            sequence_length=tf.cast(logit_length, dtype=tf.int32),
                                            preprocess_collapse_repeated=False,
                                            ctc_merge_repeated=True,
                                            ignore_longer_outputs_than_inputs=True,
                                            time_major=True)

        ctc_loss = tf.reduce_mean(ctc_loss)

        return ctc_loss


class CXLoss(tf.keras.losses.Loss):
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
                 alpha=1.0,
                 similarity='cosine',
                 epsilon=1e-5,
                 name='cx_loss',
                 **kwargs):
        """
        Initialize the CXLoss instance.

        Parameters
        ----------
        sigma : float, optional
            Sharpness parameter of the similarity function.
        alpha : float, optional
            Scaling factor for weighting the distances.
        similarity : str, optional
            Type of loss to be used.
        epsilon : float, optional
            Small constant to avoid division by zero.
        name : str, optional
            A name for the instance.
        **kwargs : dict
            Additional keyword arguments for the loss function.
        """

        super().__init__(name=name, **kwargs)

        self.sigma = sigma
        self.alpha = alpha
        self.epsilon = epsilon
        self.similarity = similarity

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

        y_true = tf.transpose(y_true, perm=[0, 3, 1, 2])
        y_pred = tf.transpose(y_pred, perm=[0, 3, 1, 2])

        if self.similarity == 'cosine':
            distance = self.compute_cosine_distance(y_true, y_pred)

        elif self.similarity == 'l1':
            distance = self.compute_l1_distance(y_true, y_pred)

        elif self.similarity == 'l2':
            distance = self.compute_l2_distance(y_true, y_pred)

        d_min = tf.math.reduce_min(distance, axis=1, keepdims=True)
        d_tilde = distance / (d_min + self.epsilon)

        w = tf.math.exp((self.alpha - d_tilde) / self.sigma)

        cx_ij = w / tf.math.reduce_sum(w, axis=1, keepdims=True)
        cx = tf.reduce_mean(tf.reduce_max(cx_ij, axis=1), axis=1)

        cx_loss = tf.math.reduce_mean(-tf.math.log(cx + self.epsilon))

        return cx_loss

    def compute_cosine_distance(self, y, x):
        """
        Compute the cosine distance between two tensors.

        Parameters
        ----------
        y : tf.Tensor
            The target tensor.
        x : tf.Tensor
            The prediction tensor.

        Returns
        -------
        tf.Tensor
            Tensor representing the cosine distance between y and x.
        """

        y_mu = tf.reduce_mean(y, axis=[0, 2, 3], keepdims=True)

        x_centered = x - y_mu
        y_centered = y - y_mu

        x_normalized = x_centered / tf.norm(x_centered, ord=2, axis=1, keepdims=True)
        y_normalized = y_centered / tf.norm(y_centered, ord=2, axis=1, keepdims=True)

        N, C, _, _ = tf.unstack(tf.shape(x))

        x_feature = tf.reshape(x_normalized, [N, C, -1])
        y_feature = tf.reshape(y_normalized, [N, C, -1])

        x_feature = tf.transpose(x_feature, perm=[0, 2, 1])

        dist = (1. - tf.matmul(x_feature, y_feature)) / 2.

        return dist

    def compute_l1_distance(self, y, x):
        """
        Compute the L1 (Manhattan) distance between two tensors.

        Parameters
        ----------
        y : tf.Tensor
            The target tensor.
        x : tf.Tensor
            The prediction tensor.

        Returns
        -------
        tf.Tensor
            Tensor representing the L1 distance between y and x.
        """

        N, C, _, _ = tf.unstack(tf.shape(x))

        x_vec = tf.reshape(x, [N, C, -1])
        y_vec = tf.reshape(y, [N, C, -1])

        dist = tf.expand_dims(x_vec, axis=2) - tf.expand_dims(y_vec, axis=3)
        dist = tf.math.reduce_sum(tf.math.abs(dist), axis=1)
        dist = tf.transpose(dist, perm=[0, 2, 1])

        dist = tf.clip_by_value(dist, clip_value_min=0., clip_value_max=100000.)

        return dist

    def compute_l2_distance(self, y, x):
        """
        Compute the L2 (Euclidean) distance between two tensors.

        Parameters
        ----------
        y : tf.Tensor
            The target tensor.
        x : tf.Tensor
            The prediction tensor.

        Returns
        -------
        tf.Tensor
            Tensor representing the L2 distance between y and x.
        """

        N, C, _, _ = tf.unstack(tf.shape(x))

        x_vec = tf.reshape(x, [N, C, -1])
        y_vec = tf.reshape(y, [N, C, -1])

        x_s = tf.math.reduce_sum(x_vec ** 2, axis=1, keepdims=True)
        y_s = tf.math.reduce_sum(y_vec ** 2, axis=1, keepdims=True)

        A = tf.transpose(y_vec, perm=[0, 2, 1]) @ x_vec
        B = tf.transpose(x_s, perm=[0, 2, 1])

        dist = y_s - 2 * A + B
        dist = tf.transpose(dist, perm=[0, 2, 1])

        dist = tf.clip_by_value(dist, clip_value_min=0., clip_value_max=100000.)

        return dist


class BetaVAELoss(tf.keras.losses.Loss):
    """
    Loss function for beta-VAE, combining normalized reconstruction error and KL divergence.

    This loss function is particularly useful for training Variational Autoencoders, where it
        balances the reconstruction accuracy and the regularization of the latent space.

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
            Additional keyword arguments for the loss function.
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
            Tuple containing generated images, latent tensor z, mean (mu), and log-variance (logvar).

        Returns
        -------
        tf.Tensor
            Total loss combining reconstruction error and KL divergence.
        """

        generated_images, z, mu, logvar = y_pred

        N = tf.cast(tf.reduce_prod(tf.shape(y_true)[1:]), tf.float32)

        b = tf.cast(tf.shape(z)[0], tf.float32)
        m = tf.cast(tf.shape(z)[1], tf.float32)

        reconstruction_loss = (tf.reduce_sum(tf.math.abs(y_true - generated_images)) / N) / b
        kld_loss = ((-0.5 * tf.reduce_sum(1 + logvar - tf.square(mu) - tf.exp(logvar), axis=1)) / m) / b

        return tf.reduce_mean(reconstruction_loss + (self.beta * kld_loss))
