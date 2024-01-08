import numpy as np
import tensorflow as tf


class CTCLoss(tf.keras.losses.Loss):
    """
    Connectionist Temporal Classification (CTC) loss for sequence recognition task.
    """

    def __init__(self, name='ctc_loss', **kwargs):
        """
        Initialize the CTCLoss instance.

        Parameters
        ----------
        name : str, optional
            A name for the instance.
        **kwargs : dict
            Additional keyword arguments for the loss function.
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

        y_true = tf.reshape(y_true, (tf.shape(y_true)[0], -1))
        y_pred = tf.reshape(y_pred, (tf.shape(y_pred)[0], -1, tf.shape(y_pred)[-1]))

        label_length = tf.math.count_nonzero(y_true, axis=-1, keepdims=True, dtype=tf.int32)
        logit_length = tf.reduce_sum(tf.reduce_sum(y_pred, axis=-1), axis=-1, keepdims=True)

        ctc_loss = tf.keras.backend.ctc_batch_cost(y_true, y_pred, logit_length, label_length)
        ctc_loss = tf.reduce_mean(ctc_loss)

        return ctc_loss


class CTXLoss(tf.keras.losses.Loss):
    """
    Contextual loss for comparing high-level features between two tensors.

    Useful in tasks like style transfer and feature alignment. This loss
        function measures the feature similarities between two tensors.
    """

    def __init__(self,
                 sigma=0.5,
                 alpha=1.0,
                 loss_type='l2',
                 epsilon=1e-7,
                 name='ctx_loss',
                 **kwargs):
        """
        Initialize the CTXLoss instance.

        Parameters
        ----------
        sigma : float, optional
            Sharpness parameter of the similarity function.
        alpha : float, optional
            Scaling factor for weighting the distances.
        loss_type : str, optional
            Type of loss to be used, can be 'cosine', 'l1', or 'l2'.
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
        self.loss_type = loss_type

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

        if self.loss_type == 'cosine':
            dist = self.compute_cosine_distance(y_true, y_pred)
        elif self.loss_type == 'l1':
            dist = self.compute_l1_distance(y_true, y_pred)
        elif self.loss_type == 'l2':
            dist = self.compute_l2_distance(y_true, y_pred)

        d_min = tf.math.reduce_min(dist, axis=2, keepdims=True)
        d_tilde = dist / (d_min + self.epsilon)

        w = tf.math.exp((self.alpha - d_tilde) / self.sigma)

        cx_ij = w / tf.math.reduce_sum(w, axis=2, keepdims=True)
        cx = tf.math.reduce_mean(tf.math.reduce_max(cx_ij, axis=1), axis=1)

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

        N, C, H, W = tf.unstack(tf.shape(x))

        y_mu = tf.reduce_mean(y, axis=[0, 2, 3], keepdims=True)

        x_centered = x - y_mu
        y_centered = y - y_mu

        x_normalized = x_centered / tf.norm(x_centered, ord=2, axis=1, keepdims=True)
        y_normalized = y_centered / tf.norm(y_centered, ord=2, axis=1, keepdims=True)

        x_normalized = tf.reshape(x_normalized, [N, C, -1])
        y_normalized = tf.reshape(y_normalized, [N, C, -1])

        x_normalized = tf.transpose(x_normalized, perm=[0, 2, 1])

        cosine_sim = tf.matmul(x_normalized, y_normalized)
        dist = 1 - cosine_sim

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

        N, C, H, W = tf.unstack(tf.shape(x))

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

        N, C, H, W = tf.unstack(tf.shape(x))

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


class KLLoss(tf.keras.losses.Loss):
    """
    KL Divergence Loss for Variational Autoencoders (VAEs).
    It encourages the latent space distribution to approximate a standard normal distribution.
    """

    def __init__(self, name='kl_loss', **kwargs):
        """
        Initialize the KLLoss instance.

        Parameters
        ----------
        name : str, optional
            A name for the instance.
        **kwargs : dict
            Additional keyword arguments for the loss function.
        """

        super().__init__(name=name, **kwargs)

    def call(self, y_true, y_pred):
        """
        Compute the KL divergence loss for VAEs.

        Parameters
        ----------
        y_true : tf.Tensor
            Latent tensor encoded.
        y_pred : tuple
            Tuple containing mean and logarithm of the variance of the latent distribution.

        Returns
        -------
        tf.Tensor
            Computed KL divergence loss.
        """

        mu, logvar = y_pred

        log_prob_std_normal = self.log_normal_pdf(y_true, 0., 0.)
        log_prob_posterior = self.log_normal_pdf(y_true, mu, logvar)

        return -tf.reduce_mean(log_prob_std_normal - log_prob_posterior)

    def log_normal_pdf(self, z, mu, logvar, axis=1):
        """
        Compute the log probability of `z` under a Gaussian distribution.

        Parameters
        ----------
        z : tf.Tensor
            The tensor for which the log probability needs to be computed.
        mu : tf.Tensor
            Mean of the Gaussian distribution.
        logvar : tf.Tensor
            Logarithm of the variance of the Gaussian distribution.
        axis : int, optional
            The axis over which to perform the summation.

        Returns
        -------
        tf.Tensor
            The computed log probability of `z` under the Gaussian distribution.
        """

        log2pi = tf.math.log(2.0 * np.pi)
        return tf.reduce_mean(-0.5 * ((z - mu) ** 2.0 * tf.exp(-logvar) + logvar + log2pi), axis=axis)


class L1Loss(tf.keras.losses.Loss):
    """
    L1 loss for image reconstruction tasks.

    The L1 loss, also known as mean absolute error (MAE), measures the
        average magnitude of differences between predictions and actual observations.
    """

    def __init__(self, name='l1_loss', **kwargs):
        """
        Initialize the L1Loss instance.

        Parameters
        ----------
        name : str, optional
            A name for the instance.
        **kwargs : dict
            Additional keyword arguments for the loss function.
        """

        super().__init__(name=name, **kwargs)

    def call(self, y_true, y_pred):
        """
        Compute the L1 loss between the true and predicted values.

        Parameters
        ----------
        y_true : tf.Tensor
            Tensor of true values.
        y_pred : tf.Tensor
            Tensor of predicted values.

        Returns
        -------
        tf.Tensor
            The computed L1 loss.
        """

        return tf.reduce_mean(tf.abs(y_pred - y_true))
