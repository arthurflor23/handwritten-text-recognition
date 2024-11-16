import tensorflow as tf


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

    def __init__(self, name='ctc_loss', reduction='sum_over_batch_size', **kwargs):
        """
        Initialize the CTCLoss instance.

        Parameters
        ----------
        name : str, optional
            A name for the instance.
        reduction : str, optional
            Define reduction mode.
        **kwargs : dict
            Additional arguments.
        """

        super().__init__(name=name, **kwargs)

        self.reduction = reduction

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

        if self.reduction == 'sum_over_batch_size':
            ctc_loss = tf.reduce_mean(ctc_loss)

        elif self.reduction == 'sum':
            ctc_loss = tf.reduce_sum(ctc_loss)

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

    def __init__(self, sigma=0.5, name='ctx_loss', **kwargs):
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


class CyclicalVAELoss(tf.keras.losses.Loss):
    """
    Loss function for beta-VAE, combining normalized reconstruction error and KL divergence.

    References
    ----------
    Auto-Encoding Variational Bayes
        https://arxiv.org/abs/1312.6114

    beta-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework
        https://openreview.net/forum?id=Sy2fzU9gl

    Cyclical Annealing Schedule: A Simple Approach to Mitigating KL Vanishing
        https://arxiv.org/abs/1903.10145

    Understanding disentangling in β-VAE
        https://arxiv.org/abs/1804.03599
    """

    def __init__(self,
                 max_beta=1.0,
                 total_cycles=4,
                 cycle_length=4000,
                 annealing_ratio=0.5,
                 schedule_type='linear',
                 name='cyclical_vae_loss',
                 **kwargs):
        """
        Initialize the BetaVAELoss instance.

        Parameters
        ----------
        max_beta : float
            Maximum value of beta.
        total_cycles : int
            Number of cycles in the annealing schedule.
        cycle_length : int
            Number of steps per cycle.
        annealing_ratio : float
            Proportion used to increase beta within a cycle.
        schedule_type : str
            Schedule type for beta annealing ('linear', 'sigmoid', or 'cosine').
        name : str, optional
            A name for the instance.
        **kwargs : dict
            Additional arguments.
        """

        super().__init__(name=name, **kwargs)

        self.max_beta = max_beta
        self.total_cycles = total_cycles
        self.cycle_length = cycle_length
        self.annealing_ratio = annealing_ratio
        self.schedule_type = schedule_type
        self.step = 0

    def cyclical_beta(self):
        """
        Compute the cyclical beta value based on the current schedule type.

        Returns
        -------
        float
            The value of beta for the current step.
        """

        current_cycle = self.step // self.cycle_length

        if current_cycle >= self.total_cycles:
            return self.max_beta

        annealing_steps = int(self.cycle_length * self.annealing_ratio)

        if self.step % self.cycle_length <= annealing_steps:
            cycle_progress = (self.step % self.cycle_length) / annealing_steps

            if self.schedule_type == 'linear':
                beta = self.max_beta * tf.minimum(2 * cycle_progress, 1.0)
            elif self.schedule_type == 'sigmoid':
                beta = self.max_beta / (1 + tf.exp(-12 * (cycle_progress - 0.5)))
            elif self.schedule_type == 'cosine':
                beta = self.max_beta * 0.5 * (1 - tf.cos(tf.experimental.numpy.pi * cycle_progress))
            else:
                raise ValueError(f"Unknown schedule_type: {self.schedule_type}")
        else:
            beta = self.max_beta

        return beta

    def call(self, y_true, y_pred):
        """
        Compute the loss given the original and reconstructed images.

        Parameters
        ----------
        y_true : tf.Tensor
            Original input images.
        y_pred : tuple
            Tuple containing generated images, mean (mu), and log-variance (logvar).

        Returns
        -------
        tf.Tensor
            Total loss combining reconstruction error and KL divergence.
        """

        y_gen, mu, logvar = y_pred

        rec_loss = tf.reduce_mean(tf.square(y_true - y_gen))
        kl_loss = -0.5 * tf.reduce_mean(1 + logvar - tf.square(mu) - tf.exp(logvar))

        beta = self.cyclical_beta()
        self.step += 1

        return rec_loss + (beta * kl_loss)
