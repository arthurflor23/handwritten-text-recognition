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

        labels = tf.cast(tf.sparse.from_dense(y_true), dtype=tf.int32)
        length = tf.fill([tf.shape(y_pred)[0]], value=tf.shape(y_pred)[1])

        ctc_loss = tf.compat.v1.nn.ctc_loss(labels=labels,
                                            logits=y_pred,
                                            sequence_length=length,
                                            preprocess_collapse_repeated=False,
                                            ctc_merge_repeated=True,
                                            ignore_longer_outputs_than_inputs=True,
                                            time_major=False)

        # ctc_loss = tf.nn.ctc_loss(labels=labels,
        #                           logits=y_pred,
        #                           label_length=length,
        #                           logit_length=length,
        #                           logits_time_major=False,
        #                           blank_index=-1)

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
                 distance='cosine',
                 name='ctx_loss',
                 **kwargs):
        """
        Initialize the CTXLoss instance.

        Parameters
        ----------
        sigma : float, optional
            Sharpness parameter of the similarity function.
        distance : str, optional
            Distance measure to use: 'cosine', 'l1', or 'l2'.
        name : str, optional
            A name for the instance.
        **kwargs : dict
            Additional arguments.
        """

        super().__init__(name=name, **kwargs)

        self.sigma = sigma
        self.distance = distance

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

        y_true_flat = tf.reshape(y_true, shape=(tf.shape(y_true)[0], -1))
        y_pred_flat = tf.reshape(y_pred, shape=(tf.shape(y_pred)[0], -1))

        if self.distance == 'cosine':
            y_true_flat = tf.nn.l2_normalize(y_true_flat, axis=-1)
            y_pred_flat = tf.nn.l2_normalize(y_pred_flat, axis=-1)

            cosine_similarity = tf.matmul(y_true_flat, y_pred_flat, transpose_b=True)
            distance = 1.0 - cosine_similarity
        else:
            y_true_expanded = tf.expand_dims(y_true_flat, axis=1)
            y_pred_expanded = tf.expand_dims(y_pred_flat, axis=0)
            diff = y_true_expanded - y_pred_expanded

            if self.distance == 'l1':
                distance = tf.reduce_sum(tf.abs(diff), axis=-1)

            elif self.distance == 'l2':
                distance = tf.reduce_sum(tf.square(diff), axis=-1)

        d_min = tf.math.reduce_min(distance, axis=1, keepdims=True)
        d_tilde = tf.keras.ops.divide_no_nan(distance, d_min)

        w = tf.math.exp((1 - d_tilde) / self.sigma)
        ctx_ij = tf.keras.ops.divide_no_nan(w, tf.reduce_sum(w, axis=1, keepdims=True))

        ctx = tf.reduce_mean(tf.reduce_max(ctx_ij, axis=1))
        ctx_loss = tf.math.reduce_mean(-tf.math.log(ctx + tf.keras.backend.epsilon()))

        return ctx_loss


class KLDivergence(tf.keras.losses.Loss):
    """
    Loss function for beta-VAE using KL divergence with cyclical annealing.

    References
    ----------
    beta-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework
        https://openreview.net/forum?id=Sy2fzU9gl

    Cyclical Annealing Schedule: A Simple Approach to Mitigating KL Vanishing
        https://arxiv.org/abs/1903.10145

    Toward Multimodal Image-to-Image Translation
        https://arxiv.org/abs/1711.11586

    Understanding disentangling in β-VAE
        https://arxiv.org/abs/1804.03599
    """

    def __init__(self,
                 max_beta=1.0,
                 total_cycles=4,
                 warmup_steps=10000,
                 annealing_ratio=0.5,
                 schedule_type='cosine',
                 cyclical_annealing=False,
                 name='cyclical_vae_loss',
                 **kwargs):
        """
        Initialize the class instance.

        Parameters
        ----------
        max_beta : float, optional
            Maximum value of beta.
        total_cycles : int, optional
            Number of cycles in the annealing schedule.
        warmup_steps : int, optional
            Number of steps per cycle.
        annealing_ratio : float, optional
            Proportion used to increase beta within a cycle.
        schedule_type : str, optional
            Schedule type for beta annealing ('linear', 'sigmoid', or 'cosine').
        cyclical_annealing : bool, optional
            Whether to use cyclical annealing process.
        name : str, optional
            A name for the instance.
        **kwargs : dict
            Additional arguments.
        """

        super().__init__(name=name, **kwargs)

        self.max_beta = max_beta
        self.total_cycles = total_cycles
        self.warmup_steps = warmup_steps
        self.annealing_ratio = annealing_ratio
        self.schedule_type = schedule_type
        self.cyclical_annealing = cyclical_annealing
        self.step = 0

    def cyclical_beta(self):
        """
        Compute the cyclical beta value based on the current schedule type.

        Returns
        -------
        float
            The value of beta for the current step.
        """

        current_cycle = self.step // self.warmup_steps

        if current_cycle >= self.total_cycles:
            beta = self.max_beta
        else:
            cycle_position = self.step % self.warmup_steps
            annealing_steps = int(self.warmup_steps * self.annealing_ratio)

            if cycle_position <= annealing_steps:
                cycle_progress = cycle_position / annealing_steps

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

        self.step += 1
        return beta

    def call(self, mu, logvar):
        """
        Compute scaled KL divergence loss.

        Parameters
        ----------
        mu : tf.Tensor
            Mean of the latent variable distribution.
        logvar : tf.Tensor
            Log-variance of the latent variable distribution.

        Returns
        -------
        tf.Tensor
            Scaled KL divergence loss.
        """

        kld_loss = tf.reduce_mean(-0.5 * tf.reduce_sum(1 + logvar - tf.square(mu) - tf.exp(logvar), axis=1))

        if self.cyclical_annealing:
            beta = self.cyclical_beta()
            kld_loss = beta * kld_loss

        return kld_loss
