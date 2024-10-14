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
            Additional keyword arguments for the loss function.
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

        rec_loss = tf.reduce_sum(tf.square(y_true - generated_images))
        rec_loss = rec_loss / N
        rec_loss = rec_loss / b

        kl_loss = -0.5 * tf.reduce_sum(1 + logvar - tf.square(mu) - tf.exp(logvar), axis=1)
        kl_loss = tf.reduce_sum(kl_loss) / m
        kl_loss = kl_loss / b

        return rec_loss + (self.beta * kl_loss)


class MaskLoss(tf.keras.losses.Loss):
    """
    Implements a multi loss combining BCE, Focal, and Dice Loss.

    References
    ----------
    Focal Loss for Dense Object Detection
        https://arxiv.org/abs/1708.02002

    Generalised Dice overlap as a deep learning loss function for highly unbalanced segmentations
        https://arxiv.org/abs/1707.03237v3

    HistoSeg : Quick attention with multi-loss function for multi-structure segmentation in digital histology images
        https://arxiv.org/pdf/2209.00729v1
    """

    def __init__(self,
                 mask_value=0,
                 beta=0.1,
                 normed=True,
                 name='msk_loss',
                 **kwargs):
        """
        Initializes the multi-loss combining BCE, Focal, and Dice Loss.

        Parameters
        ----------
        mask_value : float or int
            Mask value.
        beta : float, optional
            Weight for the KL divergence term.
        normed : bool, optional
            Whether in normed mode.
        name : str
            Loss function name.
        **kwargs : dict
            Additional keyword arguments.
        """

        super().__init__(name=name, **kwargs)

        self.mask_value = mask_value
        self.beta = beta
        self.normed = normed

        if self.normed:
            self.mask_value = (self.mask_value / 127.5) - 1

    def call(self, y_true, y_pred):
        """
        Calculate multi loss between mask and predicted tensors.

        Parameters
        ----------
        y_true : tf.Tensor
            Mask tensor values.
        y_pred : tf.Tensor
            Predicted tensor values.

        Returns
        -------
        tf.Tensor
            Mask loss.
        """

        mask_pred = self.generate_mask(y_pred, mask_value=self.mask_value)

        bce_loss = self.compute_bce_loss(y_true, mask_pred)
        focal_loss = self.compute_focal_loss(y_true, mask_pred)
        dice_loss = self.compute_dice_loss(y_true, mask_pred)

        msk_loss = bce_loss + focal_loss + dice_loss

        return self.beta * msk_loss

    def compute_bce_loss(self, y_true, y_pred):
        """
        Computes Binary Crossentropy (BCE) Loss.

        Parameters
        ----------
        y_true : tf.Tensor
            Mask tensor values.
        y_pred : tf.Tensor
            Predicted mask tensor values.

        Returns
        -------
        tf.Tensor
            BCE loss value.
        """

        positive_loss = y_true * tf.math.log(y_pred + 1e-8)
        negative_loss = (1 - y_true) * tf.math.log(1 - y_pred + 1e-8)

        bce_loss = -(positive_loss + negative_loss)
        bce_loss = tf.reduce_mean(bce_loss)

        return bce_loss

    def compute_focal_loss(self, y_true, y_pred):
        """
        Computes Focal Loss for handling class imbalance.

        Parameters
        ----------
        y_true : tf.Tensor
            Mask tensor values.
        y_pred : tf.Tensor
            Predicted mask tensor values.

        Returns
        -------
        tf.Tensor
            Focal loss value.
        """

        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        focal_weight = 0.25 * tf.pow(1.0 - p_t, 2.0)

        focal_loss = -focal_weight * tf.math.log(p_t + 1e-8)
        focal_loss = tf.reduce_mean(focal_loss)

        return focal_loss

    def compute_dice_loss(self, y_true, y_pred):
        """
        Computes Dice Loss to measure similarity between true and predicted masks.

        Parameters
        ----------
        y_true : tf.Tensor
            Mask tensor values.
        y_pred : tf.Tensor
            Predicted mask tensor values.

        Returns
        -------
        tf.Tensor
            Dice loss value.
        """

        y_true_f = tf.keras.backend.flatten(y_true)
        y_pred_f = tf.keras.backend.flatten(y_pred)

        intersection = tf.reduce_sum(y_true_f * y_pred_f)
        union = tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f)

        dice_coeff = (2. * intersection + 1e-8) / (union + 1e-8)
        dice_loss = 1 - dice_coeff

        return dice_loss

    def generate_mask(self, input_data, mask_value):
        """
        Create a mask for padded areas in the input data.

        Parameters
        ----------
        input_data : tf.Tensor
            The input tensor.
        mask_value : float, or int,
            The mask value.

        Returns
        -------
        tf.Tensor
            Boolean mask tensor.
        """

        def _get_mask(input_data, mask_value, transpose):
            shape = tf.shape(input_data)[1:-1]

            if transpose:
                shape = shape[::-1]
                input_data = tf.transpose(input_data, perm=[0, 2, 1, 3])

            input_mean = tf.reduce_mean(input_data, axis=[2, 3])

            data_reversed = tf.reverse(input_mean, axis=[1])
            padding_mask = tf.equal(data_reversed, tf.cast(mask_value, input_data.dtype))

            lengths = tf.argmax(tf.cast(~padding_mask, tf.int32), axis=1, output_type=tf.int32)
            origin_lens = tf.where(tf.equal(lengths, 0), shape[0], shape[0] - lengths)

            scale = tf.cast(origin_lens, tf.float32) / (tf.cast(shape[0], tf.float32) + 1e-8)
            target_lens = tf.cast(shape[0], tf.float32) * scale

            mask = tf.sequence_mask(tf.math.ceil(target_lens), maxlen=shape[0])
            mask = tf.expand_dims(tf.expand_dims(mask, axis=-1), axis=-1)
            mask = tf.tile(mask, multiples=[1, 1, shape[1], 1])

            if transpose:
                mask = tf.transpose(mask, perm=[0, 2, 1, 3])

            return mask

        v_mask = _get_mask(input_data, mask_value, transpose=False)
        h_mask = _get_mask(input_data, mask_value, transpose=True)

        mask = tf.cast(tf.logical_and(v_mask, h_mask), dtype=input_data.dtype)

        return mask
