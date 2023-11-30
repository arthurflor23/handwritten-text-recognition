import tensorflow as tf


class CTCLoss(tf.keras.losses.Loss):
    """
    Connectionist Temporal Classification (CTC) loss for sequence recognition task.
    """

    def __init__(self, epsilon=1e-7, **kwargs):
        """
        Initializes the CTCLoss instance.

        Args:
            epsilon: float optional
                Small constant to avoid log of zero.
            **kwargs:
                Additional keyword arguments.
        """

        super().__init__(**kwargs)

        self.epsilon = epsilon

    def call(self, y_true, y_pred):
        """
        Computes the CTC loss between y_true and y_pred.

        Args:
            y_true: tensor,
                The target tensor.
            y_pred: tensor,
                The prediction tensor.

        Returns:
            A scalar tensor representing the CTC loss.
        """

        # y_true = tf.reshape(y_true, (tf.shape(y_true)[0], -1))
        # y_pred = tf.reshape(y_pred, (tf.shape(y_pred)[0], -1, tf.shape(y_pred)[-1]))

        # label_length = tf.math.count_nonzero(y_true, axis=-1, keepdims=True, dtype=tf.int32)
        # logit_length = tf.reduce_sum(tf.reduce_sum(y_pred, axis=-1), axis=-1, keepdims=True)

        # ctc_loss = tf.keras.backend.ctc_batch_cost(y_true, y_pred, logit_length, label_length)

        y_true = tf.reshape(y_true, (tf.shape(y_true)[0], -1))
        y_pred = tf.reshape(y_pred, (tf.shape(y_pred)[0], -1, tf.shape(y_pred)[-1]))

        labels = tf.sparse.from_dense(y_true)
        logits = tf.math.log(tf.transpose(y_pred, perm=[1, 0, 2]) + self.epsilon)

        label_length = tf.math.count_nonzero(y_true, axis=-1)
        logit_length = tf.reduce_sum(tf.reduce_sum(y_pred, axis=-1), axis=-1)

        ctc_loss = tf.nn.ctc_loss(labels=tf.cast(labels, dtype=tf.int32),
                                  logits=tf.cast(logits, dtype=tf.float32),
                                  label_length=tf.cast(label_length, dtype=tf.int32),
                                  logit_length=tf.cast(logit_length, dtype=tf.int32),
                                  logits_time_major=True,
                                  blank_index=-1)

        ctc_loss = tf.reduce_mean(ctc_loss)

        return ctc_loss


class CTXLoss(tf.keras.losses.Loss):
    """
    Contextual loss for comparing high-level features between two tensors.
    Useful in tasks like style transfer and feature alignment.
    """

    def __init__(self, sigma=0.5, alpha=1.0, epsilon=1e-7, **kwargs):
        """
        Initializes the CTXLoss instance.

        Args:
            sigma: float, optional
                Sharpness parameter of the similarity function.
            alpha: float, optional
                Scaling factor for weighting the distances.
            epsilon: float, optional
                Small constant to avoid division by zero.
            **kwargs:
                Additional keyword arguments.
        """

        super().__init__(**kwargs)

        self.sigma = sigma
        self.alpha = alpha
        self.epsilon = epsilon

    def call(self, y_true, y_pred):
        """
        Computes the contextual loss between y_true and y_pred.

        Args:
            y_true: tensor,
                The target tensor.
            y_pred: tensor,
                The prediction tensor.

        Returns:
            A scalar tensor representing the contextual loss.
        """

        y_mean_true = tf.reduce_mean(y_true, axis=[0, 1, 2], keepdims=True)
        y_pred_centered = y_pred - y_mean_true
        y_true_centered = y_true - y_mean_true

        y_pred_norm = tf.nn.l2_normalize(y_pred_centered, axis=-1)
        y_true_norm = tf.nn.l2_normalize(y_true_centered, axis=-1)

        B, H, W, C = y_true_norm.shape
        P = H * W
        y_true_patches = tf.reshape(y_true_norm, (B, P, C))
        y_true_patches = tf.transpose(y_true_patches, perm=[0, 2, 1])

        y_pred_norm = tf.transpose(y_pred_norm, perm=[0, 3, 1, 2])
        dist = tf.nn.conv2d(y_pred_norm, y_true_patches, strides=[1, 1, 1, 1], padding='VALID')

        raw_dist = (1. - dist) / 2.
        div = tf.reduce_min(raw_dist, axis=1, keepdims=True)
        relative_dist = raw_dist / (div + self.epsilon)

        cx_weights = tf.exp((self.alpha - relative_dist) / self.sigma)
        cx_weights_sum = tf.reduce_sum(cx_weights, axis=1, keepdims=True)
        cx = cx_weights / cx_weights_sum

        cx_max = tf.reduce_max(cx, axis=[2, 3])
        cx_mean = tf.reduce_mean(cx_max, axis=1)
        cx_loss = tf.reduce_mean(-tf.math.log(cx_mean + self.epsilon))

        return cx_loss


class L1Loss(tf.keras.losses.Loss):
    """
    L1 loss for image reconstruction task.
    """

    def __init__(self, **kwargs):
        """
        Initializes the L1Loss instance.

        Args:
            **kwargs:
                Additional keyword arguments.
        """

        super().__init__(**kwargs)

    def call(self, y_true, y_pred):
        """
        Computes the L1 loss between y_true and y_pred.

        Args:
            y_true: tensor,
                The target tensor.
            y_pred: tensor,
                The prediction tensor.

        Returns:
            A scalar tensor representing the L1 loss.
        """

        y_diff = tf.math.abs(y_pred - y_true)

        sum_diff = tf.cast(tf.reduce_sum(y_diff), tf.float32)
        num_elements = tf.cast(tf.reduce_prod(tf.shape(y_diff)[1:]), tf.float32)

        l1_loss = sum_diff / num_elements

        return l1_loss
