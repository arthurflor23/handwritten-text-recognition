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

    def __init__(self, sigma=0.5, alpha=1.0, epsilon=1e-7, loss_type='l2', **kwargs):
        """
        Initializes the CTXLoss instance.

        Args:
            sigma: float, optional
                Sharpness parameter of the similarity function.
            alpha: float, optional
                Scaling factor for weighting the distances.
            epsilon: float, optional
                Small constant to avoid division by zero.
            loss_type: str, optional
                Type of loss to be used. Can be 'cosine', 'l1', or 'l2'.
            **kwargs:
                Additional keyword arguments.
        """

        super().__init__(**kwargs)

        self.sigma = sigma
        self.alpha = alpha
        self.epsilon = epsilon
        self.loss_type = loss_type

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

        cx_loss = tf.math.reduce_mean(-tf.math.log(cx))

        return cx_loss

    def compute_cosine_distance(self, y, x):
        """
        Computes the cosine distance between two tensors.

        Args:
            y: tensor,
                The target tensor.
            x: tensor,
                The prediction tensor.

        Returns:
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
        Computes the L1 (Manhattan) distance between two tensors.

        Args:
            y: tensor,
                The target tensor.
            x: tensor,
                The prediction tensor.

        Returns:
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
        Computes the L2 (Euclidean) distance between two tensors.

        Args:
            y: tensor,
                The target tensor.
            x: tensor,
                The prediction tensor.

        Returns:
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
