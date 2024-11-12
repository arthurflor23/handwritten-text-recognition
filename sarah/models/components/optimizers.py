import tensorflow as tf


class GradientNormalization(tf.keras.optimizers.Optimizer):
    """
    Gradient-normalizing optimizer wrapper.

    This wrapper normalizes gradients based on the specified norm type and
        wraps around an existing `tf.keras.optimizers.Optimizer` instance.

    References
    ----------
    Block-Normalized Gradient Method: An Empirical Study for Training Deep Neural Network
        https://arxiv.org/abs/1707.04822
    """

    def __init__(self,
                 optimizer,
                 normalization='std',
                 epsilon=1e-7,
                 **kwargs):
        """
        Initialize the class instance.

        Parameters
        ----------
        optimizer : tf.keras.optimizers.Optimizer
            Base optimizer to wrap.
        normalization : str, optional
            Type of gradient normalization to apply.
        epsilon : float, optional
            Small constant for numerical stability.
        **kwargs : dict
            Additional arguments.
        """

        lr = float(tf.keras.backend.get_value(optimizer.learning_rate))

        super().__init__(learning_rate=lr, **kwargs)

        self.optimizer = optimizer
        self.normalization = normalization
        self.epsilon = epsilon

        normalization_functions = {
            'avg_l1_l2': self._average_l1_l2_norm,
            'avg_l1': self._average_l1_norm,
            'avg_l2': self._average_l2_norm,
            'l1_l2': self._l1_l2_norm,
            'l1': self._l1_norm,
            'l2': self._l2_norm,
            'max': self._max_norm,
            'min_max': self._min_max_norm,
            'std': self._std_norm,
        }

        self.fn = normalization_functions.get(normalization)

    def get_config(self):
        """
        Return the optimizer configuration.

        Returns
        -------
        dict
            Optimizer configuration as a dictionary.
        """

        config = super().get_config()

        config.update({
            'optimizer': tf.keras.optimizers.serialize(self.optimizer),
            'normalization': self.normalization,
            'epsilon': self.epsilon,
        })

        return config

    def apply_gradients(self, grads_and_vars):
        """
        Apply normalized gradients to variables.

        Parameters
        ----------
        grads_and_vars : list
            List of (gradient, variable) pairs.

        Returns
        -------
        tf.Operation
            The operation that applies gradients.
        """

        grads_and_vars = [(grad / (self.fn(grad) + self.epsilon), var)
                          for grad, var in grads_and_vars if grad is not None]

        return self.optimizer.apply_gradients(grads_and_vars)

    def _average_l1_l2_norm(self, grad):
        """
        Compute the average of L1 and L2 norms of the gradient.

        Parameters
        ----------
        grad : tf.Tensor
            Gradient tensor.

        Returns
        -------
        tf.Tensor
            Average of the L1 and L2 norms.
        """

        return (self._average_l1_norm(grad) + self._average_l2_norm(grad)) / 2.0

    def _average_l1_norm(self, grad):
        """
        Compute the average L1 norm of the gradient.

        Parameters
        ----------
        grad : tf.Tensor
            Gradient tensor.

        Returns
        -------
        tf.Tensor
            Average L1 norm.
        """

        return tf.reduce_mean(tf.abs(grad))

    def _average_l2_norm(self, grad):
        """
        Compute the average L2 norm of the gradient.

        Parameters
        ----------
        grad : tf.Tensor
            Gradient tensor.

        Returns
        -------
        tf.Tensor
            Average L2 norm.
        """

        return tf.sqrt(tf.reduce_mean(tf.square(grad)))

    def _l1_l2_norm(self, grad):
        """
        Compute the combined L1 and L2 norms of the gradient.

        Parameters
        ----------
        grad : tf.Tensor
            Gradient tensor.

        Returns
        -------
        tf.Tensor
            Combined L1 and L2 norms.
        """

        return (self._l1_norm(grad) + self._l2_norm(grad)) / 2.0

    def _l1_norm(self, grad):
        """
        Compute the L1 norm of the gradient.

        Parameters
        ----------
        grad : tf.Tensor
            Gradient tensor.

        Returns
        -------
        tf.Tensor
            L1 norm.
        """

        return tf.reduce_sum(tf.abs(grad))

    def _l2_norm(self, grad):
        """
        Compute the L2 norm of the gradient.

        Parameters
        ----------
        grad : tf.Tensor
            Gradient tensor.

        Returns
        -------
        tf.Tensor
            L2 norm.
        """

        return tf.sqrt(tf.reduce_sum(tf.square(grad)))

    def _max_norm(self, grad):
        """
        Compute the maximum (L-infinity) norm of the gradient.

        Parameters
        ----------
        grad : tf.Tensor
            Gradient tensor.

        Returns
        -------
        tf.Tensor
            Maximum norm.
        """

        return tf.reduce_max(tf.abs(grad))

    def _min_max_norm(self, grad):
        """
        Compute the average of minimum and maximum norms of the gradient.

        Parameters
        ----------
        grad : tf.Tensor
            Gradient tensor.

        Returns
        -------
        tf.Tensor
            Average of minimum and maximum norms.
        """

        return (tf.reduce_max(tf.abs(grad)) + tf.reduce_min(tf.abs(grad))) / 2.0

    def _std_norm(self, grad):
        """
        Compute the standard deviation of the gradient.

        Parameters
        ----------
        grad : tf.Tensor
            Gradient tensor.

        Returns
        -------
        tf.Tensor
            Standard deviation of the gradient.
        """
        return tf.math.reduce_std(grad)

    @classmethod
    def from_config(cls, config):
        """
        Instantiate NormalizedOptimizer from its configuration.

        Parameters
        ----------
        config : dict
            Configuration dictionary.

        Returns
        -------
        NormalizedOptimizer
            New NormalizedOptimizer instance.
        """

        optimizer = tf.keras.optimizers.deserialize(config.pop('optimizer'))
        normalization = config.pop('normalization', 'std')
        epsilon = config.pop('epsilon', 1e-7)

        return cls(optimizer, normalization, epsilon, **config)
