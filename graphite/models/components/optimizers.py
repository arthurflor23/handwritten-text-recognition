import tensorflow as tf


class NormalizedOptimizer(tf.keras.optimizers.Optimizer):
    """
    Custom optimizer wrapper for applying gradient normalization.

    This optimizer wraps around another tf.keras optimizer and normalizes
        the gradients according to a specified norm.

    References
    ----------
    Block-Normalized Gradient Method: An Empirical Study for Training Deep Neural Network
        https://arxiv.org/abs/1707.04822
    """

    def __init__(self,
                 optimizer,
                 normalization='l2',
                 epsilon=1e-7,
                 name='normalized_optimizer',
                 **kwargs):
        """
        Initializes the NormalizedOptimizer.

        Parameters
        ----------
        optimizer : tf.keras.optimizers.Optimizer
            The optimizer to wrap.
        normalization : str, optional
            The type of normalization.
        epsilon : float, optional
            Small float for numerical stability.
        name : str, optional
            Optional name for the operations created.
        **kwargs
            Additional keyword arguments.
        """

        super().__init__(name, **kwargs)

        self.optimizer = optimizer
        self._learning_rate = optimizer.learning_rate

        self.normalization = normalization
        self.epsilon = epsilon

        fn = {
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

        self.fn = fn[normalization]

    def get_config(self):
        """
        Returns the config of the optimizer.

        Returns
        -------
        dict
            A dictionary containing the configuration of the optimizer.
        """

        config = super().get_config()

        config.update({
            'optimizer': tf.keras.optimizers.serialize(self.optimizer),
            'normalization': self.normalization,
            'epsilon': self.epsilon,
        })

        return config

    def apply_gradients(self, grads_and_vars, name=None, skip_gradients_aggregation=False):
        """
        Apply gradients to variables.

        Parameters
        ----------
        grads_and_vars : list
            List of (gradient, variable) pairs.
        name : str, optional
            Optional name for the operations created.
        skip_gradients_aggregation : bool, optional
            If True, skip the gradient normalization.

        Returns
        -------
        An `Operation` that applies the specified gradients.
        """

        if not skip_gradients_aggregation:
            grads_and_vars = [(grad / (self.fn(grad) + self.epsilon), var)
                              for grad, var in grads_and_vars if grad is not None]

        return self.optimizer.apply_gradients(grads_and_vars, name=name)

    def _average_l1_l2_norm(self, grad):
        """
        Computes the average of the L1 and L2 norms of the gradient.

        Parameters
        ----------
        grad : tensor
            A tensor representing a gradient.

        Returns
        -------
        float
            The average of the L1 and L2 norms of the gradient.
        """

        return (self._average_l1_norm(grad) + self._average_l2_norm(grad)) / 2.

    def _average_l1_norm(self, grad):
        """
        Computes the average L1 norm of the gradient.

        Parameters
        ----------
        grad : tensor
            A tensor representing a gradient.

        Returns
        -------
        float
            The average L1 norm of the gradient.
        """

        return tf.reduce_mean(tf.abs(grad))

    def _average_l2_norm(self, grad):
        """
        Computes the average L2 norm of the gradient.

        Parameters
        ----------
        grad : tensor
            A tensor representing a gradient.

        Returns
        -------
        float
            The average L2 norm of the gradient.
        """

        return tf.sqrt(tf.reduce_mean(tf.square(grad)))

    def _l1_l2_norm(self, grad):
        """
        Computes the combination of L1 and L2 norms of the gradient.

        Parameters
        ----------
        grad : tensor
            A tensor representing a gradient.

        Returns
        -------
        float
            The combined L1 and L2 norm of the gradient.
        """

        return (self._l1_norm(grad) + self._l2_norm(grad)) / 2.

    def _l1_norm(self, grad):
        """
        Computes the L1 norm of the gradient.

        Parameters
        ----------
        grad : tensor
            A tensor representing a gradient.

        Returns
        -------
        float
            The L1 norm of the gradient.
        """

        return tf.reduce_sum(tf.abs(grad))

    def _l2_norm(self, grad):
        """
        Computes the L2 norm of the gradient.

        Parameters
        ----------
        grad : tensor
            A tensor representing a gradient.

        Returns
        -------
        float
            The L2 norm of the gradient.
        """

        return tf.sqrt(tf.reduce_sum(tf.square(grad)))

    def _max_norm(self, grad):
        """
        Computes the maximum norm (L-infinity norm) of the gradient.

        Parameters
        ----------
        grad : tensor
            A tensor representing a gradient.

        Returns
        -------
        float
            The maximum norm of the gradient.
        """

        return tf.math.maximum(tf.math.abs(grad))

    def _min_max_norm(self, grad):
        """
        Computes the average of the minimum and maximum norms of the gradient.

        Parameters
        ----------
        grad : tensor
            A tensor representing a gradient.

        Returns
        -------
        float
            The average of the minimum and maximum norms of the gradient.
        """

        return (tf.math.maximum(tf.math.abs(grad)) + tf.math.minimum(tf.math.abs(grad))) / 2.

    def _std_norm(self, grad):
        """
        Computes the standard deviation of the gradient.

        Parameters
        ----------
        grad : tensor
            A tensor representing a gradient.

        Returns
        -------
        float
            The standard deviation of the gradient.
        """

        return tf.math.reduce_std(grad)

    @classmethod
    def from_config(cls, config):
        """
        Creates an optimizer from its configuration.

        Parameters
        ----------
        config : dict
            Dictionary containing the optimizer configuration.

        Returns
        -------
        NormalizedOptimizer
            A `NormalizedOptimizer` instance.
        """

        optimizer = tf.keras.optimizers.deserialize(config.pop('optimizer'))
        normalization = config.pop('normalization', 'l2')
        epsilon = config.pop('epsilon', 1e-7)
        return cls(optimizer, normalization, epsilon, **config)
