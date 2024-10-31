import tensorflow as tf


class MetricTracker():
    """
    Tracks and adapts weighted losses over training.
    """

    def __init__(self, metrics=None):
        """
        Initializes metrics tracker.

        Parameters
        ----------
        metrics : list of str, optional
            Names of metrics to initialize.
        """

        self.metrics = {}
        self.weights = {}

        if metrics is not None:
            self.add(metrics)

    def add(self, metrics):
        """
        Adds new metrics to track.

        Parameters
        ----------
        metrics : list of str
            Metric names to add.
        """

        for name in metrics:
            if name not in self.metrics:
                self.metrics[name] = tf.keras.metrics.Mean()

                self.weights[name] = tf.keras.Variable(name=f"weight_{name}",
                                                       initializer=1.0,
                                                       dtype=tf.float32,
                                                       trainable=True)

    def reset(self):
        """
        Resets all tracked metrics.
        """

        for metric in self.metrics.values():
            metric.reset_states()

    def result(self):
        """
        Computes averages of all tracked metrics.

        Returns
        -------
        dict
            Dictionary of average values per metric.
        """

        return {name: metric.result() for name, metric in self.metrics.items()}

    def update(self, metrics):
        """
        Updates metrics with new values.

        Parameters
        ----------
        metrics : dict of tf.Tensor
            Metric names and values to update.
        """

        for name, value in metrics.items():
            if name not in self.metrics:
                self.add([name])

            tf.cond(pred=tf.reduce_any(tf.math.is_nan(value)),
                    true_fn=lambda: None,
                    false_fn=lambda: self.metrics[name].update_state(value))

    def weight(self, metrics):
        """
        Calculates adaptive weighted metrics.

        Parameters
        ----------
        metrics : dict of tf.Tensor
            Metric names and values to compute weights.

        Returns
        -------
        dict of tf.Tensor
            Dictionary of weighted individual metrics.
        """

        weighted_losses = {}
        trainable_weights = []

        for name, value in metrics.items():
            if name not in self.metrics:
                self.add([name])

            sigma = tf.nn.relu(self.weights[name]) + 1e-8
            weighted_losses[name] = (1 / (2 * tf.square(sigma))) * value + tf.math.log(sigma)

            trainable_weights.append(self.weights[name])

        return weighted_losses, trainable_weights
