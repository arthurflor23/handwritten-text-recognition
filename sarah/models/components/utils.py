import tensorflow as tf


class MetricsTracker():
    """
    A metrics tracker and manager over time.
    """

    def __init__(self, metrics=None):
        """
        Initialize the tracker instance.

        Parameters
        ----------
        metrics : list, optional
            List of metric names.
        """

        self.metrics = {}

        if metrics is not None:
            self.add(metrics)

    def add(self, metrics):
        """
        Add new metrics to the tracker.

        Parameters
        ----------
        metrics : list
            List of metric names to add.
        """

        for name in metrics:
            if name not in self.metrics:
                self.metrics[name] = tf.keras.metrics.Mean()

    def update(self, metrics):
        """
        Update the metrics with new values.

        Parameters
        ----------
        metrics : dict
            Dictionary with metric names as keys and their new values.
        """

        for name, value in metrics.items():
            if name not in self.metrics:
                self.add([name])

            tf.cond(pred=tf.reduce_any(tf.math.is_nan(value)),
                    true_fn=lambda: None,
                    false_fn=lambda: self.metrics[name].update_state(value))

    def result(self):
        """
        Return the current average results of all metrics.

        Returns
        -------
        dict
            Dictionary containing the current average of each metric.
        """

        return {name: metric.result() for name, metric in self.metrics.items()}

    def reset(self):
        """
        Reset the state of the metrics.
        """

        for metric in self.metrics.values():
            metric.reset_states()
