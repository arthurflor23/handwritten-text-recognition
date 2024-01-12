import tensorflow as tf


class MetricsTracker:
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

    def update(self, values):
        """
        Update the metrics with new values.

        Parameters
        ----------
        values : dict
            Dictionary with metric names as keys and their new values.
        """

        for name, value in values.items():
            if name in self.metrics:
                self.metrics[name].update_state(value)

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
