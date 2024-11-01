import tensorflow as tf


class MeasureTracker():
    """
    Tracks measures during training.

    References
    ----------
    Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics
        https://arxiv.org/abs/1705.07115
    """

    def __init__(self):
        """
        Initializes trackers.
        """

        self.history = {}
        self.tracker = {}
        self.weights = {}

    def add(self, name):
        """
        Adds a new measure to track if not already present.

        Parameters
        ----------
        name : str
            Name of the measure to track.
        """

        if name not in self.history:
            self.history[name] = []
            self.tracker[name] = tf.keras.metrics.Mean()
            self.weights[name] = tf.keras.Variable(name=f"weight_{name}",
                                                   initializer=1.0,
                                                   dtype=tf.float32,
                                                   trainable=True)

    def result(self, val_only=False, reduction=None):
        """
        Returns the current measure values.

        Parameters
        ----------
        val_only : bool, optional
            Whether to return validation values only.
        reduction : str, optional
            Specifies reduction type: 'mean' or None.

        Returns
        -------
        dict
            Dictionary with average or latest measure values.
        """

        results = {}

        for name, values in self.history.items():
            if val_only == name.startswith('val_'):
                result_name = name.lstrip('val_')

                if reduction == 'mean':
                    results[result_name] = self.tracker[name].result()
                else:
                    results[result_name] = values[-1]

        return results

    def update(self, measures):
        """
        Updates the tracked measures with new values.

        Parameters
        ----------
        measures : dict of tf.Tensor
            Dictionary of measure names and values.
        """

        def _update(name, value):
            self.history[name].append(value)
            self.tracker[name].update_state(value)

        for name, value in measures.items():
            if name not in self.history:
                self.add(name)

            tf.cond(pred=tf.reduce_any(tf.math.is_nan(value)),
                    true_fn=lambda: None,
                    false_fn=lambda: _update(name, value))

    def weight(self, measures):
        """
        Calculates weighted measures with adaptive regularization.

        Parameters
        ----------
        measures : dict of tf.Tensor
            Dictionary of measure names and their current values.

        Returns
        -------
        tuple
            Weighted measures dictionary and list of trainable weights.
        """

        weighted_measures = {}
        trainable_weights = []

        for name, value in measures.items():
            if name not in self.history:
                self.add(name)

            weighted_value = 0.5 / (self.weights[name] ** 2) * value
            regularization = tf.math.log(1 + self.weights[name] ** 2)

            weighted_measures[name] = weighted_value + regularization
            trainable_weights.append(self.weights[name])

        return weighted_measures, trainable_weights
