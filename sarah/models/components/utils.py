import tensorflow as tf


class MeasureTracker():
    """
    Tracks measures during training.

    References
    ----------
    Auxiliary Tasks in Multi-task Learning
        https://arxiv.org/abs/1805.06334

    Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics
        https://arxiv.org/abs/1705.07115
    """

    def __init__(self):
        """
        Initializes trackers.
        """

        self.means = {}
        self.values = {}
        self.weights = {}

    def add(self, measures, weight_initializer=None):
        """
        Adds new measures to track if not already present.

        Parameters
        ----------
        measures : list of str
            Measure names to add.
        weight_initializer : float, optional
            Initial value for trainable weights, if applicable.
        """

        for name in measures:
            if name not in self.values:
                self.means[name] = tf.keras.metrics.Mean()

                self.values[name] = tf.keras.Variable(name=f"{name}_value",
                                                      initializer=0.0,
                                                      dtype=tf.float32,
                                                      trainable=False)

            if weight_initializer is not None and name not in self.weights:
                self.weights[name] = tf.keras.Variable(name=f"{name}_weight",
                                                       initializer=weight_initializer,
                                                       dtype=tf.float32,
                                                       trainable=True)

    def reset(self):
        """
        Resets all tracked measures to their initial states.
        """

        for name in self.values.keys():
            self.means[name].reset_states()
            self.values[name].assign(0.0)

    def result(self, weighted=True, val_only=False, reduction=None):
        """
        Returns the current measure values.

        Parameters
        ----------
        weighted : bool, optional
            Whether to return weighted values.
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

        for name in self.values.keys():
            if not weighted and name.endswith('_w'):
                continue

            if val_only == name.startswith('val_'):
                v_name = name.removeprefix('val_')

                if reduction == 'mean':
                    results[v_name] = self.means[name].result()
                else:
                    results[v_name] = self.values[name]

        return results

    def update(self, measures):
        """
        Updates the tracked measures with new values.

        Parameters
        ----------
        measures : dict of tf.Tensor
            Dictionary of measure names and values.
        """

        for name, value in measures.items():
            if name not in self.values:
                self.add([name])

            self.means[name].update_state(value)
            self.values[name].assign(value)

    def weight(self, measures, use_variance=False):
        """
        Calculates weighted measures with adaptive regularization.

        Parameters
        ----------
        measures : dict of tf.Tensor
            Dictionary of measure names and their current values.
        use_variance : bool, optional
            Whether to use Kendall or Liebel weighting.

        Returns
        -------
        tuple
            Weighted measures dictionary and list of trainable weights.
        """

        weighted_measures = {}
        trainable_weights = []

        for name, value in measures.items():
            name = f"{name}_w"

            if name not in self.weights:
                self.add([name], weight_initializer=(1.0 if use_variance else 0.0))

            if use_variance:
                weighted_value = 0.5 / (self.weights[name] ** 2) * value
                regularization = tf.math.log(1 + self.weights[name] ** 2)
            else:
                weighted_value = 0.5 * tf.math.exp(-self.weights[name]) * value
                regularization = 0.5 * self.weights[name]

            weighted_measures[name] = weighted_value + regularization
            trainable_weights.append(self.weights[name])

            self.update({name: weighted_measures[name]})

        return weighted_measures, trainable_weights
