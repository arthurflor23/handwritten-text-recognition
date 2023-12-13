import tensorflow as tf

from models.components.loss import CTCLoss
from models.components.metric import EditDistance
from models.components.optimizer import NormalizedOptimizer


class HandwritingSynthesisRecognition(tf.keras.Model):
    """
    """

    def __init__(self, synthesis, recognition, **kwargs):
        """
        """

        super().__init__(**kwargs)

        self.style_backbone = synthesis.style_backbone
        self.style_encoder = synthesis.style_encoder
        self.generator = synthesis.generator
        self.handwriting_recognition = recognition

        self.names = [
            self.style_backbone.name,
            self.style_encoder.name,
            self.generator.name,
            self.handwriting_recognition.name,
        ]

    def __repr__(self):
        """
        Provides a formatted string with useful information.

        Returns
        -------
        str
            Formatted string with useful information.
        """

        info = "=================================================="
        info += f"\n{self.__class__.__name__.center(50)}"

        for name in self.names:
            if not hasattr(self, name):
                continue

            model = getattr(self, name)

            trainable_count = sum([tf.size(x).numpy() for x in model.trainable_variables])
            non_trainable_count = sum([tf.size(x).numpy() for x in model.non_trainable_variables])
            total_count = trainable_count + non_trainable_count

            info += "\n--------------------------------------------------"
            info += f"\n{'Model':<{25}}: {model.name}"
            info += "\n--------------------------------------------------"
            info += f"\n{'Total params':<{25}}: {total_count:,}"
            info += f"\n{'Trainable params':<{25}}: {trainable_count:,}"
            info += f"\n{'Non-trainable params':<{25}}: {non_trainable_count:,}"
            info += f"\n{'Size (MB)':<{25}}: {(total_count*4) / (1024**2):,.2f}"

        return info

    def get_weights(self):
        """
        Retrieve the weights of the submodels.

        Returns
        -------
        dict
            A dictionary with submodel names as keys and their weights as values.
        """

        with self.distribute_strategy.scope():
            weights = {}

            for name in self.names:
                if getattr(self, name) is None:
                    continue

                weights[name] = getattr(self, name).get_weights()

            return weights

    def set_weights(self, weights):
        """
        Set the weights for the submodels.

        Parameters
        ----------
        weights : dict
            A dictionary with submodel names as keys and their weights as values.
        """

        for name in self.names:
            if getattr(self, name) is None:
                continue

            getattr(self, name).set_weights(weights[name])

    def save_weights(self, filepath, overwrite=True, save_format=None, options=None):
        """
        Save the weights of the submodels.

        Parameters
        ----------
        filepath : str
            Filepath for saving the weights.
        overwrite : bool, optional
            Whether to overwrite the existing file.
        save_format : str, optional
            Format of the file to save the weights.
        options : tf.train.CheckpointOptions, optional
            Optional arguments to pass to tf.train.Checkpoint.save.
        """

        for name in self.names:
            if getattr(self, name) is None:
                continue

            getattr(self, name).save_weights(filepath=filepath.replace('model', name),
                                             overwrite=overwrite,
                                             save_format=save_format,
                                             options=options)

    def load_weights(self, filepath, by_name=False, skip_mismatch=False, options=None):
        """
        Load the weights for the submodels.

        Parameters
        ----------
        filepath : str
            Filepath for loading the weights.
        by_name : bool, optional
            Load weights by name.
        skip_mismatch : bool, optional
            Skip loading of layers where there is a mismatch in the number of weights.
        options : tf.train.CheckpointOptions, optional
            Optional arguments to pass to tf.train.Checkpoint.load.
        """

        for name in self.names:
            if getattr(self, name) is None:
                continue

            getattr(self, name).load_weights(filepath=filepath.replace('model', name),
                                             by_name=by_name,
                                             skip_mismatch=skip_mismatch,
                                             options=options)

    def compile(self, learning_rate=0.001):
        """
        Configure the submodels for training.

        This method sets up the optimizers, loss functions, and metrics for the model.

        Parameters
        ----------
        learning_rate : float, optional
            The learning rate for the optimizer.
        """

        super().compile(run_eagerly=False)

        self.r_optimizer = NormalizedOptimizer(
            tf.keras.optimizers.RMSprop(learning_rate=learning_rate))

        self.ctc_loss = CTCLoss()
        self.edit_distance = EditDistance()

    def train_step(self, input_data):
        """
        Perform the training step on the provided batch of data.

        Parameters
        ----------
        input_data : list or tuple
            A batch of data (x_data, y_data).

        Returns
        -------
        dict
            A dictionary containing metrics and losses.
        """

        print('train_step')

    def test_step(self, input_data):
        """
        Perform the testing step on the provided batch of data.

        Parameters
        ----------
        input_data : list or tuple
            A batch of data (x_data, y_data).

        Returns
        -------
        dict
            A dictionary containing evaluation metrics.
        """

        print('test_step')

    def call(self, x_data, training=None):
        """
        Processes input images and transcribes handwritten texts from them.

        Parameters
        ----------
        input_data : list or tuple
            A batch of data (x_data).
        training : bool, optional
            Indicates whether the call is for training or inference.

        Returns
        -------
        tf.Tensor
            The generated images.
        """

        print('call')
