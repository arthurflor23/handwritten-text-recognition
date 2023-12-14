import random
import tensorflow as tf

from models.components.loss import CTCLoss
from models.components.metric import EditDistance
from models.components.optimizer import NormalizedOptimizer


class HandwritingSynthesisRecognition(tf.keras.Model):
    """
    A handwriting synthesis with recognition model on the TensorFlow Keras framework.

    This model combines components for style transfer and text generation (synthesis)
        with a handwriting recognition model.
    """

    def __init__(self,
                 synthesis,
                 recognition,
                 prob_fake_images=1.0,
                 prob_fake_texts=1.0,
                 **kwargs):
        """
        Initialize the synthesis with recognition model.

        Parameters
        ----------
        synthesis : HandwritingSynthesis instance
            Synthesis model for style transfer.
        recognition : HandwritingRecognition instance
            Recognition model for transcribing text.
        prob_fake_images : float, optional
            Probability to use fake images.
        prob_fake_texts : float, optional
            Probability to use fake texts.
        **kwargs : dict
            Additional keyword arguments.
        """

        super().__init__(**kwargs)

        self.style_backbone = synthesis.style_backbone
        self.style_encoder = synthesis.style_encoder
        self.generator = synthesis.generator
        self.handwriting_recognition = recognition

        self.prob_fake_images = prob_fake_images
        self.prob_fake_texts = prob_fake_texts

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

        self.optimizer = NormalizedOptimizer(
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

        (image_inputs, text_inputs, aug_image_inputs, aug_text_inputs, _), _ = input_data

        images = aug_image_inputs
        texts = text_inputs

        if random.random() < self.prob_fake_images:
            images = image_inputs

            if random.random() < self.prob_fake_texts:
                texts = aug_text_inputs

            features_inputs, _ = self.style_backbone(images, training=False)
            latent_inputs, _, _ = self.style_encoder(features_inputs, training=False)
            images = self.generator([latent_inputs, texts], training=False)

        with tf.GradientTape() as tape:
            ctc_logits = self.handwriting_recognition(images, training=True)
            ctc_loss = self.ctc_loss(texts, ctc_logits)

        gradients = tape.gradient(ctc_loss, self.handwriting_recognition.trainable_weights)
        self.optimizer.apply_gradients(zip(gradients, self.handwriting_recognition.trainable_weights))

        self.edit_distance.update_state(texts, ctc_logits)

        return {
            'ctc_loss': ctc_loss,
            'edit_distance': self.edit_distance.result(),
        }

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

        x_data, y_data = input_data

        ctc_logits = self.call(x_data, training=False)
        ctc_loss = self.ctc_loss(y_data, ctc_logits)

        self.edit_distance.update_state(y_data, ctc_logits)

        return {
            'ctc_loss': ctc_loss,
            'edit_distance': self.edit_distance.result(),
        }

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

        image_inputs, _, _, _, _ = x_data

        ctc_logits = self.handwriting_recognition(image_inputs, training=training)

        return ctc_logits
