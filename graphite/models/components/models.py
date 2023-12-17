import re
import string
import random
import numpy as np
import editdistance
import tensorflow as tf

from models.components.losses import CTCLoss
from models.components.losses import CTXLoss
from models.components.losses import L1Loss
from models.components.metrics import EditDistance
from models.components.metrics import KernelInceptionDistance
from models.components.optimizers import NormalizedOptimizer


class BaseModel(tf.keras.Model):
    """
    A base model class that extends tf.keras.Model, providing additional
        functionalities for model representation and weight management.
    """

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
            model = getattr(self, name, None)

            if model is None:
                continue

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
                model = getattr(self, name, None)

                if model is not None:
                    weights[name] = model.get_weights()

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
            model = getattr(self, name, None)

            if model is not None:
                model.set_weights(weights[name])

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
            model = getattr(self, name, None)

            if model is not None:
                model.save_weights(filepath=filepath.replace('<model>', name),
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
            model = getattr(self, name, None)

            if model is not None:
                model.load_weights(filepath=filepath.replace('<model>', name),
                                   by_name=by_name,
                                   skip_mismatch=skip_mismatch,
                                   options=options)


class SynthesisBaseModel(BaseModel):
    """
    SynthesisBaseModel extends BaseModel to provide additional
        functionalities to synthesis models.
    """

    def __init__(self,
                 image_shape,
                 lexical_shape,
                 writers_shape,
                 **kwargs):
        """
        Initialize the synthesis model with specified parameters for each submodel.

        Parameters
        ----------
        image_shape : tuple or list
            The shape of the input images.
        lexical_shape : tuple or list
            The shape of the lexical input.
        writers_shape : int
            The dimension for the writer identification.
        **kwargs : dict
            Additional keyword arguments.
        """

        super().__init__(name='synthesis', **kwargs)

        self.image_shape = image_shape
        self.lexical_shape = lexical_shape
        self.writers_shape = writers_shape

        self.generator = None
        self.style_backbone = None
        self.style_encoder = None
        self.discriminator = None
        self.patch_discriminator = None
        self.identification = None
        self.recognition = None

        self.names = [
            'generator',
            'style_backbone',
            'style_encoder',
            'discriminator',
            'patch_discriminator',
            'identification',
            'recognition',
        ]

        self.monitor = 'kid'
        self.build_model()

    def get_config(self):
        """
        Retrieves the configuration of the model for serialization.

        Returns
        -------
        dict
            A dictionary containing the configuration of the model.
        """

        config = super().get_config()

        config.update({
            'image_shape': self.image_shape,
            'lexical_shape': self.lexical_shape,
            'writers_shape': self.writers_shape,
        })

        return config

    def compile(self, learning_rate=0.001):
        """
        Configure the submodels.

        This method sets up the optimizers, loss functions, and metrics for the model.

        Parameters
        ----------
        learning_rate : float, optional
            The learning rate for the optimizer.
        """

        super().compile(run_eagerly=False)

        self.g_optimizer = NormalizedOptimizer(
            tf.keras.optimizers.AdamW(learning_rate=learning_rate, weight_decay=0.001))

        self.d_optimizer = NormalizedOptimizer(
            tf.keras.optimizers.AdamW(learning_rate=learning_rate, weight_decay=0.001))

        self.p_optimizer = NormalizedOptimizer(
            tf.keras.optimizers.AdamW(learning_rate=learning_rate, weight_decay=0.001))

        self.b_optimizer = NormalizedOptimizer(
            tf.keras.optimizers.AdamW(learning_rate=learning_rate, weight_decay=0.001))

        self.e_optimizer = NormalizedOptimizer(
            tf.keras.optimizers.AdamW(learning_rate=learning_rate, weight_decay=0.001))

        self.w_optimizer = NormalizedOptimizer(
            tf.keras.optimizers.AdamW(learning_rate=learning_rate, weight_decay=0.001))

        self.r_optimizer = NormalizedOptimizer(
            tf.keras.optimizers.AdamW(learning_rate=learning_rate, weight_decay=0.001))

        self.l1_loss = L1Loss()
        self.ctx_loss = CTXLoss()
        self.ctc_loss = CTCLoss()
        self.kld_loss = tf.keras.losses.KLDivergence()
        self.cls_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        self.kid = KernelInceptionDistance()

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

        x_data, _ = input_data

        generated_images = self.call(x_data, training=False)
        self.kid.update_state(x_data[0], generated_images)

        return {
            self.kid.name: self.kid.result(),
        }

    def call(self, x_data, training=None):
        """
        Processes input images and text through the style backbone, encoder,
            and generator to produce generated images.

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

        image_inputs, text_inputs, _, _, _ = x_data

        features_inputs, _ = self.style_backbone(image_inputs, training=training)
        latent_inputs, _, _ = self.style_encoder(features_inputs, training=training)
        generated_images = self.generator([latent_inputs, text_inputs], training=training)

        return generated_images


class SynthesisRecognitionBaseModel(BaseModel):
    """
    SynthesisRecognitionBaseModel extends BaseModel to provide additional
        functionalities to synthesis and recognition models.
    """

    def __init__(self,
                 image_shape,
                 lexical_shape,
                 generator=None,
                 style_backbone=None,
                 style_encoder=None,
                 synthesis_ratio=1.0,
                 **kwargs):
        """
        Initializes the synthesis and recognition model.

        Parameters
        ----------
        image_shape : tuple or list
            The shape of the input images.
        lexical_shape : tuple or list
            The shape of the lexical input.
        generator : Generator instance
            Generator model for image generation.
        style_backbone : StyleBackbone instance
            StyleBackbone model for extracting style patterns from images.
        style_encoder : StyleEncoder instance
            StyleEncoder model for encoding extracted style features.
        synthesis_ratio : float, optional
            Probability to use synthetic data.
        **kwargs : dict
            Additional keyword arguments.
        """

        super().__init__(name='recognition', **kwargs)

        self.image_shape = image_shape
        self.lexical_shape = lexical_shape
        self.synthesis_ratio = synthesis_ratio

        self.generator = generator
        self.style_backbone = style_backbone
        self.style_encoder = style_encoder
        self.recognition = None

        self.names = [
            'generator',
            'style_backbone',
            'style_encoder',
            'recognition',
        ]

        self.monitor = 'val_cer'
        self.build_model()

    def get_config(self):
        """
        Retrieves the configuration of the model for serialization.

        Returns
        -------
        dict
            A dictionary containing the configuration of the model.
        """

        config = super().get_config()

        config.update({
            'image_shape': self.image_shape,
            'lexical_shape': self.lexical_shape,
        })

        return config

    def compile(self, learning_rate=0.001):
        """
        Configure the submodels.

        This method sets up the optimizers, loss functions, and metrics for the model.

        Parameters
        ----------
        learning_rate : float, optional
            The learning rate for the optimizer.
        """

        super().compile(run_eagerly=False)

        self.optimizer = NormalizedOptimizer(
            tf.keras.optimizers.AdamW(learning_rate=learning_rate, weight_decay=0.001))

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

        if self.generator and self.style_backbone and self.style_encoder:
            if random.random() <= self.synthesis_ratio:
                images = image_inputs
                texts = aug_text_inputs

                features_inputs, _ = self.style_backbone(images, training=False)
                latent_inputs, _, _ = self.style_encoder(features_inputs, training=False)
                images = self.generator([latent_inputs, texts], training=False)

        with tf.GradientTape() as tape:
            ctc_logits = self.recognition(images, training=True)
            ctc_loss = self.ctc_loss(texts, ctc_logits)

        gradients = tape.gradient(ctc_loss, self.recognition.trainable_weights)
        self.optimizer.apply_gradients(zip(gradients, self.recognition.trainable_weights))

        self.edit_distance.update_state(texts, ctc_logits)

        return {
            self.ctc_loss.name: ctc_loss,
            self.edit_distance.name: self.edit_distance.result(),
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
            self.ctc_loss.name: ctc_loss,
            self.edit_distance.name: self.edit_distance.result(),
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

        ctc_logits = self.recognition(image_inputs, training=training)

        return ctc_logits

    def ctc_decode(self, predictions, top_paths=1, beam_width=30, tokenizer=None, verbose=1):
        """
        Decode CTC predictions using beam search.

        Parameters
        ----------
        predictions : numpy.ndarray
            CTC predictions to be decoded.
        top_paths : int
            Number of top paths to consider.
        beam_width : int
            Beam width for beam search.
        tokenizer : Tokenizer, optional
            Tokenizer for decoding text predictions.
        verbose : int, optional
            Verbosity mode.

        Returns
        -------
        tuple
            A tuple containing the decoded predictions and corresponding probabilities.
        """

        predictions, probabilities = np.log(predictions + 1e-8), np.array([])
        progbar = tf.keras.utils.Progbar(target=predictions.shape[1], unit_name='decode', verbose=verbose)

        decoded_paths, probabilities_list = [], []
        sequence_length = [predictions.shape[2]] * predictions.shape[0]

        beam_width = max(top_paths, beam_width)

        for i in range(predictions.shape[1]):
            progbar.update(i)

            inputs = tf.transpose(predictions[:, i, :, :], perm=[1, 0, 2])
            decoded, log_probabilities = tf.nn.ctc_beam_search_decoder(inputs=inputs,
                                                                       sequence_length=sequence_length,
                                                                       beam_width=beam_width,
                                                                       top_paths=top_paths)

            decoded_pads = []
            for j in range(len(decoded)):
                sparse_decoded = tf.sparse.to_dense(decoded[j], default_value=-1)
                paddings = [[0, 0], [0, predictions.shape[2] - tf.reduce_max(tf.shape(sparse_decoded)[1])]]
                decoded_pads.append(tf.pad(sparse_decoded, paddings=paddings, constant_values=-1))

            decoded_paths.append(decoded_pads)
            probabilities_list.append(tf.exp(log_probabilities))

            progbar.update(i + 1)

        predictions = np.transpose(tf.stack(decoded_paths, axis=1), (2, 0, 1, 3))
        probabilities = np.transpose(tf.stack(probabilities_list, axis=1), (0, 2, 1))

        if tokenizer is not None:
            predictions = np.array([[tokenizer.decode_text(top_path) for top_path in item]
                                   for item in predictions], dtype=object)

        return predictions, probabilities

    def ctc_evaluate(self, x, steps, predictions, verbose=1):
        """
        Evaluate CTC predictions on the given data.

        Parameters
        ----------
        x : Dataset generator
            Input data for evaluation.
        steps : int
            Number of steps for evaluation.
        predictions : np.ndarray
            Predictions to be evaluated.
        verbose : int, optional
            Verbosity level.

        Returns
        -------
        tuple
            Metrics and evaluations.
        """

        progbar = tf.keras.utils.Progbar(target=steps, unit_name='evaluate', verbose=verbose)
        batch_index = 0

        metrics = {'cer': [], 'wer': []}
        evaluations = []

        for i in range(steps):
            progbar.update(i)

            _, y_true = next(x)
            batch_size = len(y_true)

            y_pred = predictions[batch_index:batch_index + batch_size]

            for true_label, pred_label in zip(y_true, y_pred):
                pattern = f'([{re.escape(string.punctuation)}])'
                true_label = ' '.join(re.sub(pattern, r' \1 ', true_label.replace('\n', ' ')).split()).strip()

                local_evaluation = {'ground_truth': true_label, 'top_paths': []}

                for _, top_path in enumerate(pred_label):
                    top_path = ' '.join(re.sub(pattern, r' \1 ', top_path.replace('\n', ' ')).split()).strip()

                    distance = editdistance.eval(list(true_label), list(top_path))
                    character_error_rate = distance / max(len(true_label), len(top_path))

                    distance = editdistance.eval(true_label.split(), top_path.split())
                    word_error_rate = distance / max(len(true_label.split()), len(top_path.split()))

                    metrics['cer'].append(character_error_rate)
                    metrics['wer'].append(word_error_rate)

                    local_evaluation['top_paths'].append(top_path)

                evaluations.append(local_evaluation)

            batch_index += batch_size
            progbar.update(i + 1)

        metrics = {k: np.mean(metrics[k]) for k in metrics}

        return metrics, evaluations
