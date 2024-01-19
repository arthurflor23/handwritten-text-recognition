import os
import re
import string
import random
import numpy as np
import editdistance
import tensorflow as tf

from graphite.models.components.losses import CTCLoss
from graphite.models.components.losses import CXLoss
from graphite.models.components.losses import BetaVAELoss
from graphite.models.components.metrics import EditDistance
from graphite.models.components.metrics import KernelInceptionDistance


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

        for name in getattr(self, 'names', ['model']):
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

            for name in getattr(self, 'names', ['model']):
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

        for name in getattr(self, 'names', ['model']):
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

        for name in getattr(self, 'names', ['model']):
            model = getattr(self, name, None)
            modelpath = filepath.replace('<model>', name)

            if model is not None:
                model.trainable = True
                model.save_weights(filepath=modelpath,
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

        for name in getattr(self, 'names', ['model']):
            model = getattr(self, name, None)
            modelpath = filepath.replace('<model>', name)

            if model is not None and os.path.isfile(modelpath):
                model.built = True
                model.load_weights(filepath=modelpath,
                                   by_name=by_name,
                                   skip_mismatch=skip_mismatch,
                                   options=options)

    def get_batch_lens(self, batch, pad_value=None):
        """
        Calculate tensor lengths in batch.

        Parameters
        ----------
        batch : tf.Tensor
            A batch of tensors.
        pad_value : float, optional
            Value used for padding.

        Returns
        -------
        tf.Tensor
            Batch of lengths for each tensor.
        """

        shape = tf.shape(batch)

        if pad_value is None:
            batch_lens = tf.fill((shape[0],), shape[1])
        else:
            reduce_axis = list(range(2, len(batch.shape)))
            batch_mean = tf.reduce_mean(batch, axis=reduce_axis)

            data_reversed = tf.reverse(batch_mean, axis=[1])
            padding_mask = tf.equal(data_reversed, tf.cast(pad_value, dtype=data_reversed.dtype))

            lengths = tf.argmax(tf.cast(~padding_mask, dtype=tf.int32), axis=1, output_type=shape.dtype)
            batch_lens = tf.where(tf.equal(lengths, 0), shape[1], shape[1] - lengths)

        batch_lens = tf.stop_gradient(batch_lens, name='batch_lens')

        return batch_lens

    def set_batch_mask(self, batch, batch_lens=None, reduce_scale=None, reduce_norm=False):
        """
        Apply a mask to a batch of tensors based on their lengths.

        Parameters
        ----------
        batch : tf.Tensor
            A batch of tensors.
        batch_lens : tf.Tensor, optional
            A tensor containing the lengths for each tensor in the batch.
        reduce_scale : float, optional
            A scaling factor to reduce the lengths.
        reduce_norm : bool, optional
            Whether to normalize the batch after masking.

        Returns
        -------
        tf.Tensor
            The batch with the mask applied.
        """

        if batch_lens is None:
            return batch

        if reduce_scale is not None:
            batch_lens_dtype = batch_lens.dtype
            batch_lens = tf.cast(batch_lens, dtype=tf.float32)
            reduce_scale = tf.cast(reduce_scale, dtype=tf.float32)

            batch_lens = tf.math.ceil(tf.math.divide(batch_lens, reduce_scale + 1e-7))
            batch_lens = tf.cast(batch_lens, dtype=batch_lens_dtype)

        batch_shape = batch.get_shape()
        mask = tf.sequence_mask(batch_lens, maxlen=batch_shape[1], dtype=batch.dtype)

        for _ in range(len(batch_shape) - 2):
            mask = tf.expand_dims(mask, axis=-1)

        batch = tf.math.multiply(batch, tf.stop_gradient(mask), name='batch')

        if reduce_norm:
            batch_dtype = batch.dtype
            batch = tf.cast(tf.reduce_sum(batch, axis=-1), dtype=tf.float32)
            batch_lens = tf.cast(tf.expand_dims(batch_lens, axis=-1), dtype=tf.float32)

            batch = tf.math.divide(batch, batch_lens + 1e-7)
            batch = tf.cast(batch, dtype=batch_dtype)

        return batch


class BaseRecognitionModel(BaseModel):
    """
    BaseRecognitionModel extends BaseModel to provide additional
        functionalities to synthesis and recognition models.
    """

    def __init__(self,
                 image_shape,
                 lexical_shape,
                 style_backbone=None,
                 style_encoder=None,
                 generator=None,
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
        style_backbone : StyleBackbone instance
            StyleBackbone model for extracting style patterns from images.
        style_encoder : StyleEncoder instance
            StyleEncoder model for encoding extracted style features.
        generator : Generator instance
            Generator model for image generation.
        synthesis_ratio : float, optional
            Probability to use synthetic data.
        **kwargs : dict
            Additional keyword arguments.
        """

        super().__init__(**kwargs)

        self.image_shape = image_shape
        self.lexical_shape = lexical_shape
        self.synthesis_ratio = synthesis_ratio

        self.style_backbone = style_backbone
        self.style_encoder = style_encoder
        self.generator = generator
        self.recognition = None

        self.names = [
            'style_backbone',
            'style_encoder',
            'generator',
            'recognition',
        ]

        self.ctc_loss = CTCLoss()
        self.edit_distance = EditDistance()

        self.monitor = self.edit_distance.name

        self.build_model()
        self.built = True

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

        (aug_image_data, aug_text_data), (image_data, text_data, _) = input_data

        images, texts = aug_image_data, text_data

        if self.generator and self.style_backbone and self.style_encoder:
            if random.random() <= self.synthesis_ratio:
                images, texts = image_data, aug_text_data

                features_data, _ = self.style_backbone(images, training=False)
                latent_inputs, _, _ = self.style_encoder(features_data, training=False)
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

        _, (image_data, text_data, _) = input_data

        ctc_logits = self.recognition(image_data)

        ctc_loss = self.ctc_loss(text_data, ctc_logits)
        self.edit_distance.update_state(text_data, ctc_logits)

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

        image_data = x_data[0] if isinstance(x_data, tuple) else x_data
        ctc_logits = self.recognition(image_data, training=training)

        return ctc_logits

    def ctc_decoder(self,
                    x,
                    steps,
                    top_paths=1,
                    beam_width=32,
                    tokenizer=None,
                    verbose=1):
        """
        Decode CTC predictions using beam search.

        Parameters
        ----------
        x : numpy.ndarray
            CTC predictions to be decoded.
        steps : int
            Number of steps for decode.
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

        progbar = tf.keras.utils.Progbar(target=steps, unit_name='decode', verbose=verbose)

        beam_width = max(top_paths, beam_width)
        batch_size = int(np.ceil(x.shape[0] / steps))

        x = np.log(x + 1e-7)
        predictions, probabilities = [], []

        for step in range(steps):
            progbar.update(step)

            start = step * batch_size
            end = start + batch_size

            batch = x[start:end, :, :, :]

            top_path_decoded, top_path_probabilities = [], []
            sequence_length = [batch.shape[2]] * batch.shape[0]

            for i in range(batch.shape[1]):
                inputs = tf.transpose(batch[:, i, :, :], perm=[1, 0, 2])
                decoded, log_probabilities = tf.nn.ctc_beam_search_decoder(inputs=inputs,
                                                                           sequence_length=sequence_length,
                                                                           beam_width=beam_width,
                                                                           top_paths=top_paths)

                decoded_pads = []
                for j in range(len(decoded)):
                    sparse_decoded = tf.sparse.to_dense(decoded[j], default_value=-1)
                    paddings = [[0, 0], [0, batch.shape[2] - tf.reduce_max(tf.shape(sparse_decoded)[1])]]
                    decoded_pads.append(tf.pad(sparse_decoded, paddings=paddings, constant_values=-1))

                top_path_decoded.append(decoded_pads)
                top_path_probabilities.append(tf.exp(log_probabilities))

            batch_decoded = np.transpose(tf.stack(top_path_decoded, axis=1), (2, 0, 1, 3))
            batch_probabilities = np.transpose(tf.stack(top_path_probabilities, axis=1), (0, 2, 1))

            if tokenizer is not None:
                batch_decoded = [[tokenizer.decode_text(top_path) for top_path in x] for x in batch_decoded]

            predictions.append(batch_decoded)
            probabilities.append(batch_probabilities)

            progbar.update(step + 1)

        predictions = np.concatenate(predictions, axis=0)
        probabilities = np.concatenate(probabilities, axis=0)
        probabilities = probabilities.squeeze(axis=-1).astype(float)

        return predictions, probabilities

    def ctc_evaluator(self, x, y, steps, probabilities=None, verbose=1):
        """
        Evaluate CTC predictions on the given data.

        Parameters
        ----------
        x : np.ndarray
            Predictions to be evaluated.
        y : Dataset generator
            Label data for evaluation.
        steps : int
            Number of steps for evaluation.
        probabilities : numpy.ndarray, optional
            Corresponding probabilities of the predictions.
        verbose : int, optional
            Verbosity level.

        Returns
        -------
        tuple
            Metrics and evaluations.
        """

        progbar = tf.keras.utils.Progbar(target=steps, unit_name='evaluate', verbose=verbose)

        metrics = {'cer': [], 'wer': []}
        evaluations = []

        if probabilities is None:
            probabilities = [None] * len(x)

        for step in range(steps):
            progbar.update(step)

            _, y_data = next(y)
            _, text_true_data, _ = y_data

            batch_size = len(text_true_data)

            start = step * batch_size
            end = start + batch_size

            text_pred_data = x[start:end]
            prob_pred_data = probabilities[start:end]

            pattern = f'([{re.escape(string.punctuation)}])'

            for text_true, text_pred, prob_pred in zip(text_true_data, text_pred_data, prob_pred_data):
                text_true = ' '.join(re.sub(pattern, r' \1 ', text_true.replace('\n', ' ')).split())
                local_evaluation = {'ground_truth': text_true}

                for i, top_path in enumerate(text_pred):
                    top_path = ' '.join(re.sub(pattern, r' \1 ', top_path.replace('\n', ' ')).split())

                    cer_distance = editdistance.eval(list(text_true), list(top_path))
                    cer = cer_distance / max(len(text_true), len(top_path))

                    wer_distance = editdistance.eval(text_true.split(), top_path.split())
                    wer = wer_distance / max(len(text_true.split()), len(top_path.split()))

                    metrics['cer'].append(cer)
                    metrics['wer'].append(wer)

                    local_evaluation[f"top_path_{i+1}"] = {
                        'probability': prob_pred if prob_pred is None else prob_pred[i],
                        'prediction': top_path,
                    }

                evaluations.append(local_evaluation)

            progbar.update(step + 1)

        metrics = {k: np.mean(metrics[k], dtype=float) for k in metrics}

        return metrics, evaluations


class BaseSynthesisModel(BaseModel):
    """
    BaseSynthesisModel extends BaseModel to provide additional
        functionalities to synthesis models.
    """

    def __init__(self,
                 image_shape,
                 lexical_shape,
                 writers_shape,
                 discriminator_steps=1,
                 generator_steps=1,
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
        discriminator_steps : int, optional
            The repetition of steps for discriminator training.
        generator_steps : int, optional
            The skipping steps for generator training.
        **kwargs : dict
            Additional keyword arguments.
        """

        super().__init__(**kwargs)

        self.image_shape = image_shape
        self.lexical_shape = lexical_shape
        self.writers_shape = writers_shape

        self.discriminator = None
        self.patch_discriminator = None
        self.style_backbone = None
        self.identification = None
        self.recognition = None
        self.style_encoder = None
        self.generator = None

        self.discriminator_steps = discriminator_steps
        self.generator_steps = generator_steps
        self.global_steps = tf.Variable(0, dtype=tf.int64)

        self.names = [
            'discriminator',
            'patch_discriminator',
            'style_backbone',
            'identification',
            'recognition',
            'style_encoder',
            'generator',
        ]

        self.bv_loss = BetaVAELoss()
        self.cls_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        self.ctc_loss = CTCLoss()
        self.cx_loss = CXLoss()
        self.kid = KernelInceptionDistance(scale=127.5, offset=127.5)

        self.monitor = self.kid.name

        self.build_model()
        self.built = True

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

        _, (image_data, text_data, _) = input_data

        features_data, _ = self.style_backbone(image_data)
        latent_data, _, _ = self.style_encoder(features_data)
        generated_images = self.generator([latent_data, text_data])

        self.kid.update_state(image_data, generated_images)

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

        image_data, text_data = x_data

        if tf.math.reduce_all(tf.equal(image_data, -1.)):
            latent_data = tf.random.normal(shape=(len(text_data), self.style_encoder.latent_dim))
        else:
            features_data, _ = self.style_backbone(image_data, training=training)
            latent_data, _, _ = self.style_encoder(features_data, training=training)

        generated_images = self.generator([latent_data, text_data], training=training)

        return generated_images

    def image_evaluator(self, x, y, steps, verbose=1):
        """
        Evaluate generator predictions on the given data.

        Parameters
        ----------
        x : np.ndarray
            Predictions to be evaluated.
        y : Dataset generator
            Label data for evaluation.
        steps : int
            Number of steps for evaluation.
        verbose : int, optional
            Verbosity level.

        Returns
        -------
        tuple
            Metrics and evaluations.
        """

        progbar = tf.keras.utils.Progbar(target=steps, unit_name='evaluate', verbose=verbose)

        metrics = {'kid': []}
        evaluations = []

        kid = KernelInceptionDistance(scale=1.0, offset=0.0)

        for step in range(steps):
            progbar.update(step)

            _, y_data = next(y)
            image_true_data, _, _ = y_data

            batch_size = len(image_true_data)

            start = step * batch_size
            end = start + batch_size

            image_pred_data = x[start:end]

            kid.update_state(image_true_data, image_pred_data)
            metrics['kid'].append(kid.result())

            evaluations.extend(list(zip(image_true_data, image_pred_data)))

            progbar.update(step + 1)

        metrics = {k: np.mean(metrics[k], dtype=float) for k in metrics}

        return metrics, evaluations


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
