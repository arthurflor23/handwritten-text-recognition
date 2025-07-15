import os
import re
import string
import numpy as np
import editdistance
import tensorflow as tf

from sarah.models.components.losses import CTCLoss
from sarah.models.components.losses import CTXLoss
from sarah.models.components.losses import KLDivergence
from sarah.models.components.metrics import EditDistance
from sarah.models.components.metrics import KernelInceptionDistance
from sarah.models.components.utils import MeasureTracker


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

        pad, width = 25, 60
        info = "=" * width
        info += f"\n{self.__class__.__name__.center(width)}"

        for name in getattr(self, 'names', ['model']):
            model = self.get_model_by_name(name)

            if model is None:
                continue

            trainable_variables = [tf.size(x) for x in model.trainable_variables if 'seed' not in x.path]
            non_trainable_variables = [tf.size(x) for x in model.non_trainable_variables if 'seed' not in x.path]

            trainable_count = sum(trainable_variables)
            non_trainable_count = sum(non_trainable_variables)

            total_count = trainable_count + non_trainable_count

            info += "\n" + "-" * width
            info += f"\n{'Model':<{pad}}: {model.name}"
            info += "\n" + "-" * width
            info += f"\n{'Total params':<{pad}}: {total_count:,}"
            info += f"\n{'Trainable params':<{pad}}: {trainable_count:,}"
            info += f"\n{'Non-trainable params':<{pad}}: {non_trainable_count:,}"
            info += f"\n{'Size (MB)':<{pad}}: {(total_count*4) / (1024**2):,.2f}"

        return info

    def call(self, inputs, training=False):
        """
        Perform a forward pass on the model with the given inputs.

        Parameters
        ----------
        inputs : tf.Tensor
            Input data to be passed through the model.
        training : bool, optional
            Whether the call is for training or inference.

        Returns
        -------
        output
            The output of the model after processing the inputs.
        """

        if hasattr(self, 'model'):
            return self.model(inputs, training=training)

    def get_model_by_name(self, name):
        """
        Retrieve the attribute model by name.

        Parameters
        ----------
        name : str
            Name of the model.

        Returns
        -------
        str
            The attribute model.
        """

        return getattr(self, name,
                       getattr(self, f"{name}_encoder",
                               getattr(self, f"{name}_decoder", None)))

    def get_summary(self):
        """
        Provides summary of model architectures.

        Returns
        -------
        str
            Formatted model architectures.
        """

        info = []

        for name in getattr(self, 'names', ['model']):
            model = self.get_model_by_name(name)
            model = getattr(model, name, model)

            if model is None:
                continue

            model.summary(print_fn=lambda r: info.append(r), expand_nested=True)

        return "\n".join(info)

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
                model = self.get_model_by_name(name)

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
            model = self.get_model_by_name(name)

            if model is not None:
                model.set_weights(weights[name])

    def save_weights(self, filepath, overwrite=True):
        """
        Save the weights of the submodels.

        Parameters
        ----------
        filepath : str
            Filepath for saving the weights.
        overwrite : bool, optional
            Whether to overwrite the existing file.
        """

        for name in getattr(self, 'names', ['model']):
            model = self.get_model_by_name(name)
            modelpath = filepath.replace('<model>', name)

            if model is not None:
                model.trainable = True
                model.save_weights(filepath=modelpath, overwrite=overwrite)

    def load_weights(self, filepath, skip_mismatch=True):
        """
        Load the weights for the submodels.

        Parameters
        ----------
        filepath : str
            Filepath for loading the weights.
        skip_mismatch : bool, optional
            Skip loading of layers where there is a mismatch in the number of weights.
        """

        for name in getattr(self, 'names', ['model']):
            model = self.get_model_by_name(name)
            modelpath = filepath.replace('<model>', name)

            if model is not None and os.path.isfile(modelpath):
                model.built = True
                model.load_weights(filepath=modelpath, skip_mismatch=skip_mismatch)


class BaseRecognitionModel(BaseModel):
    """
    BaseRecognitionModel extends BaseModel to provide additional
        functionalities to synthesis and recognition models.
    """

    def __init__(self,
                 image_shape,
                 lexical_shape,
                 writer_encoder=None,
                 style_encoder=None,
                 generator=None,
                 synthesis_probability=1.0,
                 seed=None,
                 **kwargs):
        """
        Initializes the synthesis and recognition model.

        Parameters
        ----------
        image_shape : tuple or list
            The shape of the input images.
        lexical_shape : tuple or list
            The shape of the lexical input.
        writer_encoder : Style extractor instance
            Style extractor model for features extraction.
        style_encoder : StyleEncoder instance
            StyleEncoder model for encoding extracted style features.
        generator : Generator instance
            Generator model for image generation.
        synthesis_probability : float, optional
            Synthetic data probability.
        seed : int, optional
            Seed for random shuffle.
        **kwargs : dict
            Additional arguments.
        """

        super().__init__(**kwargs)

        tf.keras.utils.set_random_seed(seed)

        self.image_shape = image_shape
        self.lexical_shape = lexical_shape
        self.synthesis_probability = synthesis_probability
        self.seed = seed

        self.writer_encoder = writer_encoder
        self.style_encoder = style_encoder
        self.generator = generator
        self.recognition = None

        self.names = [
            'writer_encoder',
            'style_encoder',
            'generator',
            'recognition',
        ]

        self.ctc_loss = CTCLoss()
        self.edit_distance = EditDistance()

        self.measure_tracker = MeasureTracker()
        self.monitor = f"val_{self.edit_distance.name}"

        self.build_model()
        self.built = True

    def get_config(self):
        """
        Return the configuration of the model.

        Returns
        -------
        dict
            Configuration dictionary.
        """

        config = super().get_config()

        config.update({
            'image_shape': self.image_shape,
            'lexical_shape': self.lexical_shape,
        })

        return config

    def train_step(self, input_data):
        """
        Executes a training step.

        Parameters
        ----------
        input_data : list or tuple
            Batch of (x_data, y_data).

        Returns
        -------
        dict
            Training metrics and losses.
        """

        x_data, y_data = input_data

        aug_image_data, aug_text_data, _, aug_mask_data = x_data
        _, text_data, _, mask_data = y_data

        images, texts = [aug_image_data], [text_data]

        if self.writer_encoder and self.style_encoder and self.generator and \
                np.random.random() <= self.synthesis_probability:

            # original images and original texts
            features_data = self.writer_encoder(aug_image_data, training=False)
            features_data = features_data[0] if isinstance(features_data, list) else features_data

            latent = self.style_encoder(features_data, training=False)
            latent = latent[0] if isinstance(latent, list) else latent

            real_real_images = self.generator([text_data, latent, mask_data], training=False)

            # original images and fake texts
            fake_real_images = self.generator([aug_text_data, latent, aug_mask_data], training=False)

            # fake images and original texts
            latent_shape = (tf.shape(aug_image_data)[0], self.style_encoder.latent_dim)
            latent = tf.random.normal(shape=latent_shape, seed=self.seed)

            real_fake_images = self.generator([text_data, latent, mask_data], training=False)

            # fake images and fake texts
            fake_fake_images = self.generator([aug_text_data, latent, aug_mask_data], training=False)

            images.extend([real_real_images, fake_real_images, real_fake_images, fake_fake_images])
            texts.extend([text_data, aug_text_data, text_data, aug_text_data])

        for image, text in zip(images, texts):
            with tf.GradientTape() as r_tape:
                ctc_logits = self.recognition(image, training=True)
                ctc_loss = self.ctc_loss(text, ctc_logits)

            r_gradients = r_tape.gradient(ctc_loss, self.recognition.trainable_weights)
            self.optimizer.apply_gradients(zip(r_gradients, self.recognition.trainable_weights))

            self.edit_distance.update_state(text, ctc_logits)

            self.measure_tracker.update({
                self.ctc_loss.name: ctc_loss,
                self.edit_distance.name: self.edit_distance.result(),
            })

        return self.measure_tracker.result()

    def test_step(self, input_data):
        """
        Executes a testing step.

        Parameters
        ----------
        input_data : list or tuple
            Batch of (x_data, y_data).

        Returns
        -------
        dict
            Training metrics and losses.
        """

        _, (image_data, text_data, _, _) = input_data

        ctc_logits = self.recognition(image_data)
        ctc_loss = self.ctc_loss(text_data, ctc_logits)

        self.edit_distance.update_state(text_data, ctc_logits)

        self.measure_tracker.update({
            f"val_{self.ctc_loss.name}": ctc_loss,
            f"val_{self.edit_distance.name}": self.edit_distance.result(),
        })

        return self.measure_tracker.result(val_only=True)

    def call(self, x_data, training=False):
        """
        Processes input handwritten images.

        Parameters
        ----------
        x_data : list or tuple
            Input batch (x_data).
        training : bool, optional
            Whether the call is for training or inference.

        Returns
        -------
        tf.Tensor
            Generated images.
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
        predictions, probabilities = [], []

        batch_index = 0
        batch_size = int(np.ceil(len(x) / steps))

        for step in range(steps):
            progbar.update(step)

            batch = x[batch_index:batch_index + batch_size, :, :, :]
            batch_index += batch_size

            top_path_decoded, top_path_probabilities = [], []
            sequence_length = [batch.shape[2]] * batch.shape[0]

            for i in range(batch.shape[1]):
                decoded, log_probabilities = tf.keras.ops.ctc_decode(inputs=batch[:, i, :, :],
                                                                     sequence_lengths=sequence_length,
                                                                     strategy='beam_search',
                                                                     beam_width=beam_width,
                                                                     top_paths=top_paths,
                                                                     merge_repeated=True,
                                                                     mask_index=0)

                top_path_decoded.append(decoded)
                top_path_probabilities.append(tf.exp(log_probabilities))

            batch_decoded = np.transpose(tf.stack(top_path_decoded, axis=1), axes=(2, 0, 1, 3))
            batch_probabilities = np.transpose(tf.stack(top_path_probabilities, axis=1), axes=(0, 2, 1))

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

        def _standardize(text):
            pattern = f'([{re.escape(string.punctuation)}])'
            return ' '.join(re.sub(pattern, r' \1 ', text.replace('\n', ' ').lower()).split())

        progbar = tf.keras.utils.Progbar(target=steps, unit_name='evaluate', verbose=verbose)

        metrics = {'cer': [], 'wer': []}
        evaluations = []

        if probabilities is None:
            probabilities = [None] * len(x)

        batch_index = 0
        for step in range(steps):
            progbar.update(step)

            _, (image_data, text_data, writer_data, _) = next(y)
            batch_size = len(text_data)

            pred_data = x[batch_index:batch_index + batch_size]
            prob_data = probabilities[batch_index:batch_index + batch_size]
            batch_index += batch_size

            for i, (image_path, ground_truth, writer_id, text_pred, prob_pred) in \
                    enumerate(zip(image_data, text_data, writer_data, pred_data, prob_data)):

                local_evaluation = {
                    'index': (batch_index - batch_size) + (i + 1),
                    'writer': writer_id,
                    'image': image_path,
                    'text': ground_truth,
                    'predictions': [],
                }

                gt = _standardize(text=ground_truth)

                char_length = max(1, len(gt))
                word_length = max(1, len(gt.split()))

                for j, predict in enumerate(text_pred):
                    pd = _standardize(text=predict)

                    cer = editdistance.eval(list(gt), list(pd)) / char_length
                    wer = editdistance.eval(gt.split(), pd.split()) / word_length

                    metrics['cer'].append(cer)
                    metrics['wer'].append(wer)

                    local_evaluation['predictions'].append({
                        'index': (j + 1),
                        'text': predict,
                        'probability': prob_pred if prob_pred is None else prob_pred[j],
                        'cer': cer,
                        'wer': wer,
                    })

                evaluations.append(local_evaluation)

            progbar.update(step + 1)

        metrics = {k: float(np.mean(metrics[k])) for k in metrics}

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
                 seed=None,
                 **kwargs):
        """
        Initialize synthesis model.

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
        seed : int, optional
            Seed for random shuffle.
        **kwargs : dict
            Additional arguments.
        """

        super().__init__(**kwargs)

        tf.keras.utils.set_random_seed(seed)

        self.image_shape = image_shape
        self.lexical_shape = lexical_shape
        self.writers_shape = writers_shape
        self.seed = seed

        self.recognition = None
        self.writer_encoder = None
        self.writer_decoder = None
        self.style_encoder = None
        self.generator = None
        self.discriminator = None
        self.patch_discriminator = None

        self.discriminator_steps = discriminator_steps
        self.generator_steps = generator_steps

        self.global_step = tf.keras.Variable(name='global_step',
                                             initializer=0,
                                             dtype=tf.int64,
                                             trainable=False)

        self.names = [
            'recognition',
            'writer_encoder',
            'writer_decoder',
            'style_encoder',
            'generator',
            'discriminator',
            'patch_discriminator',
        ]

        self.sce_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, name='sce_loss')
        self.ctc_loss = CTCLoss()
        self.ctx_loss = CTXLoss()
        self.kid = KernelInceptionDistance(image_shape=self.image_shape)
        self.kld_loss = KLDivergence()

        self.measure_tracker = MeasureTracker()
        self.monitor = f"d_{self.ctc_loss.name}"

        self.build_model()
        self.built = True

    def get_config(self):
        """
        Return the configuration of the model.

        Returns
        -------
        dict
            Configuration dictionary.
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
        Executes a testing step.

        Parameters
        ----------
        input_data : list or tuple
            Batch of (x_data, y_data).

        Returns
        -------
        dict
            Training metrics and losses.
        """

        _, (image_data, text_data, _, mask_data) = input_data

        features_data = self.writer_encoder(image_data)
        features_data = features_data[0] if isinstance(features_data, list) else features_data

        latent_data = self.style_encoder(features_data)
        latent_data = latent_data[0] if isinstance(latent_data, list) else latent_data

        generated_images = self.generator([text_data, latent_data, mask_data])

        self.kid.update_state(image_data, generated_images)

        self.measure_tracker.update({
            f"val_{self.kid.name}": self.kid.result(),
        })

        return self.measure_tracker.result(val_only=True)

    def call(self, x_data, training=False):
        """
        Processes input handwritten images.

        Parameters
        ----------
        x_data : list or tuple
            Input batch (x_data).
        training : bool, optional
            Whether the call is for training or inference.

        Returns
        -------
        tf.Tensor
            Generated images.
        """

        image_data, text_data, _, mask_data = x_data

        def _random_latent():
            latent_shape = (tf.shape(image_data)[0], self.style_encoder.latent_dim)
            return tf.random.normal(shape=latent_shape, seed=self.seed)

        def _extract_latent():
            features_data = self.writer_encoder(image_data, training=training)
            features_data = features_data[0] if isinstance(features_data, list) else features_data

            latent_data = self.style_encoder(features_data, training=training)
            latent_data = latent_data[0] if isinstance(latent_data, list) else latent_data

            return latent_data

        latent_data = tf.cond(pred=tf.math.reduce_all(tf.equal(image_data, 1.)),
                              true_fn=_random_latent,
                              false_fn=_extract_latent)

        generated_images = self.generator([text_data, latent_data, mask_data], training=training)

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

        batch_index = 0
        for step in range(steps):
            progbar.update(step)

            _, (image_true_data, _, _, _) = next(y)
            batch_size = len(image_true_data)

            image_pred_data = x[batch_index:batch_index + batch_size]
            batch_index += batch_size

            kid.update_state(image_true_data, image_pred_data)
            metrics['kid'].append(kid.result())

            evaluations.extend(list(zip(image_true_data, image_pred_data)))

            progbar.update(step + 1)

        metrics = {k: float(np.mean(metrics[k])) for k in metrics}

        return metrics, evaluations


class BaseWriterIdentificationModel(BaseModel):
    """
    BaseWriterIdentificationModel extends BaseModel to provide additional
        functionalities to synthesis and writer identification models.
    """

    def __init__(self,
                 image_shape,
                 writers_shape,
                 writer_encoder=None,
                 style_encoder=None,
                 generator=None,
                 synthesis_probability=1.0,
                 return_features=False,
                 seed=None,
                 **kwargs):
        """
        Initializes the synthesis and writer identification model.

        Parameters
        ----------
        image_shape : tuple or list
            The shape of the input images.
        writers_shape : tuple or list
            The shape of the writers input.
        writer_encoder : Style extractor instance
            Style extractor model for features extraction.
        style_encoder : StyleEncoder instance
            StyleEncoder model for encoding extracted style features.
        generator : Generator instance
            Generator model for image generation.
        synthesis_probability : float, optional
            Synthetic data probability.
        return_features : bool, optional
            Whether to return the intermediate features.
        seed : int, optional
            Seed for random shuffle.
        **kwargs : dict
            Additional arguments.
        """

        super().__init__(**kwargs)

        tf.keras.utils.set_random_seed(seed)

        self.image_shape = image_shape
        self.writers_shape = writers_shape
        self.synthesis_probability = synthesis_probability
        self.return_features = return_features
        self.seed = seed

        self.writer_encoder = writer_encoder
        self.style_encoder = style_encoder
        self.generator = generator
        self.writer_identification = None

        self.names = [
            'writer_encoder',
            'style_encoder',
            'generator',
            'writer_identification',
        ]

        self.sce_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, name='sce_loss')

        self.measure_tracker = MeasureTracker()
        self.monitor = f"val_{self.sce_loss.name}"

        self.build_model()
        self.built = True

    def get_config(self):
        """
        Return the configuration of the model.

        Returns
        -------
        dict
            Configuration dictionary.
        """

        config = super().get_config()

        config.update({
            'image_shape': self.image_shape,
            'writers_shape': self.writers_shape,
        })

        return config

    def train_step(self, input_data):
        """
        Executes a training step.

        Parameters
        ----------
        input_data : list or tuple
            Batch of (x_data, y_data).

        Returns
        -------
        dict
            Training metrics and losses.
        """

        x_data, y_data = input_data

        aug_image_data, aug_text_data, _, aug_mask_data = x_data
        _, text_data, writer_data, mask_data = y_data

        images, writers = [aug_image_data], [writer_data]

        if self.writer_encoder and self.style_encoder and self.generator and \
                np.random.random() <= self.synthesis_probability:

            # original images and original texts
            features_data = self.writer_encoder(aug_image_data, training=False)
            features_data = features_data[0] if isinstance(features_data, list) else features_data

            latent = self.style_encoder(features_data, training=False)
            latent = latent[0] if isinstance(latent, list) else latent

            real_real_images = self.generator([text_data, latent, mask_data], training=False)

            # original images and fake texts
            fake_real_images = self.generator([aug_text_data, latent, aug_mask_data], training=False)

            images.extend([real_real_images, fake_real_images])
            writers.extend([writer_data, writer_data])

        for image, writer in zip(images, writers):
            with tf.GradientTape() as w_tape:
                wid_logits = self.writer_identification(image, training=True)
                sce_loss = self.sce_loss(writer, wid_logits)

            w_gradients = w_tape.gradient(sce_loss, self.writer_identification.trainable_weights)
            self.optimizer.apply_gradients(zip(w_gradients, self.writer_identification.trainable_weights))

            self.measure_tracker.update({
                self.sce_loss.name: sce_loss,
            })

        return self.measure_tracker.result()

    def test_step(self, input_data):
        """
        Executes a testing step.

        Parameters
        ----------
        input_data : list or tuple
            Batch of (x_data, y_data).

        Returns
        -------
        dict
            Training metrics and losses.
        """

        _, (image_data, _, writer_data, _) = input_data

        wid_logits = self.writer_identification(image_data)
        sce_loss = self.sce_loss(writer_data, wid_logits)

        self.measure_tracker.update({
            f"val_{self.sce_loss.name}": sce_loss,
        })

        return self.measure_tracker.result(val_only=True)

    def call(self, x_data, training=False):
        """
        Processes input handwritten images.

        Parameters
        ----------
        x_data : list or tuple
            Input batch (x_data).
        training : bool, optional
            Whether the call is for training or inference.

        Returns
        -------
        tf.Tensor
            Generated images.
        """

        image_data = x_data[0] if isinstance(x_data, tuple) else x_data
        wid_logits = self.writer_identification(image_data, training=training)

        return wid_logits

    def writer_evaluator(self, x, y, steps, verbose=1):
        """
        Evaluate writer predictions on the given data.

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

        metrics = {'accuracy': []}
        evaluations = []

        batch_index = 0
        for step in range(steps):
            progbar.update(step)

            _, (image_data, _, writer_data, _) = next(y)
            batch_size = len(image_data)

            y_pred = x[batch_index:batch_index + batch_size]
            batch_index += batch_size

            for i in range(batch_size):
                is_correct = int(writer_data[i] == y_pred[i])

                evaluations.append({
                    'index': (batch_index - batch_size) + (i + 1),
                    'image': image_data[i],
                    'writer': writer_data[i],
                    'writer_prediction': y_pred[i],
                    'is_correct': is_correct,
                })

                metrics['accuracy'].append(is_correct)

            progbar.update(step + 1)

        metrics = {k: float(np.mean(v)) for k, v in metrics.items()}

        return metrics, evaluations
