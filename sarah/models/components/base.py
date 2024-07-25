import os
import re
import string
import numpy as np
import editdistance
import tensorflow as tf

from sarah.models.components.losses import CTCLoss
from sarah.models.components.losses import CTXLoss
from sarah.models.components.losses import BetaVAELoss
from sarah.models.components.metrics import EditDistance
from sarah.models.components.metrics import KernelInceptionDistance


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
            model = getattr(self, name, None)

            if model is None:
                continue

            trainable_count = sum([tf.size(x).numpy() for x in model.trainable_variables])
            non_trainable_count = sum([tf.size(x).numpy() for x in model.non_trainable_variables])
            total_count = trainable_count + non_trainable_count

            info += "\n" + "-" * width
            info += f"\n{'Model':<{pad}}: {model.name}"
            info += "\n" + "-" * width
            info += f"\n{'Total params':<{pad}}: {total_count:,}"
            info += f"\n{'Trainable params':<{pad}}: {trainable_count:,}"
            info += f"\n{'Non-trainable params':<{pad}}: {non_trainable_count:,}"
            info += f"\n{'Size (MB)':<{pad}}: {(total_count*4) / (1024**2):,.2f}"

        return info

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
            model = getattr(self, name, None)

            if model is None:
                continue

            model.summary(print_fn=lambda x: info.append(x))

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
            model = getattr(self, name, None)
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
            model = getattr(self, name, None)
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
                 style_encoder=None,
                 generator=None,
                 synthetic_data_ratio=0.99,
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
        style_encoder : StyleEncoder instance
            StyleEncoder model for encoding extracted style features.
        generator : Generator instance
            Generator model for image generation.
        synthetic_data_ratio : float, optional
            Probability to use synthetic data.
        seed : int, optional
            Seed for random shuffle.
        **kwargs : dict
            Additional keyword arguments.
        """

        super().__init__(**kwargs)

        seed = seed or 0
        tf.keras.utils.set_random_seed(seed)

        self.image_shape = image_shape
        self.lexical_shape = lexical_shape
        self.synthetic_data_ratio = synthetic_data_ratio
        self.seed = seed

        self.style_encoder = style_encoder
        self.generator = generator
        self.recognition = None

        self.names = [
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

        if self.generator and self.style_encoder:
            if np.random.random() <= self.synthetic_data_ratio:
                images, texts = image_data, aug_text_data

                latent_data = self.style_encoder(images, training=False)

                if isinstance(latent_data, list):
                    latent_data = latent_data[0]

                images = self.generator([latent_data, texts], training=False)

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

        x = np.log(x + 1e-7)
        x = x.transpose((0, 2, 1, 3))

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
                inputs = tf.transpose(batch[:, i, :, :], perm=[1, 0, 2])
                decoded, log_probabilities = tf.nn.ctc_beam_search_decoder(inputs=inputs,
                                                                           sequence_length=sequence_length,
                                                                           beam_width=beam_width,
                                                                           top_paths=top_paths)

                decoded_pads = []
                for j in range(len(decoded)):
                    decoded[j] = tf.sparse.to_dense(decoded[j], default_value=-1)
                    paddings = [[0, 0], [0, batch.shape[2] - tf.reduce_max(tf.shape(decoded[j])[1])]]
                    decoded_pads.append(tf.pad(decoded[j], paddings=paddings, constant_values=-1))

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

        batch_index = 0
        for step in range(steps):
            progbar.update(step)

            _, (image_data, text_data, writer_data) = next(y)
            batch_size = len(text_data)

            pred_data = x[batch_index:batch_index + batch_size]
            prob_data = probabilities[batch_index:batch_index + batch_size]
            batch_index += batch_size

            pattern = f'([{re.escape(string.punctuation)}])'

            for image_path, text_true, writer_id, text_pred, prob_pred in \
                    zip(image_data, text_data, writer_data, pred_data, prob_data):

                local_evaluation = {
                    'writer': writer_id,
                    'image': image_path,
                    'text': text_true,
                    'predictions': [],
                }

                gt = ' '.join(re.sub(pattern, r' \1 ', text_true.replace('\n', ' ').lower()).split())

                for i, top_path in enumerate(text_pred):
                    pd = ' '.join(re.sub(pattern, r' \1 ', top_path.replace('\n', ' ').lower()).split())

                    cer_distance = editdistance.eval(list(gt), list(pd))
                    cer = cer_distance / max(len(gt), len(pd))

                    wer_distance = editdistance.eval(gt.split(), pd.split())
                    wer = wer_distance / max(len(gt.split()), len(pd.split()))

                    metrics['cer'].append(cer)
                    metrics['wer'].append(wer)

                    local_evaluation['predictions'].append({
                        'probability': prob_pred if prob_pred is None else prob_pred[i],
                        'cer': cer,
                        'wer': wer,
                        'text': top_path,
                    })

                evaluations.append(local_evaluation)

            progbar.update(step + 1)

        evaluations = sorted(evaluations, key=lambda x: x['predictions'][0]['cer'])
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
                 seed=None,
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
        seed : int, optional
            Seed for random shuffle.
        **kwargs : dict
            Additional keyword arguments.
        """

        super().__init__(**kwargs)

        seed = seed or 0
        tf.keras.utils.set_random_seed(seed)

        self.image_shape = image_shape
        self.lexical_shape = lexical_shape
        self.writers_shape = writers_shape
        self.seed = seed

        self.recognition = None
        self.identification = None
        self.style_encoder = None
        self.generator = None
        self.discriminator = None
        self.patch_discriminator = None

        self.discriminator_steps = discriminator_steps
        self.generator_steps = generator_steps
        self.global_steps = tf.Variable(0, dtype=tf.int64)

        self.names = [
            'recognition',
            'identification',
            'style_encoder',
            'generator',
            'discriminator',
            'patch_discriminator',
        ]

        self.bva_loss = BetaVAELoss()
        self.cls_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        self.ctc_loss = CTCLoss()
        self.ctx_loss = CTXLoss()

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

        latent_data = self.style_encoder(image_data)

        if isinstance(latent_data, list):
            latent_data = latent_data[0]

        generated_images = self.generator([latent_data, text_data])

        self.kid.update_state(image_data, generated_images)

        return {
            self.kid.name: self.kid.result(),
        }

    def call(self, x_data, training=None):
        """
        Processes input images and text through the style encoder,
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
            latent_data = self.style_encoder(image_data, training=training)

            if isinstance(latent_data, list):
                latent_data = latent_data[0]

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

        batch_index = 0
        for step in range(steps):
            progbar.update(step)

            _, (image_true_data, _, _) = next(y)
            batch_size = len(image_true_data)

            image_pred_data = x[batch_index:batch_index + batch_size]
            batch_index += batch_size

            kid.update_state(image_true_data, image_pred_data)
            metrics['kid'].append(kid.result())

            evaluations.extend(list(zip(image_true_data, image_pred_data)))

            progbar.update(step + 1)

        metrics = {k: np.mean(metrics[k], dtype=float) for k in metrics}

        return metrics, evaluations
