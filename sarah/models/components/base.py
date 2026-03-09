import os
import re
import string
import numpy as np
import Levenshtein
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

        pad, width = 25, 68

        module_name = self.__class__.__module__.split('.')[-1]
        class_name = self.__class__.__name__.center(width)

        info = "=" * width
        info += f"\n{class_name}"

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
            info += f"\n{'Module':<{pad}}: {module_name}"
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

        return (getattr(self, name, None) or
                getattr(self, f"{name}_encoder", None) or
                getattr(self, f"{name}_decoder", None))

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

    def unwrap_call_output(self, output):
        """
        Unwrap model output.

        If the output is a list or tuple, returns its last element.
            Otherwise, returns the output unchanged.

        Parameters
        ----------
        output : object
            Output from the model.

        Returns
        -------
        object
            Model output.
        """

        if isinstance(output, (list, tuple)):
            return output[-1]

        return output


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
                 return_features=False,
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
        style_encoder : Style encoder instance
            Style encoder model for encoding extracted style features.
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

        if seed is not None:
            tf.keras.utils.set_random_seed(seed)

        self.image_shape = image_shape
        self.lexical_shape = lexical_shape
        self.synthesis_probability = synthesis_probability
        self.return_features = return_features
        self.seed = seed

        self.writer_encoder = writer_encoder
        self.style_encoder = style_encoder
        self.generator = generator
        self.recognition = None

        self.global_step = tf.keras.Variable(name='global_step',
                                             initializer=0,
                                             dtype=tf.int64,
                                             trainable=False)

        self.names = [
            'segmentation_encoder',
            'writer_encoder',
            'style_encoder',
            'generator',
            'recognition',
        ]

        self.ctc_loss = CTCLoss()
        self.edit_distance = EditDistance()

        self.bce_loss = tf.keras.losses.BinaryCrossentropy(from_logits=False, name='bce_loss')
        self.dice_loss = tf.keras.losses.Dice(name='dice_loss')

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

    def build_input_data(self, input_data):
        """
        Builds image and text inputs for the model.
        Includes synthesized samples when synthesis is enabled.

        Parameters
        ----------
        input_data : list or tuple
            Model inputs and targets (x_data, y_data).

        Returns
        -------
        tuple
            Lists of images and texts.
        """

        x_data, y_data = input_data

        aug_image_data, aug_text_data, _, aug_mask_data, _ = x_data
        _, text_data, _, mask_data, _ = y_data

        images, texts = [aug_image_data], [text_data]

        if self.writer_encoder and self.style_encoder and self.generator and \
                np.random.random() <= self.synthesis_probability:

            # extract writer style from real images
            features_data = self.writer_encoder(aug_image_data, training=False)
            features_data = self.unwrap_call_output(features_data)

            latent = self.style_encoder(features_data, training=False)
            latent = self.unwrap_call_output(latent)

            # real images and real texts
            real_real_images = self.generator([text_data, latent, mask_data], training=False)

            # real images and fake texts
            fake_real_images = self.generator([aug_text_data, latent, aug_mask_data], training=False)

            # random style sampling
            random_latent_shape = (tf.shape(aug_image_data)[0], self.style_encoder.latent_dim)
            random_latent_data = tf.random.normal(shape=random_latent_shape)

            # fake images and real texts
            real_fake_images = self.generator([text_data, random_latent_data, mask_data], training=False)

            # fake images and fake texts
            fake_fake_images = self.generator([aug_text_data, random_latent_data, aug_mask_data], training=False)

            images.extend([real_real_images, fake_real_images, real_fake_images, fake_fake_images])
            texts.extend([text_data, aug_text_data, text_data, aug_text_data])

        return images, texts

    def train_step(self, input_data):
        """
        Executes a training step.

        Parameters
        ----------
        input_data : list or tuple
            Model inputs and targets (x_data, y_data).

        Returns
        -------
        dict
            Training metrics and losses.
        """

        images, texts = self.build_input_data(input_data)

        for image, text in zip(images, texts):
            with tf.GradientTape() as r_tape:
                ctc_logits = self.recognition(image, training=True)
                ctc_logits = self.unwrap_call_output(ctc_logits)

                ctc_loss = self.ctc_loss(text, ctc_logits)

            r_gradients = r_tape.gradient(ctc_loss, self.recognition.trainable_weights)
            self.optimizer.apply_gradients(zip(r_gradients, self.recognition.trainable_weights))

            self.edit_distance.update_state(text, ctc_logits)

            self.measure_tracker.update({
                self.ctc_loss.name: ctc_loss,
                self.edit_distance.name: self.edit_distance.result(),
            })

        self.global_step.assign_add(value=1)

        return self.measure_tracker.result()

    def test_step(self, input_data):
        """
        Executes a testing step.

        Parameters
        ----------
        input_data : list or tuple
            Model inputs and targets (x_data, y_data).

        Returns
        -------
        dict
            Training metrics and losses.
        """

        _, y_data = input_data
        image_data, text_data, _, _, _ = y_data

        ctc_logits = self.recognition(image_data)
        ctc_logits = self.unwrap_call_output(ctc_logits)

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

        image_data = x_data[0] if isinstance(x_data, (list, tuple)) else x_data

        ctc_logits = self.recognition(image_data, training=training)
        ctc_logits = self.unwrap_call_output(ctc_logits)

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

        evaluations = {'data': [], 'images': []}
        metrics = {'cer': [], 'wer': []}

        if probabilities is None:
            probabilities = [None] * len(x)

        batch_index = 0
        for step in range(steps):
            progbar.update(step)

            _, y_data = next(y)
            image_data, text_data, writer_data, _, _ = y_data

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

                    max_char_distance = max(char_length, max(1, len(pd)))
                    max_word_distance = max(word_length, max(1, len(pd.split())))

                    cer = Levenshtein.distance(gt, pd, score_cutoff=max_char_distance) / char_length
                    wer = Levenshtein.distance(gt.split(), pd.split(), score_cutoff=max_word_distance) / word_length

                    metrics['cer'].append(cer)
                    metrics['wer'].append(wer)

                    local_evaluation['predictions'].append({
                        'index': (j + 1),
                        'text': predict,
                        'probability': prob_pred[j] if prob_pred is not None else None,
                        'cer': cer,
                        'wer': wer,
                    })

                evaluations['data'].append(local_evaluation)
                # evaluations['images'].append({'authentic': image_path})

            progbar.update(step + 1)

        metrics = {k: float(np.mean(metrics[k])) for k in metrics}

        return metrics, evaluations


class BaseSegmentationModel(BaseModel):
    """
    BaseSegmentationModel extends BaseModel to provide additional
        functionalities to handwriting segmentation models.
    """

    def __init__(self,
                 image_shape,
                 return_features=False,
                 seed=None,
                 **kwargs):
        """
        Initializes the segmentation model.

        Parameters
        ----------
        image_shape : tuple or list
            The shape of the input images.
        return_features : bool, optional
            Whether to return intermediate features.
        seed : int, optional
            Seed for random shuffle.
        **kwargs : dict
            Additional arguments.
        """

        super().__init__(**kwargs)

        if seed is not None:
            tf.keras.utils.set_random_seed(seed)

        self.image_shape = image_shape
        self.return_features = return_features
        self.seed = seed

        self.segmentation = None

        self.global_step = tf.keras.Variable(name='global_step',
                                             initializer=0,
                                             dtype=tf.int64,
                                             trainable=False)

        self.names = [
            'segmentation',
        ]

        self.bce_loss = tf.keras.losses.BinaryCrossentropy(from_logits=False, name='bce_loss')
        self.dice_loss = tf.keras.losses.Dice(name='dice_loss')

        self.measure_tracker = MeasureTracker()
        self.monitor = f"val_{self.dice_loss.name}"

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
        })

        return config

    def train_step(self, input_data):
        """
        Executes a training step.

        Parameters
        ----------
        input_data : list or tuple
            Model inputs and targets (x_data, y_data).

        Returns
        -------
        dict
            Training metrics and losses.
        """

        x_data, _ = input_data
        aug_image_data, _, _, _, aug_segmentation_data = x_data

        with tf.GradientTape() as s_tape:
            seg_logits = self.segmentation(aug_image_data, training=True)
            seg_logits = self.unwrap_call_output(seg_logits)

            bce_loss = self.bce_loss(aug_segmentation_data, seg_logits)
            dice_loss = self.dice_loss(aug_segmentation_data, seg_logits)

            seg_loss = bce_loss + dice_loss

        s_gradients = s_tape.gradient(seg_loss, self.segmentation.trainable_weights)
        self.optimizer.apply_gradients(zip(s_gradients, self.segmentation.trainable_weights))

        self.measure_tracker.update({
            self.bce_loss.name: bce_loss,
            self.dice_loss.name: dice_loss,
            'seg_loss': seg_loss,
        })

        self.global_step.assign_add(value=1)

        return self.measure_tracker.result()

    def test_step(self, input_data):
        """
        Executes a testing step.

        Parameters
        ----------
        input_data : list or tuple
            Model inputs and targets (x_data, y_data).

        Returns
        -------
        dict
            Validation metrics and losses.
        """

        _, y_data = input_data
        image_data, _, _, _, segmentation_data = y_data

        seg_features = self.segmentation(image_data, training=False)
        seg_logits = self.unwrap_call_output(seg_features)

        bce_loss = self.bce_loss(segmentation_data, seg_logits)
        dice_loss = self.dice_loss(segmentation_data, seg_logits)

        self.measure_tracker.update({
            f"val_{self.bce_loss.name}": bce_loss,
            f"val_{self.dice_loss.name}": dice_loss,
            'val_seg_loss': bce_loss + dice_loss,
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
            Predicted segmentation masks.
        """

        image_data = x_data[0] if isinstance(x_data, (list, tuple)) else x_data

        seg_features = self.segmentation(image_data, training=training)
        seg_logits = self.unwrap_call_output(seg_features)

        return seg_logits

    def segmentation_evaluator(self, x, y, steps, threshold=0.5, verbose=1):
        """
        Evaluate segmentation predictions on the given data.

        Parameters
        ----------
        x : np.ndarray
            Predictions to be evaluated.
        y : Dataset generator
            Label data for evaluation.
        steps : int
            Number of steps for evaluation.
        threshold : float, optional
            Threshold for binarizing predictions.
        verbose : int, optional
            Verbosity level.

        Returns
        -------
        tuple
            Metrics and evaluations.
        """

        progbar = tf.keras.utils.Progbar(target=steps, unit_name='evaluate', verbose=verbose)

        evaluations = {'data': [], 'images': []}
        metrics = {'accuracy': [], 'dice': [], 'iou': []}

        batch_index = 0
        for step in range(steps):
            progbar.update(step)

            _, y_data = next(y)
            image_data, _, _, _, segmentation_data = y_data

            batch_size = len(image_data)

            y_pred = x[batch_index:batch_index + batch_size]
            batch_index += batch_size

            for i in range(batch_size):
                gt = np.expand_dims(segmentation_data[i], axis=-1)
                pd = y_pred[i][:gt.shape[0], :gt.shape[1], :gt.shape[2]]

                gt = (gt > threshold).astype(np.uint8)
                pd = (pd > threshold).astype(np.uint8)

                intersection = np.sum(gt * pd)
                union = np.sum((gt + pd) > 0)

                iou = np.nan_to_num(intersection / union)
                dice = np.nan_to_num(2 * intersection / (np.sum(gt) + np.sum(pd)))
                accuracy = np.mean(gt == pd)

                evaluations['data'].append({
                    'index': (batch_index - batch_size) + (i + 1),
                    'accuracy': accuracy,
                    'dice': dice,
                    'iou': iou,
                })

                evaluations['images'].append({
                    'authentic': image_data[i],
                    'label': gt * 255,
                    'segment': pd * 255,
                })

                metrics['accuracy'].append(accuracy)
                metrics['dice'].append(dice)
                metrics['iou'].append(iou)

            progbar.update(step + 1)

        metrics = {k: float(np.mean(v)) for k, v in metrics.items()}

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
                 return_features=False,
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
        return_features : bool, optional
            Whether to return the intermediate features.
        seed : int, optional
            Seed for random shuffle.
        **kwargs : dict
            Additional arguments.
        """

        super().__init__(**kwargs)

        if seed is not None:
            tf.keras.utils.set_random_seed(seed)

        self.image_shape = image_shape
        self.lexical_shape = lexical_shape
        self.writers_shape = writers_shape
        self.return_features = return_features
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
            Model inputs and targets (x_data, y_data).

        Returns
        -------
        dict
            Training metrics and losses.
        """

        _, y_data = input_data
        image_data, text_data, _, mask_data, _ = y_data

        features_data = self.writer_encoder(image_data)
        features_data = self.unwrap_call_output(features_data)

        latent_data = self.style_encoder(features_data)
        latent_data = self.unwrap_call_output(latent_data)

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
            random_latent_shape = (tf.shape(image_data)[0], self.style_encoder.latent_dim)
            return tf.random.normal(shape=random_latent_shape)

        def _extract_latent():
            features_data = self.writer_encoder(image_data, training=training)
            features_data = self.unwrap_call_output(features_data)

            latent_data = self.style_encoder(features_data, training=training)
            latent_data = self.unwrap_call_output(latent_data)

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

        evaluations = {'data': [], 'images': []}
        metrics = {'kid': []}

        kid = KernelInceptionDistance(scale=1.0, offset=0.0)

        batch_index = 0
        for step in range(steps):
            progbar.update(step)

            _, y_data = next(y)
            image_true_data, _, _, _, _ = y_data

            batch_size = len(image_true_data)

            image_pred_data = x[batch_index:batch_index + batch_size]
            batch_index += batch_size

            kid.update_state(image_true_data, image_pred_data)
            metrics['kid'].append(kid.result())

            for y, x in zip(image_true_data, image_pred_data):
                evaluations['images'].append({'authentic': y, 'synthetic': x})

            progbar.update(step + 1)

        metrics = {k: float(np.mean(metrics[k])) for k in metrics}

        return metrics, evaluations


class BaseWriterIdentificationModel(BaseModel):
    """
    BaseWriterIdentificationModel extends BaseModel to provide additional
        functionalities to writer identification models.
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

        if seed is not None:
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

        self.global_step = tf.keras.Variable(name='global_step',
                                             initializer=0,
                                             dtype=tf.int64,
                                             trainable=False)

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

    def build_input_data(self, input_data):
        """
        Builds image and writer inputs for the model.
        Includes synthesized samples when synthesis is enabled.

        Parameters
        ----------
        input_data : list or tuple
            Model inputs and targets (x_data, y_data).

        Returns
        -------
        tuple
            Lists of images and writers.
        """

        x_data, y_data = input_data

        aug_image_data, aug_text_data, _, aug_mask_data, _ = x_data
        _, text_data, writer_data, mask_data, _ = y_data

        images, writers = [aug_image_data], [writer_data]

        if self.writer_encoder and self.style_encoder and self.generator and \
                np.random.random() <= self.synthesis_probability:

            # extract writer style from real images
            features_data = self.writer_encoder(aug_image_data, training=False)
            features_data = self.unwrap_call_output(features_data)

            latent = self.style_encoder(features_data, training=False)
            latent = self.unwrap_call_output(latent)

            # real images and real texts
            real_real_images = self.generator([text_data, latent, mask_data], training=False)

            # real images and fake texts
            fake_real_images = self.generator([aug_text_data, latent, aug_mask_data], training=False)

            images.extend([real_real_images, fake_real_images])
            writers.extend([writer_data, writer_data])

        return images, writers

    def train_step(self, input_data):
        """
        Executes a training step.

        Parameters
        ----------
        input_data : list or tuple
            Model inputs and targets (x_data, y_data).

        Returns
        -------
        dict
            Training metrics and losses.
        """

        images, writers = self.build_input_data(input_data)

        for image, writer in zip(images, writers):
            with tf.GradientTape() as w_tape:
                wid_logits = self.writer_identification(image, training=True)
                wid_logits = self.unwrap_call_output(wid_logits)

                sce_loss = self.sce_loss(writer, wid_logits)

            w_gradients = w_tape.gradient(sce_loss, self.writer_identification.trainable_weights)
            self.optimizer.apply_gradients(zip(w_gradients, self.writer_identification.trainable_weights))

            self.measure_tracker.update({
                self.sce_loss.name: sce_loss,
            })

        self.global_step.assign_add(value=1)

        return self.measure_tracker.result()

    def test_step(self, input_data):
        """
        Executes a testing step.

        Parameters
        ----------
        input_data : list or tuple
            Model inputs and targets (x_data, y_data).

        Returns
        -------
        dict
            Training metrics and losses.
        """

        _, y_data = input_data
        image_data, _, writer_data, _, _ = y_data

        wid_logits = self.writer_identification(image_data)
        wid_logits = self.unwrap_call_output(wid_logits)

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

        image_data = x_data[0] if isinstance(x_data, (list, tuple)) else x_data

        wid_logits = self.writer_identification(image_data, training=training)
        wid_logits = self.unwrap_call_output(wid_logits)

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

        evaluations = {'data': [], 'images': []}
        metrics = {'accuracy': []}

        batch_index = 0
        for step in range(steps):
            progbar.update(step)

            _, y_data = next(y)
            image_data, _, writer_data, _, _ = y_data

            batch_size = len(image_data)

            y_pred = x[batch_index:batch_index + batch_size]
            batch_index += batch_size

            for i in range(batch_size):
                is_correct = int(writer_data[i] == y_pred[i])

                evaluations['data'].append({
                    'index': (batch_index - batch_size) + (i + 1),
                    'image': image_data[i],
                    'writer': writer_data[i],
                    'writer_prediction': y_pred[i],
                    'is_correct': is_correct,
                })

                # evaluations['images'].append({'authentic': image_data[i]})
                metrics['accuracy'].append(is_correct)

            progbar.update(step + 1)

        metrics = {k: float(np.mean(v)) for k, v in metrics.items()}

        return metrics, evaluations
