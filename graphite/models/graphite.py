import os
import random
import mlflow
import pickle
import datetime
import importlib
import tensorflow as tf

from models.components.callback import GANMonitor
from models.components.loss import CTCLoss
from models.components.metric import EditDistance
from models.components.optimizer import NormalizedOptimizer


class Graphite():
    """
    """

    def __init__(self,
                 workflow=None,
                 synthesis=None,
                 recognition=None,
                 spelling=None,
                 image_shape=None,
                 tokenizer=None,
                 synthesis_ratio=1.0,
                 experiment_name='Default'):
        """
        """

        self.workflow = workflow
        self.synthesis = synthesis
        self.recognition = recognition
        self.spelling = spelling
        self.image_shape = image_shape
        self.tokenizer = tokenizer
        self.synthesis_ratio = synthesis_ratio
        self.experiment_name = experiment_name

        self.model = None
        self.synthesis_label = f"synthesis.{synthesis}"
        self.recognition_label = f"recognition.{recognition}"

        if workflow is not None:
            self.experiment = mlflow.set_experiment(experiment_name)

            SynthesizerModel = None
            SynthesizerRecognizerModel = None

            if 'synthesis' in self.workflow:
                SynthesizerModel = self._import_model(module=self.synthesis_label,
                                                      class_name='SynthesizerModel')

            if 'recognition' in self.workflow:
                SynthesizerRecognizerModel = self._import_model(module=self.recognition_label,
                                                                class_name='SynthesizerRecognizerModel')

            if SynthesizerModel and not SynthesizerRecognizerModel:
                self.model = SynthesizerModel(image_shape=self.image_shape,
                                              lexical_shape=self.tokenizer.lexical_shape,
                                              writers_shape=self.tokenizer.writers_shape)
            elif SynthesizerRecognizerModel:
                synthesizer_params = {}

                if SynthesizerModel:
                    synthesizer = SynthesizerModel(image_shape=self.image_shape,
                                                   lexical_shape=self.tokenizer.lexical_shape,
                                                   writers_shape=self.tokenizer.writers_shape)
                    synthesizer_params = {
                        'style_backbone': synthesizer.style_backbone,
                        'style_encoder': synthesizer.style_encoder,
                        'generator': synthesizer.generator,
                        'synthesis_ratio': self.synthesis_ratio,
                    }

                self.model = SynthesizerRecognizerModel(image_shape=self.image_shape,
                                                        lexical_shape=self.tokenizer.lexical_shape,
                                                        **synthesizer_params)

        # with mlflow.start_run(run_id=None, run_name=str(datetime.datetime.now())) as _:
        #     mlflow.set_tags({'graphite.label': 'synthesis:gan,recognition:bluche'})
        # exit()

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
        info += f"\n{self.model}"

        return info

    def _import_model(self, module, class_name):
        """
        Dynamically imports and returns a class from a specified module.

        Parameters
        ----------
        module : str
            Module name containing the class.
        class_name : str
            Class name to be imported.

        Returns
        -------
        class
            The dynamically imported class.
        """

        module_name = importlib.util.resolve_name(f".{module}", __package__)
        module_spec = importlib.util.find_spec(module_name)
        assert module_spec is not None, 'model file must be created'

        module = importlib.import_module(module_name, __package__)
        assert hasattr(module, class_name), f"`{class_name}` class must be created"

        model = getattr(module, class_name)
        return model

    def compile(self, learning_rate=0.001):
        """
        Configure the submodels for training.

        Parameters
        ----------
        learning_rate : float, optional
            The learning rate for the optimizer.
        """

        self.model.compile(learning_rate=learning_rate)

    def fit(self,
            training_data,
            training_steps=None,
            validation_data=None,
            validation_steps=None,
            monitor_samples_data=None,
            monitor_samples_steps=None,
            plateau_factor=0.1,
            plateau_cooldown=0,
            plateau_patience=20,
            patience=30,
            epochs=None):
        """
        Trains the model.

        Parameters
        ----------
        training_data : generator
            Generator yielding training data batches.
        training_steps : int, optional
            Number of steps per training epoch.
        validation_data : generator, optional
            Generator yielding validation data batches.
        validation_steps : int, optional
            Number of steps per validation run.
        monitor_samples_data : generator, optional
            Generator yielding samples data batches.
        monitor_samples_steps : int, optional
            Number of steps per sample run.
        plateau_factor : float, optional
            Factor for reducing the learning rate.
        plateau_cooldown : int, optional
            Epochs to wait after a learning rate reduction.
        plateau_patience : int, optional
            Epochs without improvement before reducing the learning rate.
        patience : int, optional
            Epochs without improvement before stopping training.
        epochs : int, optional
            Number of training epochs.

        Returns
        -------
        history object
            Training and validation progress details.
        """

        start_time = datetime.datetime.now()

        with mlflow.start_run(run_name=str(start_time)) as run:
            self.run = run

            artifact_uri = os.path.join(run.info.artifact_uri, 'artifacts')
            artifact_path = artifact_uri.replace('file://', '')

            logs_path = os.path.join(artifact_path, 'logs')
            os.makedirs(logs_path, exist_ok=True)

            tensorboard_path = os.path.join(artifact_path, 'tensorboard')
            os.makedirs(tensorboard_path, exist_ok=True)

            callbacks = [
                # tf.keras.callbacks.CSVLogger(
                #     filename=os.path.join(logs_path, 'epochs.log'),
                #     separator=',',
                #     append=True,
                # ),
                # tf.keras.callbacks.ModelCheckpoint(
                #     filepath=os.path.join(artifact_path, 'model.keras'),
                #     mode='min',
                #     monitor=self.model.monitor,
                #     save_freq='epoch',
                #     save_best_only=True,
                #     save_weights_only=True,
                #     verbose=1,
                # ),
                # tf.keras.callbacks.TensorBoard(
                #     log_dir=tensorboard_path,
                #     histogram_freq=0,
                #     write_graph=True,
                #     write_images=False,
                #     write_steps_per_second=False,
                #     update_freq='epoch',
                #     profile_batch=0,
                #     embeddings_freq=0,
                #     embeddings_metadata=None,
                # ),
                # tf.keras.callbacks.EarlyStopping(
                #     mode='min',
                #     monitor=self.model.monitor,
                #     min_delta=1e-8,
                #     patience=patience,
                #     start_from_epoch=0,
                #     restore_best_weights=True,
                #     verbose=1,
                # ),
                # tf.keras.callbacks.ReduceLROnPlateau(
                #     mode='min',
                #     monitor=self.model.monitor,
                #     min_lr=1e-4,
                #     min_delta=1e-8,
                #     factor=plateau_factor,
                #     cooldown=plateau_cooldown,
                #     patience=plateau_patience,
                #     verbose=1,
                # ),
            ]

            if 'None' not in self.synthesis_label and \
                    monitor_samples_data is not None and \
                    monitor_samples_steps is not None:

                samples_path = os.path.join(artifact_path, 'samples')
                os.makedirs(samples_path, exist_ok=True)

                callbacks.extend([
                    GANMonitor(filepath=samples_path,
                               sample_data=monitor_samples_data,
                               sample_steps=monitor_samples_steps,
                               latent_dim=self.model.generator.latent_dim,
                               monitor=self.model.monitor),
                ])

            history = self.model.fit(x=training_data,
                                     steps_per_epoch=training_steps,
                                     validation_data=validation_data,
                                     validation_steps=validation_steps,
                                     callbacks=callbacks,
                                     epochs=epochs or int(1e+6),
                                     verbose=1)

        return history

    @staticmethod
    def get_tokenizer(synthesis=None,
                      synthesis_index=None,
                      recognition=None,
                      recognition_index=None,
                      experiment_name='Default'):
        """
        Retrieves a tokenizer from MLflow artifacts.

        Parameters
        ----------
        synthesis : str, optional
            Identifier for synthesis model.
        synthesis_index : int, optional
            Run index for the synthesis model.
        recognition : str, optional
            Identifier for recognition model.
        recognition_index : int, optional
            Run index for the recognition model.
        experiment_name : str, optional
            MLflow experiment name.

        Returns
        -------
        tuple
            (tokenizer, artifacts_path) or (None, None) if not found.
        """

        def get_artifacts_path(label, run_index):
            if run_index is None:
                return None

            experiment = mlflow.set_experiment(experiment_name)
            experiment_ids = [experiment.experiment_id]
            filter_string = f"status='FINISHED' AND tag.graphite.label LIKE '%{label}%'"

            df = mlflow.search_runs(experiment_ids=experiment_ids,
                                    filter_string=filter_string,
                                    order_by=['tags.mlflow.runName ASC'])

            if not df.empty:
                run_context = mlflow.get_run(df.iloc[run_index]['run_id'])
                artifacts_uri = os.path.join(run_context.info.artifact_uri, 'artifacts')
                artifacts_path = artifacts_uri.replace('file://', '')
                return artifacts_path if os.path.isdir(artifacts_path) else None
            return None

        synthesis_uri = get_artifacts_path(label=f"synthesis.{synthesis}", run_index=synthesis_index)
        recognition_uri = get_artifacts_path(label=f"recognition.{recognition}", run_index=recognition_index)

        artifacts_path = (synthesis_uri or recognition_uri)
        tokenizer = None

        if artifacts_path and os.path.isdir(artifacts_path):
            tokenizer_uri = os.path.join(artifacts_path, 'tokenizer.pkl')

            if os.path.isfile(tokenizer_uri):
                with open(tokenizer_uri, 'rb') as f:
                    tokenizer = pickle.load(f)

        return tokenizer, artifacts_path


class CarbonModel(tf.keras.Model):
    """
    A handwriting synthesis with recognition model on the TensorFlow Keras framework.

    This model combines components for style transfer and text generation (synthesis)
        with a handwriting recognition model.
    """

    def __init__(self,
                 generator,
                 style_encoder,
                 style_backbone,
                 recognition,
                 synthesis_ratio=1.0,
                 **kwargs):
        """
        Initialize the synthesis with recognition model.

        Parameters
        ----------
        generator : Generator instance
            Generator model for image generation.
        style_encoder : StyleEncoder instance
            StyleEncoder model for encoding extracted style features.
        style_backbone : StyleBackbone instance
            StyleBackbone model for extracting style patterns from images.
        recognition : HandwritingRecognition instance
            Recognition model for transcribing text.
        synthesis_ratio : float, optional
            Probability to use synthetic data.
        **kwargs : dict
            Additional keyword arguments.
        """

        super().__init__(name='synthesis_recognition', **kwargs)

        self.generator = generator
        self.style_encoder = style_encoder
        self.style_backbone = style_backbone
        self.recognition = recognition
        self.synthesis_ratio = synthesis_ratio

        self.names = [
            self.generator.name,
            self.style_encoder.name,
            self.style_backbone.name,
            self.recognition.name,
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

        ctc_logits = self.recognition(image_inputs, training=training)

        return ctc_logits
