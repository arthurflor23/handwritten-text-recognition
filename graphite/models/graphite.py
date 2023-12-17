import os
import mlflow
import pickle
import datetime
import importlib
import tensorflow as tf

from models.components.callbacks import GANMonitor


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
        self.spell_checker = None

        self._mlflow_run = None
        self._synthesis_module = None
        self._recognition_module = None

        if workflow is not None:
            if 'synthesis' in workflow:
                self._synthesis_module = f"synthesis.{synthesis}"

            if 'recognition' in workflow:
                self._recognition_module = f"recognition.{recognition}"

            mlflow.set_experiment(experiment_name)

            SynthesisModel = None
            SynthesisRecognitionModel = None

            if self._synthesis_module:
                SynthesisModel = self._import_model(module=self._synthesis_module,
                                                    class_name='SynthesisModel')

            if self._recognition_module:
                SynthesisRecognitionModel = self._import_model(module=self._recognition_module,
                                                               class_name='SynthesisRecognitionModel')

            if SynthesisModel and not SynthesisRecognitionModel:
                self.model = SynthesisModel(image_shape=self.image_shape,
                                            lexical_shape=self.tokenizer.lexical_shape,
                                            writers_shape=self.tokenizer.writers_shape)
            elif SynthesisRecognitionModel:
                synthesis_params = {}

                if SynthesisModel:
                    synthesis = SynthesisModel(image_shape=self.image_shape,
                                               lexical_shape=self.tokenizer.lexical_shape,
                                               writers_shape=self.tokenizer.writers_shape)
                    synthesis_params = {
                        'style_backbone': synthesis.style_backbone,
                        'style_encoder': synthesis.style_encoder,
                        'generator': synthesis.generator,
                        'synthesis_ratio': self.synthesis_ratio,
                    }

                self.model = SynthesisRecognitionModel(image_shape=self.image_shape,
                                                       lexical_shape=self.tokenizer.lexical_shape,
                                                       **synthesis_params)

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
        Compile the models.

        Parameters
        ----------
        learning_rate : float, optional
            The learning rate for the optimizer.
        """

        self.model.compile(learning_rate=learning_rate)

    def get_run_info(self):
        """
        Get information about the current MLflow run.

        Returns
        -------
        tuple
            A tuple containing the run ID, run name and artifacts path.
        """

        run_id = None
        run_name = str(datetime.datetime.now())
        artifact_path = None

        if self._mlflow_run is not None:
            run_id = self._mlflow_run.info.run_id
            run_name = self._mlflow_run.info.run_name

            artifact_uri = os.path.join(self._mlflow_run.info.artifact_uri, 'artifacts')
            artifact_path = artifact_uri.replace('file://', '')

        return run_id, run_name, artifact_path

    def set_run_info(self, run=None):
        """
        Set the MLflow run information.

        Parameters
        ----------
        run : MLflow Run, optional
            MLflow Run object to set as the current run.

        Returns
        -------
        str
            The artifacts path.
        """

        if run is not None:
            self._mlflow_run = run

        _, _, artifact_path = self.get_run_info()

        return artifact_path

    def fit(self,
            training_gen,
            training_steps=None,
            validation_gen=None,
            validation_steps=None,
            monitor_samples_gen=None,
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
        training_gen : generator
            Generator yielding training data batches.
        training_steps : int, optional
            Number of steps per training epoch.
        validation_gen : generator, optional
            Generator yielding validation data batches.
        validation_steps : int, optional
            Number of steps per validation run.
        monitor_samples_gen : generator, optional
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

        with mlflow.start_run(run_name=str(datetime.datetime.now())) as run:
            artifact_path = self.set_run_info(run)

            logs_path = os.path.join(artifact_path, 'logs')
            os.makedirs(logs_path, exist_ok=True)

            tensorboard_path = os.path.join(artifact_path, 'tensorboard')
            os.makedirs(tensorboard_path, exist_ok=True)

            callbacks = [
                tf.keras.callbacks.CSVLogger(
                    filename=os.path.join(logs_path, 'epochs.log'),
                    separator=',',
                    append=True,
                ),
                tf.keras.callbacks.ModelCheckpoint(
                    filepath=os.path.join(artifact_path, '<model>.keras'),
                    mode='min',
                    monitor=self.model.monitor,
                    save_freq='epoch',
                    save_best_only=True,
                    save_weights_only=True,
                    verbose=1,
                ),
                tf.keras.callbacks.TensorBoard(
                    log_dir=tensorboard_path,
                    histogram_freq=0,
                    write_graph=True,
                    write_images=False,
                    write_steps_per_second=False,
                    update_freq='epoch',
                    profile_batch=0,
                    embeddings_freq=0,
                    embeddings_metadata=None,
                ),
                tf.keras.callbacks.EarlyStopping(
                    mode='min',
                    monitor=self.model.monitor,
                    min_delta=1e-8,
                    patience=patience,
                    start_from_epoch=0,
                    restore_best_weights=True,
                    verbose=1,
                ),
                tf.keras.callbacks.ReduceLROnPlateau(
                    mode='min',
                    monitor=self.model.monitor,
                    min_lr=1e-4,
                    min_delta=1e-8,
                    factor=plateau_factor,
                    cooldown=plateau_cooldown,
                    patience=plateau_patience,
                    verbose=1,
                ),
            ]

            if self._synthesis_module:
                samples_path = os.path.join(artifact_path, 'samples')
                os.makedirs(samples_path, exist_ok=True)

                callbacks.extend([
                    GANMonitor(filepath=samples_path,
                               sample_gen=monitor_samples_gen,
                               sample_steps=monitor_samples_steps,
                               latent_dim=self.model.generator.latent_dim,
                               monitor=self.model.monitor),
                ])

            mlflow.set_tags({
                'graphite.module': f"{self._synthesis_module}:{self._recognition_module}",
            })

            with open(os.path.join(artifact_path, 'tokenizer.pkl'), 'wb') as f:
                pickle.dump(self.tokenizer, f)

            history = self.model.fit(x=training_gen,
                                     steps_per_epoch=training_steps,
                                     validation_gen=validation_gen,
                                     validation_steps=validation_steps,
                                     callbacks=callbacks,
                                     epochs=(epochs or 1000000),
                                     verbose=1)

        return history

    def predict(self,
                test_gen,
                test_steps,
                top_paths=1,
                beam_width=30,
                ctc_decode=True,
                token_decode=True):
        """
        Make predictions on test data with optional CTC decoding and spelling correction.

        Parameters
        ----------
        test_gen : tf.data.Dataset
            Test data for predictions.
        test_steps : int
            Number of steps for prediction.
        top_paths : int, optional
            Number of top paths for CTC decoding.
        beam_width : int, optional
            Beam width for CTC decoding.
        ctc_decode : bool, optional
            Perform CTC decoding on predictions.
        token_decode : bool, optional
            Decode tokens during CTC decoding.

        Returns
        -------
        tuple
            Predictions, corrections, and probabilities (if CTC decoding is used).
        """

        predictions = self.model.predict(x=test_gen, steps=test_steps, verbose=1)
        corrections, probabilities = None, None

        if ctc_decode:
            tokenizer = self.tokenizer if token_decode else None
            predictions, probabilities = self.model.ctc_decode(predictions=predictions,
                                                               top_paths=top_paths,
                                                               beam_width=beam_width,
                                                               tokenizer=tokenizer,
                                                               verbose=1)

            corrections = self.spell_checker.predict(predictions) \
                if not token_decode or self.spell_checker is None else predictions

        return predictions, corrections, probabilities

    def evaluate(self,
                 label_gen,
                 label_steps,
                 predictions):
        """
        Evaluate CTC predictions on the given labeled data.

        Parameters
        ----------
        label_gen : Dataset generator
            Labeled data for evaluation.
        label_steps : int
            Number of steps for evaluation.
        predictions : np.ndarray
            Predictions to be evaluated.

        Returns
        -------
        tuple
            Metrics and evaluations.
        """

        metrics, evaluations = self.model.ctc_evaluate(x=label_gen,
                                                       steps=label_steps,
                                                       predictions=predictions,
                                                       verbose=1)

        return metrics, evaluations

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
            Identification for synthesis model.
        synthesis_index : int, optional
            Run index for the synthesis model.
        recognition : str, optional
            Identification for recognition model.
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
            filter_string = f"status='FINISHED' AND tag.graphite.module LIKE '%{label}%'"

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
