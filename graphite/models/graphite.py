import os
import json
import pickle
import mlflow
import datetime
import importlib
import tensorflow as tf

from models.components.callbacks import GANMonitor


class Graphite():
    """
    Graphite is a configurable model framework for synthesis and recognition tasks,
        integrating various components and supporting MLflow experimentation.
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
        Initializes the Graphite model with specified components.

        Parameters
        ----------
        workflow : str, optional
            Workflow to be used.
        synthesis : str, optional
            Identifier for the synthesis model to be used.
        recognition : str, optional
            Identifier for the recognition model to be used.
        spelling : str, optional
            Identifier for the spelling correction model to be used.
        image_shape : tuple, optional
            Shape of the input images.
        tokenizer : Tokenizer, optional
            Tokenizer for processing text data.
        synthesis_ratio : float, optional
            Ratio determining the synthesis influence.
        experiment_name : str, optional
            Name of the MLflow experiment.
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
        self.spelling_model = None

        self._mlrun = None
        self._mlrun_synthesis = None
        self._mlrun_recognition = None

        if workflow is not None:
            mlflow.set_experiment(experiment_name)

            if 'synthesis' in workflow:
                self._mlrun_synthesis = str(synthesis)

            if 'recognition' in workflow:
                self._mlrun_recognition = str(recognition)

            SynthesisModel = None
            RecognitionModel = None

            if self._mlrun_synthesis:
                module = f"synthesis.{self._mlrun_synthesis}"
                SynthesisModel = self._import_model(module=module, class_name='SynthesisModel')

            if self._mlrun_recognition:
                module = f"recognition.{self._mlrun_recognition}"
                RecognitionModel = self._import_model(module=module, class_name='RecognitionModel')

            if SynthesisModel and not RecognitionModel:
                self.model = SynthesisModel(image_shape=self.image_shape,
                                            lexical_shape=self.tokenizer.lexical_shape,
                                            writers_shape=self.tokenizer.writers_shape)
            elif RecognitionModel:
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

                self.model = RecognitionModel(image_shape=self.image_shape,
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

    def compile(self, learning_rate=0.001, mlrun=None):
        """
        Compile the models.

        Parameters
        ----------
        learning_rate : float, optional
            The learning rate for the optimizer.
        mlrun : mlflow.entities.Run object, optional
            MLFlow Run entity.
        """

        if mlrun is not None:
            run_info = self.get_run_info(mlrun=mlrun)
            artifact_path = os.path.join(run_info['artifact_path'], '<model>.h5')

            self.model.load_weights(filepath=artifact_path, by_name=True, skip_mismatch=True)

        self.model.compile(learning_rate=learning_rate)

    def get_run_info(self, mlrun=None, create=False):
        """
        Get information about the current MLflow run.

        Parameters
        ----------
        run : MLflow Run, optional
            MLflow Run object to set as the current run.
        create : bool, optional
            Create a new mlrun.

        Returns
        -------
        dict
            A dict containing the run ID, run name and artifacts path.
        """

        run_id = None
        run_name = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        artifact_path = None

        if mlrun is not None:
            self._mlrun = mlrun

        if self._mlrun is not None and not create:
            run_id = self._mlrun.info.run_id
            run_name = self._mlrun.info.run_name
            artifact_path = self._mlrun.info.artifact_uri.replace('file://', '')

        run_info = {
            'id': run_id,
            'name': run_name,
            'artifact_path': artifact_path,
        }

        return run_info

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

        run_info = self.get_run_info(create=True)

        with mlflow.start_run(run_name=run_info['name']) as run:
            run_info = self.get_run_info(mlrun=run)

            logs_path = os.path.join(run_info['artifact_path'], 'logs')
            os.makedirs(logs_path, exist_ok=True)

            tensorboard_path = os.path.join(run_info['artifact_path'], 'tensorboard')
            os.makedirs(tensorboard_path, exist_ok=True)

            callbacks = [
                tf.keras.callbacks.CSVLogger(
                    filename=os.path.join(logs_path, 'epochs.log'),
                    separator=',',
                    append=True,
                ),
                tf.keras.callbacks.ModelCheckpoint(
                    filepath=os.path.join(run_info['artifact_path'], '<model>.h5'),
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

            if 'synthesis' in self.workflow:
                synthesis_path = os.path.join(run_info['artifact_path'], 'synthesis')
                os.makedirs(synthesis_path, exist_ok=True)

                callbacks.extend([
                    GANMonitor(filepath=synthesis_path,
                               sample_gen=monitor_samples_gen,
                               sample_steps=monitor_samples_steps,
                               latent_dim=self.model.generator.latent_dim),
                ])

            with open(os.path.join(run_info['artifact_path'], 'tokenizer.pkl'), 'wb') as f:
                pickle.dump(self.tokenizer, f)

            history = self.model.fit(x=training_gen,
                                     steps_per_epoch=training_steps,
                                     validation_data=validation_gen,
                                     validation_steps=validation_steps,
                                     callbacks=callbacks,
                                     epochs=(epochs or 1000000),
                                     verbose=1)

            monitor = self.model.monitor

            if self.model.monitor not in history.history:
                monitor = self.model.monitor.replace('val_', '')

            min_monitor_value = min(history.history[monitor])
            best_monitor_index = history.history[monitor].index(min_monitor_value)

            metrics = {k: history.history[k][best_monitor_index] for k in history.history if k != 'lr'}

            train_metrics = {k: metrics[k] for k in metrics if not k.startswith('val')}
            valid_metrics = {k: metrics[k] for k in metrics if k.startswith('val')}

            mlflow.log_metrics(train_metrics)
            mlflow.log_metrics(valid_metrics)

            mlflow.set_tags({'graphite.synthesis': self._mlrun_synthesis})
            mlflow.set_tags({'graphite.recognition': self._mlrun_recognition})
            mlflow.end_run()

        self.save_context(metrics=train_metrics, prefix='train')
        self.save_context(metrics=valid_metrics, prefix='valid')

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

        if test_gen is None:
            return None, None, None

        predictions = self.model.predict(x=test_gen, steps=test_steps, verbose=1)
        corrections, probabilities = None, None

        if ctc_decode:
            tokenizer = self.tokenizer if token_decode else None
            predictions, probabilities = self.model.ctc_decode(x=predictions,
                                                               steps=test_steps,
                                                               top_paths=top_paths,
                                                               beam_width=beam_width,
                                                               tokenizer=tokenizer,
                                                               verbose=1)

            corrections = self.spelling_model.predict(predictions) \
                if token_decode and self.spelling_model is not None else predictions

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

        if label_gen is None:
            return None, None

        metrics, evaluations = self.model.ctc_evaluate(x=label_gen,
                                                       steps=label_steps,
                                                       predictions=predictions,
                                                       verbose=1)

        return metrics, evaluations

    def save_context(self,
                     dataset=None,
                     augmentor=None,
                     metrics=None,
                     evaluations=None,
                     spelling_metrics=None,
                     spelling_evaluations=None,
                     prefix='test'):
        """
        Save relevant context information to MLflow and log files.

        Parameters
        ----------
        dataset : Dataset instance or None, optional
            Dataset object instance.
        augmentor : Augmentor instance or None, optional
            Augmentor object instance.
        metrics : dict or None, optional
            Model metrics.
        evaluations : list or None, optional
            Model evaluations.
        spelling_metrics : dict or None, optional
            Spelling metrics.
        spelling_evaluations : list or None, optional
            Spelling evaluations.
        prefix : str, optional
            Prefix used in the metric logs.
        """

        run_info = self.get_run_info()

        with mlflow.start_run(run_id=run_info['id'], run_name=run_info['name']) as run:
            run_info = self.get_run_info(mlrun=run)

            logs_path = os.path.join(run_info['artifact_path'], 'logs')
            os.makedirs(logs_path, exist_ok=True)

            def save_content(name, content, metric=False, json_content=False):
                if content is not None:
                    artifact = os.path.join(logs_path, f"{name}.log")

                    if metric:
                        sufix = '_'.join(name.split('_')[2:])
                        sufix = f"_{sufix}" if sufix else ''
                        mlflow.log_metrics({f"test_{k}{sufix}": content[k] for k in content})

                    if json_content:
                        content = json.dumps(content, indent=4)

                    with open(artifact, 'w') as f:
                        f.write(f"{content}".strip())

            save_content('dataset', dataset)
            save_content('augmentor', augmentor)
            save_content('model', self.model)

            save_content(f"{prefix}_metrics", metrics, metric=True, json_content=True)
            save_content(f"{prefix}_metrics_spelling", spelling_metrics, metric=True, json_content=True)
            save_content(f"{prefix}_samples", evaluations, json_content=True)
            save_content(f"{prefix}_samples_spelling", spelling_evaluations, json_content=True)
            mlflow.end_run()

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
            (tokenizer, mlrun) or (None, None) if not found.
        """

        def get_artifacts_path(tag_name, tag_value, run_index):
            if run_index is not None:
                experiment = mlflow.set_experiment(experiment_name)
                experiment_ids = [experiment.experiment_id]

                filter_string = f"status='FINISHED' AND tag.graphite.{tag_name}='{tag_value}'"

                df = mlflow.search_runs(experiment_ids=experiment_ids,
                                        filter_string=filter_string,
                                        order_by=['tags.mlflow.runName ASC'])

                if not df.empty and run_index < len(df):
                    mlrun = mlflow.get_run(df.iloc[run_index]['run_id'])
                    artifact_path = mlrun.info.artifact_uri.replace('file://', '')
                    return mlrun, artifact_path

            return None, None

        s_mlrun, s_path = get_artifacts_path('synthesis', synthesis, synthesis_index)
        r_mlrun, r_path = get_artifacts_path('recognition', recognition, recognition_index)

        tokenizer = None
        mlrun = s_mlrun or r_mlrun
        artifacts_path = s_path or r_path

        if artifacts_path:
            tokenizer_uri = os.path.join(artifacts_path, 'tokenizer.pkl')

            if os.path.isfile(tokenizer_uri):
                try:
                    with open(tokenizer_uri, 'rb') as f:
                        tokenizer = pickle.load(f)
                except Exception as e:
                    print(f"Error loading tokenizer: {e}")

        if mlrun:
            print('==================================================')
            print(f"{'Loading Mlrun'.center(50)}")
            print('--------------------------------------------------')
            print(f"{'experiment_id':<{25}}: {mlrun.info.experiment_id[:23]}")
            print(f"{'run_id':<{25}}: {mlrun.info.run_id[:23]}")
            print(f"{'run_name':<{25}}: {mlrun.info.run_name[:23]}")

        return tokenizer, mlrun
