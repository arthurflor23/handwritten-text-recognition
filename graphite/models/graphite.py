import os
import cv2
import yaml
import glob
import json
import pickle
import mlflow
import importlib
import numpy as np
import tensorflow as tf

from datetime import datetime
from graphite.models.components.callbacks import GANMonitor


class Graphite():
    """
    Graphite is a configurable model framework for synthesis and recognition tasks,
        integrating various components and supporting MLflow experimentation.
    """

    def __init__(self,
                 synthesis=None,
                 recognition=None,
                 spelling=None,
                 image_shape=None,
                 tokenizer=None,
                 discriminator_steps=1,
                 generator_steps=4,
                 synthesis_ratio=1.0,
                 experiment_name=None,
                 seed=None):
        """
        Initializes the Graphite model with specified components.

        Parameters
        ----------
        synthesis : str, optional
            Identifier for the synthesis model.
        recognition : str, optional
            Identifier for the recognition model.
        spelling : str, optional
            Identifier for the spelling correction model.
        image_shape : tuple, optional
            Shape of the input images.
        tokenizer : Tokenizer, optional
            Tokenizer for processing text data.
        discriminator_steps : int, optional
            The repetition of steps for discriminator training.
        generator_steps : int, optional
            The skipping steps for generator training.
        synthesis_ratio : float, optional
            Ratio determining the synthesis influence.
        experiment_name : str, optional
            Name of the MLflow experiment.
        seed : int, optional
            Seed for random shuffle.
        """

        self.synthesis = synthesis
        self.recognition = recognition
        self.spelling = spelling
        self.tokenizer = tokenizer
        self.experiment_name = experiment_name or 'Default'

        self.model = None
        self.spelling_model = None
        self.run_context = None

        if self.synthesis or self.recognition:
            mlflow.set_experiment(self.experiment_name)

            SynthesisModel = None
            RecognitionModel = None
            SpellingModel = None

            if self.synthesis:
                SynthesisModel = self._import_model(module=f"synthesis.{self.synthesis}",
                                                    class_name='SynthesisModel')

            if self.recognition:
                RecognitionModel = self._import_model(module=f"recognition.{self.recognition}",
                                                      class_name='RecognitionModel')

                if self.spelling:
                    SpellingModel = self._import_model(module=f"spelling.{self.spelling}",
                                                       class_name='SpellingModel')

            if SynthesisModel and not RecognitionModel:
                self.model = SynthesisModel(name='synthesis',
                                            image_shape=image_shape,
                                            lexical_shape=self.tokenizer.lexical_shape,
                                            writers_shape=self.tokenizer.writers_shape,
                                            discriminator_steps=discriminator_steps,
                                            generator_steps=generator_steps,
                                            seed=seed)
            elif RecognitionModel:
                synthesis_params = {}

                if SynthesisModel:
                    synthesis = SynthesisModel(name='synthesis',
                                               image_shape=image_shape,
                                               lexical_shape=self.tokenizer.lexical_shape,
                                               writers_shape=self.tokenizer.writers_shape,
                                               discriminator_steps=discriminator_steps,
                                               generator_steps=generator_steps,
                                               seed=seed)
                    synthesis_params = {
                        'style_encoder': synthesis.style_encoder,
                        'generator': synthesis.generator,
                        'synthesis_ratio': synthesis_ratio,
                    }

                self.model = RecognitionModel(name='recognition',
                                              image_shape=image_shape,
                                              lexical_shape=self.tokenizer.lexical_shape,
                                              seed=seed,
                                              **synthesis_params)

                if SpellingModel:
                    self.spelling_model = SpellingModel()

    def __repr__(self):
        """
        Provides a formatted string with useful information.

        Returns
        -------
        str
            Formatted string with useful information.
        """

        if not self.model:
            return str(None)

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

    def compile(self, learning_rate=None, run_context=None):
        """
        Compile the models.

        Parameters
        ----------
        learning_rate : float, optional
            The learning rate for the optimizer.
        run_context : mlflow.entities.Run object, optional
            MLFlow run context.
        """

        if run_context is not None:
            run_info = self.get_run_info(run_context=run_context)
            artifact_path = os.path.join(run_info['artifact_path'], '<model>.h5')

            self.model.load_weights(filepath=artifact_path, skip_mismatch=True, by_name=True)

        self.model.compile(learning_rate=learning_rate)

    def fit(self,
            training_gen,
            training_steps=None,
            validation_gen=None,
            validation_steps=None,
            monitor_sample_gen=None,
            monitor_sample_steps=None,
            plateau_factor=0.1,
            plateau_cooldown=0,
            plateau_patience=20,
            patience=40,
            epochs=None,
            verbose=1):
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
        monitor_sample_gen : generator, optional
            Generator yielding samples data batches.
        monitor_sample_steps : int, optional
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
        verbose : int, optional
            Verbosity level.

        Returns
        -------
        history object
            Training and validation progress details.
        """

        new_context = True

        if self.run_context is not None:
            path = self.run_context.info.artifact_uri.replace('file://', '')
            new_context = bool(glob.glob(os.path.join(path, '**', '*.h5'), recursive=True))

        run_info = self.get_run_info(new_context=new_context)

        with mlflow.start_run(run_id=run_info['id'], run_name=run_info['name']) as run:
            run_info = self.get_run_info(run_context=run)

            logs_path = os.path.join(run_info['artifact_path'], 'logs')
            os.makedirs(logs_path, exist_ok=True)

            tensorboard_path = os.path.join(run_info['artifact_path'], 'tensorboard')
            os.makedirs(tensorboard_path, exist_ok=True)

            monitor = self.model.monitor
            if validation_gen is not None:
                monitor = f"val_{self.model.monitor}"

            callbacks = [
                tf.keras.callbacks.CSVLogger(
                    filename=os.path.join(logs_path, 'epochs.log'),
                    separator=',',
                    append=True,
                ),
                tf.keras.callbacks.ModelCheckpoint(
                    filepath=os.path.join(run_info['artifact_path'], '<model>.h5'),
                    mode='min',
                    monitor=monitor,
                    save_freq='epoch',
                    save_best_only=True,
                    save_weights_only=True,
                    verbose=verbose,
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
                    monitor=monitor,
                    min_delta=1e-7,
                    patience=patience,
                    start_from_epoch=0,
                    restore_best_weights=True,
                    verbose=verbose,
                ),
            ]

            if self.recognition:
                callbacks.extend([
                    tf.keras.callbacks.ReduceLROnPlateau(
                        mode='min',
                        monitor=monitor,
                        min_lr=1e-4,
                        min_delta=1e-7,
                        factor=plateau_factor,
                        cooldown=plateau_cooldown,
                        patience=plateau_patience,
                        verbose=verbose,
                    ),
                ])

            elif self.synthesis:
                synthesis_path = os.path.join(run_info['artifact_path'], 'synthesis', 'training')
                os.makedirs(synthesis_path, exist_ok=True)

                callbacks.extend([
                    GANMonitor(
                        filepath=synthesis_path,
                        sample_gen=monitor_sample_gen,
                        sample_steps=monitor_sample_steps,
                        latent_dim=self.model.style_encoder.latent_dim,
                    ),
                ])

            mlflow.set_tags({'graphite.synthesis': str(self.synthesis)})
            mlflow.set_tags({'graphite.recognition': str(self.recognition)})
            mlflow.set_tags({'graphite.spelling': str(self.spelling)})

            with open(os.path.join(run_info['artifact_path'], 'tokenizer.pkl'), 'wb') as f:
                pickle.dump(self.tokenizer, f)

            history = self.model.fit(x=training_gen,
                                     steps_per_epoch=training_steps,
                                     validation_data=validation_gen,
                                     validation_steps=validation_steps,
                                     callbacks=callbacks,
                                     epochs=(epochs or 1000000),
                                     verbose=verbose)
            mlflow.end_run()

        if monitor in history.history:
            best_metric_index = history.history[monitor].index(min(history.history[monitor]))
            metrics = {k: history.history[k][best_metric_index] for k in history.history if k != 'lr'}

            training_metrics = {k: metrics[k] for k in metrics if not k.startswith('val_')}
            validation_metrics = {k.replace('val_', ''): metrics[k] for k in metrics if k.startswith('val_')}

            self.save_context(metrics=training_metrics, prefix='training')
            self.save_context(metrics=validation_metrics, prefix='validation')

        return history

    def predict_recognition(self,
                            x,
                            steps,
                            top_paths=1,
                            beam_width=32,
                            ctc_decode=True,
                            token_decode=True,
                            corrections=False,
                            verbose=1):
        """
        Make predictions on test data with CTC decoding and spelling correction.

        Parameters
        ----------
        x : Dataset generator
            Data for predictions.
        steps : int
            Number of steps for prediction.
        top_paths : int, optional
            Number of top paths for CTC decoding.
        beam_width : int, optional
            Beam width for CTC decoding.
        ctc_decode : bool, optional
            Perform CTC decoding on predictions.
        token_decode : bool, optional
            Decode tokens during CTC decoding.
        corrections : str, optional
            Peform corrections using spelling model.
        verbose : int, optional
            Verbosity level.

        Returns
        -------
        tuple
            Predictions and probabilities.
        """

        if x is None:
            return None, None

        predictions = self.model.predict(x=x, steps=steps, verbose=verbose)
        probabilities = None

        if ctc_decode:
            tokenizer = self.tokenizer if token_decode else None
            predictions, probabilities = self.model.ctc_decoder(x=predictions,
                                                                steps=steps,
                                                                top_paths=top_paths,
                                                                beam_width=beam_width,
                                                                tokenizer=tokenizer,
                                                                verbose=verbose)

            if token_decode and corrections and self.spelling_model:
                predictions = self.spelling_model.predict(x=predictions,
                                                          steps=steps,
                                                          verbose=verbose)

        return predictions, probabilities

    def predict_synthesis(self, x, steps, verbose=1):
        """
        Make image generations with synthesis model using test data.

        Parameters
        ----------
        x : Dataset generator
            Data for predictions.
        steps : int
            Number of steps for prediction.
        verbose : int, optional
            Verbosity level.

        Returns
        -------
        np.ndarray
            Predictions.
        """

        if x is None:
            return None

        predictions = self.model.predict(x=x, steps=steps, verbose=verbose)
        predictions = np.asarray((predictions + 1.0) * 127.5, dtype=np.uint8)

        return predictions

    def evaluate_recognition(self, x, y, steps, probabilities=None, verbose=1):
        """
        Evaluate CTC predictions on the given source data.

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

        if y is None:
            return None, None

        metrics, evaluations = self.model.ctc_evaluator(x=x,
                                                        y=y,
                                                        steps=steps,
                                                        probabilities=probabilities,
                                                        verbose=verbose)

        return metrics, evaluations

    def evaluate_synthesis(self, x, y, steps, verbose=1):
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

        if y is None:
            return None, None

        metrics, evaluations = self.model.image_evaluator(x=x,
                                                          y=y,
                                                          steps=steps,
                                                          verbose=verbose)

        return metrics, evaluations

    def get_run_info(self, run_context=None, new_context=False):
        """
        Get information about the current MLflow run.

        Parameters
        ----------
        run_context : MLflow run, optional
            MLflow Run object to set as the current run.
        new_context : bool, optional
            Create a new run context.

        Returns
        -------
        dict
            A dict containing the run ID, run name and artifacts path.
        """

        run_id = None
        run_name = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        artifact_path = None

        if run_context is not None:
            self.run_context = run_context

        if self.run_context is not None and not new_context:
            run_id = self.run_context.info.run_id
            run_name = self.run_context.info.run_name
            artifact_path = self.run_context.info.artifact_uri.replace('file://', '')

        info = {
            'id': run_id,
            'name': run_name,
            'artifact_path': artifact_path,
        }

        return info

    def save_context(self,
                     params=None,
                     dataset=None,
                     augmentor=None,
                     model=None,
                     metrics=None,
                     evaluations=None,
                     evaluation_images=None,
                     prefix='test',
                     suffix=None,
                     new_context=False):
        """
        Save relevant context information to MLflow and log files.

        Parameters
        ----------
        params : dict or argparse.Namespace, optional
            Parameters to be logged.
        dataset : Dataset instance or None, optional
            Dataset object instance.
        augmentor : Augmentor instance or None, optional
            Augmentor object instance.
        model : Model instance or None, optional
            Model object instance.
        metrics : dict or None, optional
            Model metrics.
        evaluations : list or None, optional
            Model evaluation data.
        evaluation_images: list or None, optional
            Model evaluation images.
        prefix : str, optional
            Prefix used in the metric logs.
        suffix : str, optional
            Suffix used in the metric logs.
        new_context : bool, optional
            Create a new run context.
        """

        if new_context and self.run_context is not None:
            new_context = bool(self.run_context.data.params)

        run_info = self.get_run_info(new_context=new_context)

        def log_content(label, content):
            if content is not None:
                logs_path = os.path.join(run_info['artifact_path'], 'logs')
                os.makedirs(logs_path, exist_ok=True)

                filepath = os.path.join(logs_path, f"{label}.log")

                if isinstance(content, dict) or isinstance(content, list):
                    filepath = filepath.replace('.log', '.json')
                    content = json.dumps(content, indent=4, ensure_ascii=False)

                with open(filepath, 'w') as f:
                    f.write(f"{content}".strip())

        def log_metric(label, content):
            if content is not None:
                for key, value in content.items():
                    mlflow.log_metric(label.replace('metrics', key), value)

        def log_params(content):
            if content is not None:
                params_dict = params if isinstance(params, dict) else vars(params)
                mlflow.log_params(params_dict)

        def log_images(label, images):
            if images is not None:
                evaluation_path = os.path.join(run_info['artifact_path'], 'synthesis', label)
                os.makedirs(evaluation_path, exist_ok=True)

                for i, image in enumerate(images):
                    authentic_filepath = os.path.join(evaluation_path, f"{i+1}_authentic.png")
                    generated_filepath = os.path.join(evaluation_path, f"{i+1}_generated.png")

                    cv2.imwrite(authentic_filepath, image[0])
                    cv2.imwrite(generated_filepath, image[1])

        with mlflow.start_run(run_id=run_info['id'], run_name=run_info['name']) as run:
            run_info = self.get_run_info(run_context=run)

            log_params(params)
            log_content('data', dataset)
            log_content('augmentor', augmentor)
            log_content('model', model)

            evaluation_label = f"{prefix or ''}_evaluations_{suffix or ''}".strip('_')
            log_content(evaluation_label, evaluations)
            log_images(evaluation_label, evaluation_images)

            metric_label = f"{prefix or ''}_metrics_{suffix or ''}".strip('_')
            log_content(metric_label, metrics)
            log_metric(metric_label, metrics)

            mlflow.end_run()

    @staticmethod
    def get_tokenizer(synthesis=None,
                      synthesis_run_index=None,
                      recognition=None,
                      recognition_run_index=None,
                      experiment_name=None,
                      status_finished=False):
        """
        Retrieves a tokenizer from MLflow artifacts.

        Parameters
        ----------
        synthesis : str, optional
            Identification for synthesis model.
        synthesis_run_index : int, optional
            Run index for the synthesis model.
        recognition : str, optional
            Identification for recognition model.
        recognition_run_index : int, optional
            Run index for the recognition model.
        experiment_name : str, optional
            MLflow experiment name.
        status_finished : bool, optional
            Restrict run index status.

        Returns
        -------
        tuple
            (tokenizer, run_context) or (None, None) if not found.
        """

        Graphite().fix_mlflow_artifacts_path()

        def get_artifacts_path(tag_name, tag_value, run_index):
            if run_index is not None:
                experiment = mlflow.set_experiment(experiment_name or 'Default')
                experiment_ids = [experiment.experiment_id]

                filter_string = f"tag.graphite.{tag_name}='{tag_value}'"

                if status_finished:
                    filter_string = f"status='FINISHED' AND {filter_string}"

                df = mlflow.search_runs(experiment_ids=experiment_ids,
                                        filter_string=filter_string,
                                        order_by=['tags.mlflow.runName ASC'])

                if not df.empty:
                    df['artifact_uri'] = df['artifact_uri'].str.replace('file://', '')

                    df['valid'] = df['artifact_uri'].apply(
                        lambda x: bool(glob.glob(os.path.join(x, '**', '*.h5'), recursive=True)))

                    df = df[df['valid']].reset_index(drop=True)

                    if run_index < len(df):
                        run = mlflow.get_run(df.iloc[run_index]['run_id'])
                        artifact_path = run.info.artifact_uri.replace('file://', '')
                        return run, artifact_path

            return None, None

        s_run, s_path = get_artifacts_path('synthesis', synthesis, synthesis_run_index)
        r_run, r_path = get_artifacts_path('recognition', recognition, recognition_run_index)

        tokenizer = None
        run_context = s_run or r_run
        artifacts_path = s_path or r_path

        if artifacts_path:
            tokenizer_uri = os.path.join(artifacts_path, 'tokenizer.pkl')

            if os.path.isfile(tokenizer_uri):
                try:
                    with open(tokenizer_uri, 'rb') as f:
                        tokenizer = pickle.load(f)
                except Exception as e:
                    print(f"Error loading tokenizer: {e}")

        if run_context:
            print('==================================================')
            print(f"{'Run context'.center(50)}")
            print('--------------------------------------------------')
            print(f"{'experiment_id':<{25}}: {run_context.info.experiment_id[:23]}")
            print(f"{'run_id':<{25}}: {run_context.info.run_id[:23]}")
            print(f"{'run_name':<{25}}: {run_context.info.run_name[:23]}")

        return tokenizer, run_context

    @staticmethod
    def fix_mlflow_artifacts_path():
        """
        This static method addresses the issue where MLflow artifact paths become
            incorrect after moving the MLflow folder, by updating the paths in 'meta.yaml' files.

        Notes
        -----
        Current workaround for fixing paths when mlflow folder is moved.
        GitHub Issue: https://github.com/mlflow/mlflow/issues/3144
        """

        artifact_path_keys = {'artifact_location': '', 'artifact_uri': 'artifacts'}
        meta_files = glob.glob(os.path.join('mlruns', '**', 'meta.yaml'), recursive=True)

        for metadata_file in meta_files:
            with open(metadata_file, 'r') as f:
                yaml_file = yaml.safe_load(f)

            experiment_id = metadata_file.split('/')[1]
            update_needed = False

            if yaml_file.get('experiment_id', experiment_id) != experiment_id:
                for key in artifact_path_keys:
                    if yaml_file.get(key):
                        yaml_file[key] = yaml_file[key].replace(yaml_file['experiment_id'], experiment_id)

                yaml_file['experiment_id'] = experiment_id
                update_needed = True

            for key in artifact_path_keys:
                new_path = os.path.dirname(os.path.abspath(metadata_file))
                new_path = os.path.join(new_path, artifact_path_keys[key])
                new_path = f"file://{new_path}".rstrip('/')

                if yaml_file.get(key, new_path) != new_path:
                    yaml_file[key] = new_path
                    update_needed = True

            if update_needed:
                with open(metadata_file, 'w') as f:
                    yaml.dump(yaml_file, f, default_flow_style=False)
