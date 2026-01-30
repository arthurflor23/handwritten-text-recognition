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
from sarah.models.components.callbacks import GANMonitor
from sarah.models.components.callbacks import TrainingLogger


class Compose():
    """
    Compose is a configurable model framework for synthesis and recognition tasks,
        integrating various components and supporting MLflow experimentation.
    """

    def __init__(self,
                 synthesis=None,
                 recognition=None,
                 spelling=None,
                 writer_identification=None,
                 image_shape=None,
                 tokenizer=None,
                 discriminator_steps=1,
                 generator_steps=1,
                 synthesis_probability=1.0,
                 experiment_name=None,
                 output_path='mlruns',
                 gpu=0,
                 seed=None):
        """
        Initializes the Compose model with specified components.

        Parameters
        ----------
        synthesis : str, optional
            Identifier for the synthesis model.
        recognition : str, optional
            Identifier for the recognition model.
        spelling : str, optional
            Identifier for the spelling correction model.
        writer_identification : str, optional
            Identifier for the writer identification model.
        image_shape : tuple, optional
            Shape of the input images.
        tokenizer : Tokenizer, optional
            Tokenizer for processing text data.
        discriminator_steps : int, optional
            The repetition of steps for discriminator training.
        generator_steps : int, optional
            The skipping steps for generator training.
        synthesis_probability : float, optional
            Synthetic data probability.
        experiment_name : str, optional
            Name of the MLflow experiment.
        output_path : str, optional
            Path to output data.
        gpu : int, list, or tuple, optional
            GPU index or sequence of indices.
        seed : int, optional
            Seed for random shuffle.
        """

        self.synthesis = synthesis
        self.recognition = recognition
        self.spelling = spelling
        self.writer_identification = writer_identification

        self.image_shape = image_shape
        self.tokenizer = tokenizer

        self.discriminator_steps = discriminator_steps
        self.generator_steps = generator_steps
        self.synthesis_probability = synthesis_probability

        self.experiment_name = experiment_name or 'Default'
        self.output_path = output_path

        self.gpu = gpu
        self.seed = seed

        self.model = None
        self.spelling_model = None
        self.run_context = None

        if self.synthesis or self.recognition or self.writer_identification:
            mlflow.set_tracking_uri(self.output_path)
            mlflow.set_experiment(self.experiment_name)

            try:
                tf.keras.backend.clear_session(free_memory=True)

                indices = gpu if isinstance(gpu, (list, tuple)) else [gpu]
                indices = [int(i) for i in indices if str(i).isdigit()]

                devices = tf.config.list_physical_devices('GPU')
                devices = [devices[i] for i in indices]

                tf.config.set_visible_devices(devices=devices, device_type='GPU')

                for device in devices:
                    tf.config.experimental.set_memory_growth(device=device, enable=True)

            except Exception:
                pass

            SynthesisModel = None
            RecognitionModel = None
            SpellingModel = None
            WriterIdentificationModel = None

            if self.synthesis:
                module = f"synthesis.{self.synthesis}"
                SynthesisModel = self._import_model(module=module, class_name='SynthesisModel')

            if self.recognition:
                module = f"recognition.{self.recognition}"
                RecognitionModel = self._import_model(module=module, class_name='RecognitionModel')

                if self.spelling:
                    module = f"spelling.{self.spelling}"
                    SpellingModel = self._import_model(module=module, class_name='SpellingModel')

            if self.writer_identification:
                module = f"writer_identification.{self.writer_identification}"
                WriterIdentificationModel = self._import_model(module=module, class_name='WriterIdentificationModel')

            if SynthesisModel and not RecognitionModel and not WriterIdentificationModel:
                self.model = SynthesisModel(name='synthesis',
                                            image_shape=self.image_shape,
                                            lexical_shape=self.tokenizer.lexical_shape,
                                            writers_shape=self.tokenizer.writers_shape,
                                            discriminator_steps=self.discriminator_steps,
                                            generator_steps=self.generator_steps,
                                            seed=self.seed)
            else:
                synthesis_params = {}

                if SynthesisModel:
                    synthesis = SynthesisModel(name='synthesis',
                                               image_shape=self.image_shape,
                                               lexical_shape=self.tokenizer.lexical_shape,
                                               writers_shape=self.tokenizer.writers_shape,
                                               discriminator_steps=self.discriminator_steps,
                                               generator_steps=self.generator_steps,
                                               seed=self.seed)
                    synthesis_params = {
                        'writer_encoder': synthesis.writer_encoder,
                        'style_encoder': synthesis.style_encoder,
                        'generator': synthesis.generator,
                        'synthesis_probability': self.synthesis_probability,
                    }

                if RecognitionModel:
                    self.model = RecognitionModel(name='recognition',
                                                  image_shape=self.image_shape,
                                                  lexical_shape=self.tokenizer.lexical_shape,
                                                  seed=self.seed,
                                                  **synthesis_params)

                    if SpellingModel:
                        self.spelling_model = SpellingModel()

                elif WriterIdentificationModel:
                    self.model = WriterIdentificationModel(name='writer_identification',
                                                           image_shape=self.image_shape,
                                                           writers_shape=self.tokenizer.writers_shape,
                                                           seed=self.seed,
                                                           **synthesis_params)

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

        width = 68

        info = "=" * width
        info += f"\n{self.__class__.__name__.center(width)}"
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

        if run_context is None:
            run_info = self.get_run_info(new_context=True)

            with mlflow.start_run(run_id=run_info['id'], run_name=run_info['name']) as run:
                run_info = self.get_run_info(run_context=run)
        else:
            run_info = self.get_run_info(run_context=run_context)
            artifact_path = os.path.join(run_info['artifact_path'], 'model', '<model>.weights.h5')

            self.model.load_weights(filepath=artifact_path)

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
            new_context = bool(glob.glob(os.path.join(path, '**', '*.weights.h5'), recursive=True))

        run_info = self.get_run_info(new_context=new_context)

        with mlflow.start_run(run_id=run_info['id'], run_name=run_info['name']) as run:
            run_info = self.get_run_info(run_context=run)

            monitor = self.model.monitor.lstrip('val_') \
                if validation_gen is None else self.model.monitor

            callbacks = [tf.keras.callbacks.SwapEMAWeights(swap_on_epoch=True)] \
                if self.model.optimizer.use_ema else []

            callbacks.extend([
                TrainingLogger(
                    mode='min',
                    monitor=monitor,
                    model_path=os.path.join(run_info['artifact_path'], 'model', '<model>.weights.h5'),
                    save_best_only=bool(self.recognition or self.writer_identification),
                    save_weights_only=True,
                    csv_path=os.path.join(run_info['artifact_path'], 'epochs.csv'),
                    csv_separator=',',
                    verbose=verbose,
                ),
                tf.keras.callbacks.EarlyStopping(
                    mode='min',
                    monitor=monitor,
                    min_delta=0,
                    patience=patience,
                    start_from_epoch=0,
                    restore_best_weights=True,
                    verbose=verbose,
                ),
            ])

            if self.recognition or self.writer_identification:
                callbacks.extend([
                    tf.keras.callbacks.ReduceLROnPlateau(
                        mode='min',
                        monitor=monitor,
                        min_lr=1e-4,
                        min_delta=0,
                        factor=plateau_factor,
                        cooldown=plateau_cooldown,
                        patience=plateau_patience,
                        verbose=verbose,
                    ),
                ])

            elif self.synthesis:
                callbacks.extend([
                    GANMonitor(
                        filepath=os.path.join(run_info['artifact_path'], 'synthesis', 'training'),
                        sample_gen=monitor_sample_gen,
                        sample_steps=monitor_sample_steps,
                        latent_dim=self.model.style_encoder.latent_dim,
                    ),
                ])

            mlflow.set_tags({'compose.synthesis': str(self.synthesis)})
            mlflow.set_tags({'compose.recognition': str(self.recognition)})
            mlflow.set_tags({'compose.writer_identification': str(self.writer_identification)})

            tokenizer_path = os.path.join(run_info['artifact_path'], 'model', 'tokenizer.pkl')
            os.makedirs(os.path.dirname(tokenizer_path), exist_ok=True)

            with open(tokenizer_path, 'wb') as f:
                pickle.dump(self.tokenizer, f)

            history = self.model.fit(x=training_gen,
                                     steps_per_epoch=training_steps,
                                     validation_data=validation_gen,
                                     validation_steps=validation_steps,
                                     callbacks=callbacks,
                                     epochs=(epochs or 1000000),
                                     shuffle=False,
                                     verbose=verbose)
            mlflow.end_run()

        if monitor in history.history:
            best_metric_index = history.history[monitor].index(min(history.history[monitor]))
            metrics = {k: history.history[k][best_metric_index] for k in history.history}

            training_metrics = {k: metrics[k] for k in metrics if k[:4] != 'val_'}
            self.save_context(metrics=training_metrics, prefix='training')

            if validation_gen is not None:
                validation_metrics = {k[4:]: metrics[k] for k in metrics if k[:4] == 'val_'}
                self.save_context(metrics=validation_metrics, prefix='validation')

        return history

    def predict_writer_identification(self, x, steps, token_decode=True, verbose=1):
        """
        Predict writers with identification model using test data.

        Parameters
        ----------
        x : Dataset generator
            Data for predictions.
        steps : int
            Number of steps for prediction.
        token_decode : bool, optional
            Decode tokens after prediction.
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

        if token_decode and self.tokenizer:
            predictions = [self.tokenizer.decode_writer(x) for x in np.argmax(predictions, axis=1)]

        return predictions

    def predict_recognition(self,
                            x,
                            steps,
                            top_paths=1,
                            beam_width=32,
                            ctc_decode=True,
                            token_decode=True,
                            verbose=1):
        """
        Make predictions on test data with CTC decoding.

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
            Decode tokens after prediction.
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

        return predictions, probabilities

    def predict_spelling(self, x, steps, verbose=1):
        """
        Make predictions with the spelling correction model.

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
            Predictions from the spelling correction model.
        """

        if x is None:
            return None

        predictions = self.spelling_model.predict(x=x, steps=steps, verbose=verbose)

        return predictions

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
        predictions = np.array((predictions + 1.0) * 127.5, dtype=np.uint8)

        return predictions

    def evaluate_writer_identification(self, x, y, steps, verbose=1):
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

        if y is None:
            return None, None

        metrics, evaluations = self.model.writer_evaluator(x=x, y=y, steps=steps, verbose=verbose)

        return metrics, evaluations

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

        metrics, evaluations = self.model.image_evaluator(x=x, y=y, steps=steps, verbose=verbose)

        return metrics, evaluations

    def get_evaluations(self):
        """
        Retrieve data, predictions, and probabilities from evaluations JSON file.

        Returns
        -------
        tuple
            Data, predictions, and probabilities
        """

        data = {'test': []}
        predictions, probabilities = [], []

        run_context = self.get_run_info()
        evaluations = os.path.join(run_context['artifact_path'], 'evaluations.json')

        if os.path.isfile(evaluations):
            with open(evaluations, 'r') as file:
                evals = json.load(file)

            for x in evals:
                data['test'].append({
                    'writer': x['writer'],
                    'image': x['image'],
                    'bbox': [],
                    'text': x['text'],
                })

                predictions.append([y['text'] for y in x['predictions']])
                probabilities.append([y['probability'] for y in x['predictions']])

        return data, predictions, probabilities

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
            title = None

            if self.run_context is None:
                title = "Run context (base)"
            elif str(self.run_context.info.run_id) != str(run_context.info.run_id):
                title = "Run context (new)"

            if title:
                pad, width = 25, 68
                print("=" * width)
                print(f"{title.center(width)}")
                print("-" * width)
                print(f"{'experiment_id':<{pad}}: {run_context.info.experiment_id}")
                print(f"{'experiment_name':<{pad}}: {self.experiment_name}")
                print(f"{'run_id':<{pad}}: {run_context.info.run_id}")
                print(f"{'run_name':<{pad}}: {run_context.info.run_name}")
                print("-" * width)

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
                filepath = os.path.join(run_info['artifact_path'], f"{label}.log")

                if isinstance(content, dict) or isinstance(content, list):
                    filepath = filepath.replace('.log', '.json')
                    content = json.dumps(content, indent=4, ensure_ascii=False)

                if hasattr(content, 'get_summary'):
                    content = content.get_summary()

                with open(filepath, 'w') as f:
                    f.write(f"{content}".strip())

        def log_metrics(label, content):
            if content is not None:
                filepath = os.path.join(run_info['artifact_path'], 'metrics.json')
                data = {label.replace('<metric>', '').replace('__', '_').strip('_'): content}

                if os.path.exists(filepath):
                    local_data = data.copy()

                    with open(filepath, 'r') as f:
                        data = json.load(f)
                        data.update(local_data)

                data = dict(sorted(data.items()))

                with open(filepath, 'w') as f:
                    f.write(json.dumps(data, indent=4, ensure_ascii=False))

                for key, value in content.items():
                    mlflow.log_metric(label.replace('<metric>', key), value)

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

            evaluation_label = f"evaluations_{suffix or ''}".strip('_')
            log_content(evaluation_label, evaluations)
            log_images(evaluation_label, evaluation_images)

            metric_label = f"{prefix or ''}_<metric>_{suffix or ''}".strip('_')
            log_metrics(metric_label, metrics)

            mlflow.end_run()

    @staticmethod
    def get_tokenizer(synthesis=None,
                      synthesis_run_id=None,
                      recognition=None,
                      recognition_run_id=None,
                      writer_identification=None,
                      writer_identification_run_id=None,
                      experiment_name=None,
                      finished_runs=False,
                      output_path='mlruns'):
        """
        Retrieves a tokenizer from MLflow artifacts.

        Parameters
        ----------
        synthesis : str, optional
            Identifier for synthesis model.
        synthesis_run_id : str or int, optional
            Run index for the synthesis model.
        recognition : str, optional
            Identifier for recognition model.
        recognition_run_id : str or int, optional
            Run index for the recognition model.
        writer_identification : str, optional
            Identifier for writer identification model.
        writer_identification_run_id : str or int, optional
            Run index for the writer identification model.
        experiment_name : str, optional
            MLflow experiment name.
        finished_runs : bool, optional
            Only finished runs for selection.
        output_path : str, optional
            Path to output data.

        Returns
        -------
        tuple
            (tokenizer, run_context) or (None, None) if not found.
        """

        Compose().fix_mlflow_artifacts_path(output_path)

        def get_artifacts_path(tag_name, tag_value, run_id):
            run, artifact_path = None, None

            if run_id is not None:
                experiment = mlflow.set_experiment(experiment_name or 'Default')
                experiment_ids = [experiment.experiment_id]

                filter_string = f"tag.compose.{tag_name}='{tag_value}'"

                if finished_runs:
                    filter_string = f"status='FINISHED' AND {filter_string}"

                df = mlflow.search_runs(experiment_ids=experiment_ids,
                                        filter_string=filter_string,
                                        order_by=['tags.mlflow.runName ASC'])

                if not df.empty:
                    df['artifact_uri'] = df['artifact_uri'].str.replace('file://', '')

                    df['valid'] = df['artifact_uri'].apply(
                        lambda x: bool(glob.glob(os.path.join(x, '**', '*.weights.h5'), recursive=True)))

                    df = df[df['valid']].reset_index(drop=True)

                    try:
                        if str(run_id).replace('-', '').isnumeric():
                            run = mlflow.get_run(df.iloc[int(run_id)]['run_id'])
                        else:
                            df = df[df['run_id'] == run_id]
                            run = mlflow.get_run(df.iloc[0]['run_id'])

                    except Exception:
                        print(f"Run ID not found: {run_id}")
                        exit(1)

                    if run is not None:
                        artifact_path = run.info.artifact_uri.replace('file://', '')

            return run, artifact_path

        s_run, s_path = get_artifacts_path('synthesis', synthesis, synthesis_run_id)
        r_run, r_path = get_artifacts_path('recognition', recognition, recognition_run_id)
        w_run, w_path = get_artifacts_path('writer_identification', writer_identification, writer_identification_run_id)

        tokenizer = None
        run_context = s_run or r_run or w_run
        artifacts_path = s_path or r_path or w_path

        if artifacts_path:
            tokenizer_uri = os.path.join(artifacts_path, 'model', 'tokenizer.pkl')

            if os.path.isfile(tokenizer_uri):
                try:
                    with open(tokenizer_uri, 'rb') as f:
                        tokenizer = pickle.load(f)

                except Exception as e:
                    print(f"Tokenizer error: {e}")
                    exit(1)

        return tokenizer, run_context

    @staticmethod
    def fix_mlflow_artifacts_path(output_path='mlruns'):
        """
        This static method addresses the issue where MLflow artifact paths become
            incorrect after moving the MLflow folder, by updating the paths in 'meta.yaml' files.

        Notes
        -----
        Current workaround for fixing paths when mlflow folder is moved.
        GitHub Issue: https://github.com/mlflow/mlflow/issues/3144

        Parameters
        ----------
        output_path : str, optional
            Path to output data.
        """

        mlflow.set_tracking_uri(output_path)

        artifact_path_keys = {'artifact_location': '', 'artifact_uri': 'artifacts'}
        meta_files = glob.glob(os.path.join(output_path, '**', 'meta.yaml'), recursive=True)

        for metadata_file in meta_files:
            with open(metadata_file, 'r') as f:
                yaml_file = yaml.safe_load(f)

            if yaml_file is None:
                continue

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
                new_path = f"file://{new_path}".removesuffix('/')

                if yaml_file.get(key, new_path) != new_path:
                    yaml_file[key] = new_path
                    update_needed = True

            if update_needed:
                with open(metadata_file, 'w') as f:
                    yaml.dump(yaml_file, f, default_flow_style=False)
