import os
import re
import time
import json
import string
import shutil
import pickle
import mlflow
import datetime
import importlib
import numpy as np
import editdistance
import tensorflow as tf


class Model():
    """
    General optical model management.
    """

    def __init__(self,
                 network,
                 pad_value=255,
                 experiment_name='Default',
                 artifact_path='mlruns',
                 seed=None):
        """
        Initializes a new instance of the Model class.

        Parameters
        ----------
        network : str
            The name of the network module to be used.
        pad_value : int, optional
            Padding value. Default is 255.
        experiment_name : str, optional
            Specify MLflow experiment name. Default is 'Default'.
        artifact_path : str, optional
            Path name to track the model. Default is 'mlruns'.
        seed : int, optional
            Seed for tensorflow operations. Default is None.
        """

        tf.random.set_seed(seed)
        # tf.config.set_visible_devices([], 'GPU')

        mlflow.set_tracking_uri(artifact_path)

        self.network = network
        self.pad_value = pad_value
        self.artifact_path = artifact_path
        self.seed = seed

        self.experiment_name = experiment_name
        self.experiment = mlflow.set_experiment(experiment_name)

        self.run_name = None
        self.run = None
        self.run_context = None

        self.model = None
        self.tokenizer = None

        self.optimizer = None
        self.learning_rate = None
        self.summary = None

        self.loss_logger = Logger(role='loss')
        self.training_logger = Logger(role='training')
        self.test_logger = Logger(role='test')
        self.evaluation_logger = Logger(role='evaluation')
        self.samples_logger = Logger(role='samples')

        self._network = self._import_network(self.network)

    def __repr__(self):
        """
        Returns a string representation of the object with useful information.

        Returns
        -------
        str
            The string representation of the object.
        """

        info = f"""
            Model Configuration\n
            Network                     {self.network}
            Padding Value               {self.pad_value}
            Experiment Name             {self.experiment_name}
            Seed                        {self.seed}

            Optimizer                   {self.optimizer or '-'}
            Learning Rate               {self.learning_rate or '-'}

            Summary\n\n                 {self.summary or '-'}
        """

        info = '\n'.join([x.strip() for x in info.splitlines()]).strip()

        return info

    def to_dict(self):
        """
        Convert the class object attributes to a dictionary.

        Returns
        -------
        dict
            A dictionary with the class attributes.
        """

        attributes = {
            'network': self.network,
            'pad_value': self.pad_value,
            'experiment_name': self.experiment_name,
            'optimizer': self.optimizer,
            'learning_rate': self.learning_rate,
            'seed': self.seed,
        }

        return attributes

    def compile(self, tokenizer=None, learning_rate=1e-3, run_index=None):
        """
        Compiles the model.

        Parameters
        ----------
        tokenizer : object
            The Tokenizer object used for tokenizing the input data. Default is None.
        learning_rate : float, optional
            The learning rate for the optimizer in the model. Default is 1e-3.
        run_index : int
            The run index which the context will be loaded. Default is None.
        """

        model_uri = None

        if run_index is not None:
            runs_df = mlflow.search_runs(experiment_ids=[self.experiment.experiment_id],
                                         filter_string=f"tags.mlflow.network='{self.network}'",
                                         order_by=['tags.mlflow.runName ASC'])

            if not runs_df.empty and run_index < len(runs_df):
                self.run_context = mlflow.get_run(runs_df.iloc[run_index]['run_id'])
                artifacts_uri = os.path.join(self.run_context.info.artifact_uri, 'artifacts')

                model_uri = os.path.join(artifacts_uri, 'model.keras')
                tokenizer_uri = os.path.join(artifacts_uri, 'tokenizer.pkl')

                if os.path.isfile(model_uri) and os.path.isfile(tokenizer_uri):
                    with open(tokenizer_uri, 'rb') as f:
                        tokenizer = pickle.load(f)
                else:
                    print("Model or tokenizer files do not exist.")
                    model_uri = None

        if tokenizer is None:
            print("Tokenizer is required.")
            return

        self.tokenizer = tokenizer

        self.model = self._network(self.tokenizer.shape, self.pad_value)
        self.model = self.model.compile_model(learning_rate=learning_rate, loss_func=self.ctc_loss_func)

        if model_uri is not None:
            self.model.load_weights(model_uri)

        self.optimizer = self.model.optimizer.get_config()['name']
        self.learning_rate = self.model.optimizer.get_config()['learning_rate']

        self.summary = []
        self.model.summary(print_fn=lambda x: self.summary.append(x))
        self.summary = '\n'.join(self.summary)

    def save_context(self,
                     dataset=None,
                     augmentor=None,
                     spelling=None,
                     baseline_metrics=None,
                     spelling_metrics=None):
        """
        Logs the context of the run including metrics, parameters, and artifacts.

        Parameters
        ----------
        dataset : object, optional
            The dataset used. Default is None.
        augmentor : object, optional
            The data augmentor used. Default is None.
        spelling : object, optional
            The spelling used. Default is None.
        baseline_metrics : list of tuples, optional
            The metrics obtained. Default is None.
        spelling_metrics : list of tuples, optional
            The metrics obtained from spelling. Default is None.
        """

        run_id = None
        run_name = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        if self.run is not None:
            run_id = self.run.info.run_id
            run_name = self.run.info.run_name

        with mlflow.start_run(run_id=run_id, run_name=run_name) as run:
            self.run = run

            artifacts_uri = self.run.info.artifact_uri
            os.makedirs(artifacts_uri, exist_ok=True)

            # Tag
            mlflow.set_tags({'mlflow.network': self.network})

            # Metrics
            if baseline_metrics is not None:
                dict_metrics = {}

                for i, metric in enumerate(baseline_metrics):
                    dict_metrics[f"baseline_top_path_{i+1}_cer"] = metric[0]
                    dict_metrics[f"baseline_top_path_{i+1}_wer"] = metric[1]
                    dict_metrics[f"baseline_top_path_{i+1}_ler"] = metric[2]
                    dict_metrics[f"baseline_top_path_{i+1}_ser"] = metric[3]

                mlflow.log_metrics(metrics=dict_metrics)

            # Metrics from spelling
            if spelling_metrics is not None:
                dict_metrics = {}

                for i, metric in enumerate(spelling_metrics):
                    dict_metrics[f"spelling_top_path_{i+1}_cer"] = metric[0]
                    dict_metrics[f"spelling_top_path_{i+1}_wer"] = metric[1]
                    dict_metrics[f"spelling_top_path_{i+1}_ler"] = metric[2]
                    dict_metrics[f"spelling_top_path_{i+1}_ser"] = metric[3]

                mlflow.log_metrics(metrics=dict_metrics)

            # Parameters
            dict_params = {
                **(dataset.to_dict() if dataset is not None else {}),
                **(dataset.tokenizer.to_dict() if dataset is not None else {}),
                **(augmentor.to_dict() if augmentor is not None else {}),
                **(spelling.to_dict() if spelling is not None else {}),
                **self.to_dict(),
                **(self.training_logger.to_dict() if self.training_logger.touched else {}),
                **(self.test_logger.to_dict() if self.test_logger.touched else {}),
            }

            if self.run_context is not None:
                dict_params = {**self.run_context.data.params, **dict_params}

            mlflow.log_params(dict_params)

            # Logs
            filelogs = [
                (dataset if dataset is not None else None, 'dataset.log'),
                (dataset.tokenizer if dataset is not None else None, 'tokenizer.log'),
                (augmentor if augmentor is not None else None, 'augmentor.log'),
                (spelling if spelling is not None else None, 'spelling.log'),
                (self, 'model.log'),
                (self.loss_logger if self.loss_logger.touched else None, 'loss.log'),
                (self.training_logger if self.training_logger.touched else None, 'training.log'),
                (self.test_logger if self.test_logger.touched else None, 'test.log'),
                (self.evaluation_logger if self.evaluation_logger.touched else None, 'evaluation.log'),
                (self.samples_logger if self.samples_logger.touched else None, 'samples.log'),
            ]

            for log, filename in filelogs:
                if log is None and self.run_context is None:
                    continue

                log_uri = os.path.join(artifacts_uri, 'logs', filename)
                os.makedirs(os.path.dirname(log_uri), exist_ok=True)

                if log is None:
                    log_context = os.path.join(self.run_context.info.artifact_uri, 'logs', filename)

                    if os.path.isfile(log_context):
                        shutil.copyfile(log_context, log_uri)
                        mlflow.log_artifact(log_uri, artifact_path='logs')

                else:
                    with open(log_uri, 'w') as f:
                        f.write(str(log).strip())

                    mlflow.log_artifact(log_uri, artifact_path='logs')

            # Tokenizer
            if self.tokenizer is not None:
                tokenizer_uri = os.path.join(artifacts_uri, 'artifacts', 'tokenizer.pkl')
                os.makedirs(os.path.dirname(tokenizer_uri), exist_ok=True)

                with open(tokenizer_uri, 'wb') as f:
                    pickle.dump(self.tokenizer, f)

                mlflow.log_artifact(tokenizer_uri, artifact_path='artifacts')

            # Model
            if self.model is not None:
                model_uri = os.path.join(artifacts_uri, 'artifacts', 'model.keras')
                os.makedirs(os.path.dirname(model_uri), exist_ok=True)

                self.model.save(model_uri)
                mlflow.log_artifact(model_uri, artifact_path='artifacts')

    def fit(self,
            training_data,
            training_steps=None,
            validation_data=None,
            validation_steps=None,
            plateau_factor=0.5,
            plateau_cooldown=10,
            plateau_patience=20,
            patience=60,
            epochs=1000,
            verbose=1):
        """
        Trains the model.

        Parameters
        ----------
        training_data : array-like
            The training data to be used.
        training_steps : int, optional
            The number of steps for each training epoch, by default None.
        validation_data : array-like, optional
            The validation data to be used, by default None.
        validation_steps : int, optional
            The number of steps for each validation run, by default None.
        plateau_factor : float, optional
            Factor by which the learning rate will be reduced, by default 0.5.
        plateau_cooldown : int, optional
            The number of epochs to wait before resuming normal operation after lr has been reduced, by default 10.
        plateau_patience : int, optional
            The number of epochs with no improvement after which learning rate will be reduced, by default 20.
        patience : int, optional
            The number of epochs with no improvement after which training will be stopped, by default 60.
        epochs : int, optional
            The number of epochs to train the model, by default 1000.
        verbose : int, optional
            Verbosity mode, by default 1.

        Returns
        -------
        history object
            Object detailing training and validation progress.
        """

        start_time = time.time()

        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                mode='min',
                monitor='val_loss',
                min_delta=1e-4,
                patience=patience,
                start_from_epoch=0,
                restore_best_weights=True,
                verbose=verbose,
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                mode='min',
                monitor='val_loss',
                min_lr=1e-4,
                min_delta=1e-4,
                factor=plateau_factor,
                cooldown=plateau_cooldown,
                patience=plateau_patience,
                verbose=verbose,
            ),
        ]

        run_name = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        with mlflow.start_run(run_name=run_name) as run:
            self.run = run

            history = self.model.fit(x=training_data,
                                     steps_per_epoch=training_steps,
                                     validation_data=validation_data,
                                     validation_steps=validation_steps,
                                     callbacks=callbacks,
                                     epochs=epochs,
                                     verbose=verbose)

        end_time = time.time()
        total_time = end_time - start_time

        self.loss_logger.set_loss_info(history.history)
        self.training_logger.set_training_info(total_time, history.history, training_data, training_steps)

        return history

    def predict(self,
                test_data,
                test_steps,
                top_paths=1,
                beam_width=30,
                ctc_decode=True,
                token_decode=True,
                verbose=1):
        """
        Generates predictions on the given test data using the model.

        Parameters
        ----------
        test_data : array-like
            The test data to generate predictions for.
        test_steps : int
            The total number of steps.
        top_paths : int, optional
            The number of top paths to extract from the predictions, by default 1.
        beam_width : int, optional
            The width of the beam for the CTC decoder, by default 25.
        ctc_decode : bool, optional
            If True, applies CTC decoding to the predictions, by default True.
        token_decode : bool, optional
            If True, decodes the tokens to their corresponding characters, by default True.
        verbose : int, optional
            Verbosity mode, 0 or 1. By default 1.

        Returns
        -------
        tuple of np.ndarray
            The first is the predictions, and the second is the probabilities.
        """

        start_time = time.time()

        predictions = self.model.predict(x=test_data, steps=test_steps, verbose=verbose)
        predictions, probabilities = np.log(predictions + 1e-7), np.array([])

        if ctc_decode:
            progbar = tf.keras.utils.Progbar(target=predictions.shape[1], unit_name='path_decode', verbose=verbose)

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
                for j in range(top_paths):
                    sparse_decoded = tf.sparse.to_dense(decoded[j], default_value=-1)
                    paddings = [[0, 0], [0, predictions.shape[2] - tf.reduce_max(tf.shape(sparse_decoded)[1])]]
                    decoded_pads.append(tf.pad(sparse_decoded, paddings=paddings, constant_values=-1))

                decoded_paths.append(decoded_pads)
                probabilities_list.append(tf.exp(log_probabilities))

                progbar.update(i + 1)

            predictions = np.transpose(tf.stack(decoded_paths, axis=1), (0, 2, 1, 3))
            probabilities = np.transpose(tf.stack(probabilities_list, axis=1), (2, 0, 1))

            if token_decode:
                predictions = np.array([[[self._format_text(line) for line in self.tokenizer.decode(top_path)]
                                         for top_path in top_paths] for top_paths in predictions], dtype=object)

        end_time = time.time()
        total_time = end_time - start_time

        self.test_logger.set_test_info(total_time, test_data, test_steps)

        return predictions, probabilities

    def evaluate(self,
                 partition,
                 baseline_predictions=None,
                 spelling_predictions=None,
                 share_top_paths=False):
        """
        Computes error metrics based on model's predictions.

        Parameters
        ----------
        partition : dict
            The data partition for evaluation.
        baseline_predictions : numpy.ndarray, optional
            List of baseline predictions.
        spelling_predictions : numpy.ndarray, optional
            List of spelling predictions.
        share_top_paths : bool, optional
            If True, consider previous paths to the metrics. Default is False.

        Returns
        -------
        numpy.ndarray
            Array of error metrics for each top path.
        """

        if baseline_predictions is None and spelling_predictions is None:
            return np.zeros((0, 4), dtype=np.float32), np.zeros((0, len(partition['raw']), 5), dtype=object)

        all_predictions = [(baseline_predictions, 'baseline'), (spelling_predictions, 'spelling')]
        results = []

        for predictions, origin in all_predictions:
            if predictions is None:
                continue

            metrics = np.zeros((len(predictions), 4), dtype=np.float32)
            samples = np.zeros((len(predictions), len(partition['raw']), 5), dtype=object)

            # Metrics
            for top_path in range(len(predictions)):
                curr_predict = self._get_shared_paths(partition, predictions=predictions[:top_path+1]) \
                    if share_top_paths else predictions[top_path:top_path+1]

                samples_error_rates = []

                for index in range(len(partition['raw'])):
                    _, _, label = partition['raw'][index]
                    pred = curr_predict[0, index]
                    error_rates = [0, 0, 0, 0]

                    for true_label, pred_label in zip(label, pred):
                        true_label = self._format_text(true_label)
                        pred_label = self._format_text(pred_label)

                        # Character
                        character_error_rate = self._calculate_metric(list(true_label), list(pred_label))
                        error_rates[0] += character_error_rate

                        # Word
                        word_error_rate = self._calculate_metric(true_label.split(), pred_label.split())
                        error_rates[1] += word_error_rate

                        # Line
                        line_error_rate = self._calculate_metric([true_label], [pred_label])
                        error_rates[2] += line_error_rate / len(label)

                    true_label = self._format_text(' '.join(label))
                    pred_label = self._format_text(' '.join(pred))

                    # Sequence
                    sequence_error_rate = self._calculate_metric([true_label], [pred_label])
                    error_rates[3] += sequence_error_rate

                    samples_error_rates.append(error_rates)

                samples_error_rates = np.array(samples_error_rates)
                metrics[top_path, :] = np.mean(samples_error_rates, axis=0)

                # Samples
                indices = np.argsort(samples_error_rates[:, 0])

                raw = partition['raw'][indices].tolist()
                pred = curr_predict[0, indices].tolist()

                err = [','.join([f"{y:.4f}" for y in x]) for x in samples_error_rates[indices].tolist()]
                err = [['top_path,cer,wer,ler,ser', f"{top_path + 1},{x}"] for x in err]

                samples[top_path, :, :] = np.array([r + [p] + [e] for r, p, e in zip(raw, pred, err)], dtype=object)

            self.evaluation_logger.set_evaluation_info(metrics, origin)
            self.samples_logger.set_samples_info(samples, origin)

            results.append(metrics)

        if len(results) == 1:
            results = results[0]

        return results

    def _import_network(self, network):
        """
        Imports the network module with the given name.

        Parameters
        ----------
        network : str
            The name of the network module to be imported.
        """

        module_name = importlib.util.resolve_name(f".network.{network}", __package__)
        module_spec = importlib.util.find_spec(module_name)
        assert module_spec is not None, "network file must be created"

        module = importlib.import_module(module_name, __package__)

        class_name = 'Network'
        assert hasattr(module, class_name), f"`{class_name}` class must be created"

        network = getattr(module, class_name)

        return network

    def _get_shared_paths(self, partition, predictions):
        """
        Finds the best predictions among the top paths.

        Parameters
        ----------
        partition : dict
            The data partition.
        predictions : numpy.ndarray
            Array of model's predictions.

        Returns
        -------
        numpy.ndarray
            Array of top predictions.
        """

        best_paths = len(predictions)
        data_length = len(partition['raw'])

        metrics = np.zeros((best_paths, data_length), dtype=np.float32)

        for best_path in range(best_paths):
            for index in range(data_length):
                _, _, label = partition['raw'][index]
                pred = predictions[best_path, index]

                true_label = self._format_text(' '.join(label))
                pred_label = self._format_text(' '.join(pred))

                error_rate = self._calculate_metric(list(true_label), list(pred_label))
                metrics[best_path, index] = error_rate

        best_indices = np.argmin(metrics, axis=0)
        best_predictions = np.zeros_like(predictions[0])

        for i, best_index in enumerate(best_indices):
            best_predictions[i] = predictions[best_index, i]

        best_predictions = np.expand_dims(best_predictions, axis=0)

        return best_predictions

    def _format_text(self, text):
        """
        Clean and format the input text.

        Parameters
        ----------
        text : str
            The input text to be cleaned.

        Returns
        -------
        str
            The formatted text.
        """

        text = re.sub(f'([{re.escape(string.punctuation)}])', r' \1 ', text)
        text = re.sub(r'\s+', ' ', text).strip()

        return text

    def _calculate_metric(self, true_label, pred_label):
        """
        Calculates the error rate between true label and predicted label.

        Parameters
        ----------
        true_labels : list
            List of true labels.
        pred_label : list
            List of predicted labels.

        Returns
        -------
        float
            The error rate between true labels and predicted label.
        """

        distance = editdistance.eval(true_label, pred_label)
        error_rate = distance / max(len(true_label), len(pred_label))

        return error_rate

    @staticmethod
    def ctc_loss_func(y_true, y_pred):
        """
        Computes the CTC (Connectionist Temporal Classification) loss.

        Parameters
        ----------
        y_true : array-like
            The ground truth labels.
        y_pred : array-like
            The predicted labels.
        """

        y_true = tf.reshape(y_true, (tf.shape(y_true)[0], -1))
        y_pred = tf.reshape(y_pred, (tf.shape(y_pred)[0], -1, tf.shape(y_pred)[-1]))

        label_length = tf.math.count_nonzero(y_true, axis=-1, keepdims=True, dtype=tf.int32)
        logit_length = tf.reduce_sum(tf.reduce_sum(y_pred, axis=-1), axis=-1, keepdims=True)

        loss = tf.keras.backend.ctc_batch_cost(y_true, y_pred, logit_length, label_length)

        # y_true = tf.reshape(y_true, (tf.shape(y_true)[0], -1))
        # y_pred = tf.reshape(y_pred, (tf.shape(y_pred)[0], -1, tf.shape(y_pred)[-1]))

        # labels = tf.sparse.from_dense(y_true)
        # logits = tf.math.log(tf.transpose(y_pred, perm=[1, 0, 2]) + 1e-7)
        # logits = tf.transpose(tf.math.log(y_pred + 1e-7), perm=[1, 0, 2])

        # label_length = tf.math.count_nonzero(y_true, axis=-1)
        # logit_length = tf.reduce_sum(tf.reduce_sum(y_pred, axis=-1), axis=-1)

        # loss = tf.nn.ctc_loss(labels=tf.cast(labels, dtype=tf.int32),
        #                       logits=tf.cast(logits, dtype=tf.float32),
        #                       label_length=tf.cast(label_length, dtype=tf.int32),
        #                       logit_length=tf.cast(logit_length, dtype=tf.int32),
        #                       logits_time_major=True,
        #                       blank_index=-1)

        loss = tf.reduce_mean(loss)

        return loss


class Logger():
    """
    Class to log and store training, test and samples information.
    """

    def __init__(self, role):
        """
        Initialize the Logger object.

        Parameters
        ----------
        role : str
            The role for which this logger is used.
        """

        self.role = role
        self.touched = False

        # Loss
        self.loss_epoch = 0
        self.loss_training = 0
        self.loss_validation = 0
        self.loss_history = []

        # Training
        self.training_total_data = 0
        self.training_total_epochs = 0
        self.training_total_steps = 0
        self.training_time = 0
        self.training_time_per_epoch = 0
        self.training_time_per_step = 0
        self.training_time_per_item = 0

        # Test
        self.test_total_data = 0
        self.test_total_epochs = 0
        self.test_total_steps = 0
        self.test_time = 0
        self.test_time_per_epoch = 0
        self.test_time_per_step = 0
        self.test_time_per_item = 0

        # Evaluation
        self.evaluation = {}

        # Samples
        self.samples = {}

    def __repr__(self):
        """
        Returns a string representation of the object with useful information.

        Returns
        -------
        str
            The string representation of the object.
        """

        info = ""

        if self.role == 'loss':
            loss_history = '\n'.join(self.loss_history)

            info = f"""
                Loss\n
                Best Epoch Loss             {self.loss_epoch}
                Best Training Loss          {self.loss_training}
                Best Validation Loss        {self.loss_validation}

                Loss History\n\n            {loss_history or '-'}
            """

            info = '\n'.join([x.strip() for x in info.splitlines()]).strip()

        elif self.role == 'training':
            info = f"""
                Training\n
                Total Data                  {self.training_total_data}
                Total Epochs                {self.training_total_epochs}
                Total Steps                 {self.training_total_steps}

                Time                        {self.training_time}
                Time per Epoch              {self.training_time_per_epoch}
                Time per Step               {self.training_time_per_step}
                Time per Item               {self.training_time_per_item}
            """

            info = '\n'.join([x.strip() for x in info.splitlines()]).strip()

        elif self.role == 'test':
            info = f"""
                Test\n
                Total Data                  {self.test_total_data}
                Total Epochs                {self.test_total_epochs}
                Total Steps                 {self.test_total_steps}

                Time                        {self.test_time}
                Time per Epoch              {self.test_time_per_epoch}
                Time per Step               {self.test_time_per_step}
                Time per Item               {self.test_time_per_item}
            """

            info = '\n'.join([x.strip() for x in info.splitlines()]).strip()

        elif self.role == 'evaluation':
            evaluation = '\n\n'.join([f"{i}\n" + '\n'.join(self.evaluation[i]) for i in self.evaluation.keys()])

            info = f"""
                Evaluation\n\n              {evaluation or '-'}
            """

            info = '\n'.join([x.strip() for x in info.splitlines()]).strip()

        elif self.role == 'samples':
            info = json.dumps(self.samples, indent=2, ensure_ascii=False, default=lambda x: str(x))

        return info

    def to_dict(self):
        """
        Convert the class object attributes to a dictionary.

        Returns
        -------
        dict
            A dictionary with the class attributes.
        """

        attributes = {}

        if self.role == 'loss':
            attributes = {
                'loss_epoch': self.loss_epoch,
                'loss_training': self.loss_training,
                'loss_validation': self.loss_validation,
            }

        elif self.role == 'training':
            attributes = {
                'training_total_data': self.training_total_data,
                'training_total_epochs': self.training_total_epochs,
                'training_total_steps': self.training_total_steps,
                'training_time': self.training_time,
                'training_time_per_epoch': self.training_time_per_epoch,
                'training_time_per_step': self.training_time_per_step,
                'training_time_per_item': self.training_time_per_item,
            }

        elif self.role == 'test':
            attributes = {
                'test_total_data': self.test_total_data,
                'test_total_epochs': self.test_total_epochs,
                'test_total_steps': self.test_total_steps,
                'test_time': self.test_time,
                'test_time_per_epoch': self.test_time_per_epoch,
                'test_time_per_step': self.test_time_per_step,
                'test_time_per_item': self.test_time_per_item,
            }

        elif self.role == 'samples':
            attributes = {
                'samples': self.samples,
            }

        return attributes

    def set_loss_info(self, loss_history):
        """
        Set the training information.

        Parameters
        ----------
        loss_history : dict
            The training history object.
        """

        loss = loss_history['loss']
        val_loss = loss_history['val_loss']

        best_val_loss = np.inf
        best_loss = None
        best_loss_epoch = -1
        epochs = []

        for epoch, (loss_value, val_loss_value) in enumerate(zip(loss, val_loss)):
            is_best = ''

            if val_loss_value < best_val_loss:
                is_best = '*'
                best_loss_epoch = epoch + 1
                best_loss = loss_value
                best_val_loss = val_loss_value

            epochs.append([epoch + 1, loss_value, val_loss_value, is_best])

        self.loss_history = ['epoch,loss,val_loss,best']

        for epoch in epochs:
            epoch, loss_value, val_loss_value, is_best = epoch
            self.loss_history += [f"{epoch},{loss_value},{val_loss_value},{is_best}"]

        self.loss_epoch = best_loss_epoch
        self.loss_training = best_loss
        self.loss_validation = best_val_loss

        self.touched = True

    def set_training_info(self, total_time, loss_history, training_data, training_steps):
        """
        Set the training information.

        Parameters
        ----------
        total_time : float
            The total training time.
        loss_history : dict
            The training history object.
        training_data : data generator
            The training data.
        training_steps : int
            The number of training steps.
        """

        training_total_data = np.sum([len(next(training_data)[0]) for _ in range(training_steps)])

        self.training_total_epochs = len(loss_history['loss'])

        self.training_total_data = training_total_data
        self.training_total_steps = training_steps

        self.training_time = str(datetime.timedelta(seconds=total_time))

        time_per_epoch = total_time / self.training_total_epochs
        self.training_time_per_epoch = str(datetime.timedelta(seconds=time_per_epoch))

        time_per_step = total_time / self.training_total_steps
        self.training_time_per_step = str(datetime.timedelta(seconds=time_per_step))

        time_per_item = total_time / (self.training_total_epochs * self.training_total_data)
        self.training_time_per_item = str(datetime.timedelta(seconds=time_per_item))

        self.touched = True

    def set_test_info(self, total_time, test_data, test_steps):
        """
        Set the test information.

        Parameters
        ----------
        total_time : float
            The total test time.
        test_data : data generator
            The test data.
        test_steps : int
            The number of test steps.
        """

        test_total_data = np.sum([len(next(test_data)[0]) for _ in range(test_steps)])

        self.test_total_data = test_total_data
        self.test_total_epochs = 1
        self.test_total_steps = self.test_total_epochs * test_steps

        self.test_time = str(datetime.timedelta(seconds=total_time))

        time_per_epoch = total_time / self.test_total_epochs
        self.test_time_per_epoch = str(datetime.timedelta(seconds=time_per_epoch))

        time_per_step = total_time / self.test_total_steps
        self.test_time_per_step = str(datetime.timedelta(seconds=time_per_step))

        time_per_item = total_time / (self.test_total_epochs * self.test_total_data)
        self.test_time_per_item = str(datetime.timedelta(seconds=time_per_item))

        self.touched = True

    def set_evaluation_info(self, metrics, origin='baseline'):
        """
        Set the evaluation information.

        Parameters
        ----------
        metrics : data generator
            Error metrics from each top path.
        origin : str, optional
            Indicates the origin name. Default is 'baseline'.
        """

        self.evaluation[origin] = ['top_path,cer,wer,ler,ser']

        for i, x in enumerate(metrics, start=1):
            self.evaluation[origin] += [f"{i},{x[0]:.4f},{x[1]:.4f},{x[2]:.4f},{x[3]:.4f}"]

        self.touched = True

    def set_samples_info(self, samples, origin='baseline'):
        """
        Set the evaluation information.

        Parameters
        ----------
        samples : int
            Samples retrieved from prediction.
        origin : str, optional
            Indicates the origin name. Default is 'baseline'.
        """

        self.samples[origin] = []

        for i, top_path in enumerate(samples):
            path = [{
                'image_path': x[0],
                'bbox': x[1],
                'label': x[2],
                'prediction': x[3],
                'metric': x[4],
            } for x in top_path]

            self.samples[origin].append({f"top_path_{i + 1}": path})

        self.touched = True
