import os
import re
import time
import json
import glob
import string
import datetime
import importlib
import numpy as np
import editdistance
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class Model():
    """
    General optical model management.
    """

    def __init__(self,
                 network,
                 tokenizer,
                 pad_value=255,
                 artifact_path='mlruns',
                 seed=None):
        """
        Initializes a new instance of the Model class.

        Parameters
        ----------
        network : str
            The name of the network module to be used.
        tokenizer : object
            The Tokenizer object used for tokenizing the input data.
        pad_value : int, optional
            Padding value. Default is 255.
        artifact_path : str, optional
            The relative path to the directory where model artifacts are stored, by default 'mlruns'.
        seed : int, optional
            The random seed to ensure repeatability of results, by default None.
        """

        tf.random.set_seed(seed)
        # tf.config.set_visible_devices([], 'GPU')

        self.network = network
        self.tokenizer = tokenizer
        self.pad_value = pad_value
        self.artifact_path = artifact_path
        self.seed = seed

        self.optimizer = None
        self.learning_rate = None
        self.summary = None

        self.logger = Logger()

        self._network = self._import_network(self.network)
        self._network = self._network(self.tokenizer.shape, self.pad_value)

    def __repr__(self):
        """
        Returns a JSON-formatted string representation of the object.

        Returns
        -------
        str
            A JSON-formatted string containing the object's attributes.
        """

        attributes = {
            'network': self.network,
            'tokenizer_shape': self.tokenizer.shape,
            'pad_value': self.pad_value,
            'optimizer': self.optimizer,
            'learning_rate': self.learning_rate,
            'summary': self.summary,
            'seed': self.seed,
        }

        attributes = json.dumps(attributes, default=lambda x: str(x))

        return attributes

    def __str__(self):
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
            Tokenizer Shape             {self.tokenizer.shape}
            Padding Value               {self.pad_value}
            Seed                        {self.seed}

            Optimizer                   {self.optimizer or '-'}
            Learning Rate               {self.learning_rate or '-'}

            Summary\n\n                 {self.summary or '-'}
        """

        info = '\n'.join([x.strip() for x in info.splitlines()])

        return info

    def compile(self, learning_rate=None, run_index=None):
        """
        Compiles the model.

        Parameters
        ----------
        learning_rate : float, optional
            The learning rate for the optimizer in the model, by default None.
        run_index : int, optional
            The index of the run to be loaded, by default None.
        """

        self.model = self._network.compile_model(learning_rate=learning_rate,
                                                 loss_func=self.ctc_loss_func)

        if run_index is not None:
            runs = sorted(glob.glob(os.path.join(self.artifact_path, self.network, '*')))

            if runs and run_index < len(runs):
                model_uri = os.path.join(runs[run_index], 'model.keras')

                if os.path.exists(model_uri) and os.path.isfile(model_uri):
                    self.model.load_weights(model_uri)

        self.optimizer = self.model.optimizer.get_config()['name']
        self.learning_rate = self.model.optimizer.get_config()['learning_rate']

        self.summary = []
        self.model.summary(print_fn=lambda x: self.summary.append(x))
        self.summary = '\n'.join(self.summary)

    def fit(self,
            training_data,
            training_steps=None,
            validation_data=None,
            validation_steps=None,
            plateau_factor=0.2,
            plateau_cooldown=0,
            plateau_patience=10,
            patience=20,
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
            Factor by which the learning rate will be reduced, by default 0.2.
        plateau_cooldown : int, optional
            The number of epochs to wait before resuming normal operation after lr has been reduced, by default 0.
        plateau_patience : int, optional
            The number of epochs with no improvement after which learning rate will be reduced, by default 10.
        patience : int, optional
            The number of epochs with no improvement after which training will be stopped, by default 20.
        epochs : int, optional
            The number of epochs to train the model, by default 1000.
        verbose : int, optional
            Verbosity mode, by default 1.
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

        history = self.model.fit(x=training_data,
                                 steps_per_epoch=training_steps,
                                 validation_data=validation_data,
                                 validation_steps=validation_steps,
                                 callbacks=callbacks,
                                 epochs=epochs,
                                 verbose=verbose)

        end_time = time.time()
        total_time = end_time - start_time

        self.logger.set_training_info(history.history,
                                      training_data,
                                      training_steps,
                                      validation_data,
                                      validation_steps,
                                      total_time)

        return history

    def predict(self,
                test_data,
                test_steps,
                top_paths=1,
                beam_width=25,
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

        self.logger.set_test_info(test_data, test_steps, total_time)

        return predictions, probabilities

    def evaluate(self,
                 partition,
                 predictions,
                 share_top_paths=True,
                 prediction_samples=10,
                 origin='vanilla'):
        """
        Computes error metrics based on model's predictions.

        Parameters
        ----------
        partition : dict
            The data partition for evaluation.
        predictions : numpy.ndarray
            Array of model's predictions.
        share_top_paths : bool, optional
            If True, considers previous paths for each path metrics. Default is True.
        prediction_samples : int, optional
            Number of samples for retrieve from evaluation. Default is 10.
        origin : str, optional
            Indicates the origin name. Default is vanilla.

        Returns
        -------
        tuple of numpy.ndarray
            Error metrics and evaluated samples for each top path.
        """

        top_paths = len(predictions)
        data_length = len(partition['raw'])

        prediction_samples = min(prediction_samples, data_length)

        metrics = np.zeros((top_paths, 4), dtype=np.float32)
        samples = np.zeros((top_paths, 3, prediction_samples, 4), dtype=object)

        # Metrics
        for top_path in range(top_paths):
            current_pred = self._get_best_predictions(partition, predictions=predictions[:top_path+1]) \
                if share_top_paths else predictions[top_path:top_path+1]

            samples_error_rate = []

            for index in range(data_length):
                _, _, label = partition['raw'][index]
                pred = current_pred[0, index]

                for true_label, pred_label in zip(label, pred):
                    true_label = self._format_text(true_label)
                    pred_label = self._format_text(pred_label)

                    # Character
                    character_error_rate = self._calculate_metric(list(true_label), list(pred_label))
                    metrics[top_path, 0] += character_error_rate

                    samples_error_rate.append(character_error_rate)

                    # Word
                    word_error_rate = self._calculate_metric(true_label.split(), pred_label.split())
                    metrics[top_path, 1] += word_error_rate

                    # Line
                    line_error_rate = self._calculate_metric([true_label], [pred_label])
                    metrics[top_path, 2] += line_error_rate / len(label)

                true_label = self._format_text(' '.join(label))
                pred_label = self._format_text(' '.join(pred))

                # Sequence
                sequence_error_rate = self._calculate_metric([true_label], [pred_label])
                metrics[top_path, 3] += sequence_error_rate

            metrics[top_path, :] /= data_length

            # Samples
            sorted_indices = np.argsort(samples_error_rate)

            t_samples = sorted_indices[:prediction_samples]
            b_samples = sorted_indices[-prediction_samples:]

            m_index = (data_length // 2) - (prediction_samples // 2)
            m_samples = sorted_indices[m_index:m_index + prediction_samples]

            raw, pred = partition['raw'][t_samples].tolist(), current_pred[0, t_samples].tolist()
            samples[top_path, 0, :, :] = np.array([x + [w] for x, w in zip(raw, pred)], dtype=object)

            raw, pred = partition['raw'][m_samples].tolist(), current_pred[0, m_samples].tolist()
            samples[top_path, 1, :, :] = np.array([x + [w] for x, w in zip(raw, pred)], dtype=object)

            raw, pred = partition['raw'][b_samples].tolist(), current_pred[0, b_samples].tolist()
            samples[top_path, 2, :, :] = np.array([x + [w] for x, w in zip(raw, pred)], dtype=object)

        self.logger.set_evaluation_info(metrics, samples, origin)

        return metrics, samples

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

    def _get_best_predictions(self, partition, predictions):
        """
        Finds the best predictions based on character error rate.

        Parameters
        ----------
        partition : dict
            The data partition.
        predictions : numpy.ndarray
            Array of model's predictions.

        Returns
        -------
        numpy.ndarray
            Array of top predictions based on character error rate.
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
    Class to log and store training, and test information.
    """

    def __init__(self):
        """
        Initialize the Logger object.
        """

        self.training_batch_size = 0
        self.training_total_data = 0
        self.training_total_epochs = 0
        self.training_total_steps = 0
        self.training_time = 0
        self.training_time_per_epoch = 0
        self.training_time_per_step = 0

        self.validation_batch_size = 0
        self.validation_total_data = 0
        self.validation_total_epochs = 0
        self.validation_total_steps = 0
        self.validation_time = 0
        self.validation_time_per_epoch = 0
        self.validation_time_per_step = 0

        self.test_batch_size = 0
        self.test_total_data = 0
        self.test_total_epochs = 0
        self.test_total_steps = 0
        self.test_time = 0
        self.test_time_per_epoch = 0
        self.test_time_per_step = 0

        self.loss_epoch = 0
        self.loss_training = 0
        self.loss_validation = 0
        self.loss_history = []

        self.evaluation = {}
        self.samples = {}

    def __repr__(self):
        """
        Returns a JSON-formatted string representation of the object.

        Returns
        -------
        str
            A JSON-formatted string containing the object's attributes.
        """

        attributes = {
            'training_batch_size': self.training_batch_size,
            'training_total_data': self.training_total_data,
            'training_total_epochs': self.training_total_epochs,
            'training_total_steps': self.training_total_steps,
            'training_time': self.training_time,
            'training_time_per_epoch': self.training_time_per_epoch,
            'training_time_per_step': self.training_time_per_step,
            'validation_batch_size': self.validation_batch_size,
            'validation_total_data': self.validation_total_data,
            'validation_total_epochs': self.validation_total_epochs,
            'validation_total_steps': self.validation_total_steps,
            'validation_time': self.validation_time,
            'validation_time_per_epoch': self.validation_time_per_epoch,
            'validation_time_per_step': self.validation_time_per_step,
            'test_batch_size': self.test_batch_size,
            'test_total_data': self.test_total_data,
            'test_total_epochs': self.test_total_epochs,
            'test_total_steps': self.test_total_steps,
            'test_time': self.test_time,
            'test_time_per_epoch': self.test_time_per_epoch,
            'test_time_per_step': self.test_time_per_step,
            'loss_epoch': self.loss_epoch,
            'loss_training': self.loss_training,
            'loss_validation': self.loss_validation,
            'loss_history': self.loss_history,
            'evaluation': self.evaluation,
            'samples': self.samples,
        }

        attributes = json.dumps(attributes, default=lambda x: str(x))

        return attributes

    def __str__(self):
        """
        Returns a string representation of the object with useful information.

        Returns
        -------
        str
            The string representation of the object.
        """

        loss_history = '\n'.join(self.loss_history)
        evaluation = '\n'.join(self.evaluation)

        info = f"""
            Training Information\n
            Training Batch Size                {self.training_batch_size}
            Training Total Data                {self.training_total_data}
            Training Total Epochs              {self.training_total_epochs}
            Training Total Steps               {self.training_total_steps}
            Training Time                      {self.training_time}
            Training Time per Epoch            {self.training_time_per_epoch}
            Training Time per Step             {self.training_time_per_step}

            Validation Batch Size              {self.validation_batch_size}
            Validation Total Data              {self.validation_total_data}
            Validation Total Epochs            {self.validation_total_epochs}
            Validation Total Steps             {self.validation_total_steps}
            Validation Time                    {self.validation_time}
            Validation Time per Epoch          {self.validation_time_per_epoch}
            Validation Time per Step           {self.validation_time_per_step}

            Test Information\n
            Test Batch Size                    {self.test_batch_size}
            Test Total Data                    {self.test_total_data}
            Test Total Epochs                  {self.test_total_epochs}
            Test Total Steps                   {self.test_total_steps}
            Test Time                          {self.test_time}
            Test Time per Epoch                {self.test_time_per_epoch}
            Test Time per Step                 {self.test_time_per_step}

            Loss\n
            Best Loss Epoch                    {self.loss_epoch}
            Best Loss Training                 {self.loss_training}
            Best Loss Validation               {self.loss_validation}

            Loss History\n\n                   {loss_history}

            Evaluation\n\n                     {evaluation}
        """

        info = '\n'.join([x.strip() for x in info.splitlines()])

        return info

    def set_training_info(self,
                          history,
                          training_data,
                          training_steps,
                          validation_data,
                          validation_steps,
                          total_time):
        """
        Set the training information.

        Parameters
        ----------
        history : dict
            The training history object.
        training_data : data generator
            The training data.
        training_steps : int
            The number of training steps per epoch.
        validation_data : data generator
            The validation data.
        validation_steps : int
            The number of validation steps per epoch.
        total_time : float
            The total training time.
        """

        loss = history['loss']
        val_loss = history['val_loss']

        best_val_loss = np.inf
        best_loss = None
        best_loss_epoch = -1
        results = []

        for epoch, (loss_value, val_loss_value) in enumerate(zip(loss, val_loss)):
            is_best = ''

            if val_loss_value < best_val_loss:
                is_best = '*'
                best_loss_epoch = epoch + 1
                best_loss = loss_value
                best_val_loss = val_loss_value

            results.append([epoch + 1, loss_value, val_loss_value, is_best])

        self.training_total_epochs = len(results)
        self.validation_total_epochs = len(results)

        self.loss_history = ['epoch,loss,val_loss,best']

        for result in results:
            epoch, loss_value, val_loss_value, is_best = result
            self.loss_history += [f"{epoch},{loss_value:.4f},{val_loss_value:.4f},{is_best}"]

        self.loss_epoch = best_loss_epoch
        self.loss_training = best_loss
        self.loss_validation = best_val_loss

        training_batch_size = len(next(training_data)[0])
        validation_batch_size = len(next(validation_data)[0])

        self.training_batch_size = training_batch_size
        self.training_total_data = training_batch_size * training_steps
        self.training_total_steps = training_steps
        self.training_time = str(datetime.timedelta(seconds=total_time))
        self.training_time_per_epoch = str(datetime.timedelta(seconds=total_time / self.training_total_epochs))
        self.training_time_per_step = str(datetime.timedelta(seconds=total_time / self.training_total_steps))

        self.validation_batch_size = validation_batch_size
        self.validation_total_data = validation_batch_size * validation_steps
        self.validation_total_steps = validation_steps
        self.validation_time = str(datetime.timedelta(seconds=total_time))
        self.validation_time_per_epoch = str(datetime.timedelta(seconds=total_time / self.validation_total_epochs))
        self.validation_time_per_step = str(datetime.timedelta(seconds=total_time / self.validation_total_steps))

    def set_test_info(self, test_data, test_steps, total_time):
        """
        Set the test information.

        Parameters
        ----------
        test_data : data generator
            The test data.
        test_steps : int
            The number of test steps.
        total_time : float
            The total test time.
        """

        test_batch_size = len(next(test_data)[0])

        self.test_batch_size = test_batch_size
        self.test_total_data = test_batch_size * test_steps
        self.test_total_epochs = 1
        self.test_total_steps = test_steps
        self.test_time = str(datetime.timedelta(seconds=total_time))
        self.test_time_per_epoch = str(datetime.timedelta(seconds=total_time / self.test_total_epochs))
        self.test_time_per_step = str(datetime.timedelta(seconds=total_time / self.test_total_steps))

    def set_evaluation_info(self, metrics, samples, origin='vanilla'):
        """
        Set the evaluation information.

        Parameters
        ----------
        metrics : data generator
            Error metrics from each top path.
        samples : int
            Samples retrieved from prediction.
        origin : str, optional
            Indicates the origin name. Default is vanilla.
        """

        self.evaluation[origin] = ['top_path,cer,wer,ler,ser']

        for i, x in enumerate(metrics, start=1):
            self.evaluation[origin] += [f"{i},{x[0]:.4f},{x[1]:.4f},{x[2]:.4f},{x[3]:.4f}"]

        self.samples[origin] = []

        for i in range(len(samples)):
            path = {'top': [], 'mid': [], 'bottom': []}

            for x in samples[i][0]:
                path['top'].append({'image_path': x[0], 'bbox': x[1], 'label': x[2], 'prediction': x[3]})

            for x in samples[i][1]:
                path['mid'].append({'image_path': x[0], 'bbox': x[1], 'label': x[2], 'prediction': x[3]})

            for x in samples[i][2]:
                path['bottom'].append({'image_path': x[0], 'bbox': x[1], 'label': x[2], 'prediction': x[3]})

            self.samples[origin].append({f"top_path_{i}": path})
