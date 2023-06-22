import os
import time
import json
import glob
import datetime
import importlib
import numpy as np
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
        tf.config.set_visible_devices([], 'GPU')

        self.network = network
        self.tokenizer = tokenizer
        self.pad_value = pad_value
        self.artifact_path = artifact_path
        self.seed = seed

        self.evaluate = Evaluate()
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

        attributes = json.dumps({
            'network': self.network,
            'tokenizer_shape': self.tokenizer.shape,
            'pad_value': self.pad_value,
            'seed': self.seed,
        }, default=lambda x: str(x))

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
            Network                 {self.network}
            Tokenizer Shape         {self.tokenizer.shape}
            Padding Value           {self.pad_value}
            Seed                    {self.seed}
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

        self.model.summary()
        self.logger.set_compile_info(self.model)

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
                beam_width=100,
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
            The width of the beam for the CTC decoder, by default 100.
        ctc_decode : bool, optional
            If True, applies CTC decoding to the predictions, by default True.
        token_decode : bool, optional
            If True, decodes the tokens to their corresponding characters, by default True.
        verbose : int, optional
            Verbosity mode, 0 or 1. By default 1.

        Returns
        -------
        tuple of np.ndarray
            The first array is the decoded predictions, and
            the second array is the probabilities of these predictions.
        """

        start_time = time.time()

        predicts = self.model.predict(x=test_data, steps=test_steps, verbose=verbose)
        predicts = np.log(predicts + 1e-7)

        decoded, probabilities = predicts, np.array([])

        if ctc_decode:
            progbar = tf.keras.utils.Progbar(target=predicts.shape[1], unit_name='path_decode', verbose=verbose)

            decoded_paths, probabilities_list = [], []
            sequence_length = [predicts.shape[2]] * predicts.shape[0]

            beam_width = max(top_paths, beam_width)

            for i in range(predicts.shape[1]):
                progbar.update(i)

                inputs = tf.transpose(predicts[:, i, :, :], perm=[1, 0, 2])
                decoded, log_probabilities = tf.nn.ctc_beam_search_decoder(inputs=inputs,
                                                                           sequence_length=sequence_length,
                                                                           beam_width=beam_width,
                                                                           top_paths=top_paths)

                decoded_pads = []
                for j in range(top_paths):
                    sparse_decoded = tf.sparse.to_dense(decoded[j], default_value=-1)
                    paddings = [[0, 0], [0, predicts.shape[2] - tf.reduce_max(tf.shape(sparse_decoded)[1])]]
                    decoded_pads.append(tf.pad(sparse_decoded, paddings=paddings, constant_values=-1))

                decoded_paths.append(decoded_pads)
                probabilities_list.append(tf.exp(log_probabilities))

                progbar.update(i + 1)

            decoded = np.transpose(tf.stack(decoded_paths, axis=1), (0, 2, 1, 3))
            probabilities = np.transpose(tf.stack(probabilities_list, axis=1), (2, 0, 1))

            if token_decode:
                decoded = [[self.tokenizer.decode(decoded[i, j, :, :])
                            for j in range(decoded.shape[1])] for i in range(decoded.shape[0])]

                decoded = np.array(decoded, dtype=object)

        end_time = time.time()
        total_time = end_time - start_time

        self.logger.set_test_info(test_data, test_steps, total_time)

        return decoded, probabilities

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


class Evaluate():

    def __init__(self):
        print('Evaluate class here...')


class Logger():
    """
    Class to log and store training, and test information.
    """

    def __init__(self):
        """
        Initialize the Logger object.
        """

        self.summary = ''

        self.optimizer = None
        self.learning_rate = None

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
        self.loss_history = ''

    def __repr__(self):
        """
        Returns a JSON-formatted string representation of the object.

        Returns
        -------
        str
            A JSON-formatted string containing the object's attributes.
        """

        attributes = json.dumps({
            'summary': self.summary,
            'optimizer': self.optimizer,
            'learning_rate': self.learning_rate,
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
            'loss_history': [','.join([str(y) for y in x]) for x in self.loss_history],
        }, default=lambda x: str(x))

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
            Summary\n\n                        {self.summary}

            Optimizer                          {self.optimizer}
            Learning Rate                      {self.learning_rate}

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

            Loss History\n\n                   {self.loss_history}
        """

        info = '\n'.join([x.strip() for x in info.splitlines()])

        return info

    def set_compile_info(self, model):
        """
        Set the compilation information of the model.

        Parameters
        ----------
        model : tf.keras.Model
            The compiled model.
        """

        self.optimizer = model.optimizer.get_config()['name']
        self.learning_rate = model.optimizer.get_config()['learning_rate']

        self.summary = []
        model.summary(print_fn=lambda x: self.summary.append(x))
        self.summary = '\n'.join(self.summary)

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

        self.loss_history = 'epoch,loss,val_loss,best'

        for result in results:
            epoch, loss_value, val_loss_value, is_best = result
            self.loss_history += f"\n{epoch},{loss_value},{val_loss_value},{is_best}"

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
