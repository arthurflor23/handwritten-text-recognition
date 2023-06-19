import os
import json
import glob
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

        self._network = self._import_network(self.network)
        self._network = self._network(self.tokenizer.shape, self.pad_value)

    def __repr__(self):
        """
        Returns a JSON-formatted string representation of the Model object.

        Returns
        -------
        str
            A JSON-formatted string containing the object's attributes.
        """

        return json.dumps({
            'network': self.network,
            'tokenizer_shape': self.tokenizer.shape,
            'pad_value': self.pad_value,
            'seed': self.seed,
        })

    def __str__(self):
        """
        Returns a string representation of the Model object with useful information.

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

    def fit(self,
            training_data,
            training_steps=None,
            validation_data=None,
            validation_steps=None,
            plateau_cooldown=0,
            plateau_factor=0.2,
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
        plateau_cooldown : int, optional
            The number of epochs to wait before resuming normal operation after lr has been reduced, by default 0.
        plateau_factor : float, optional
            Factor by which the learning rate will be reduced, by default 0.2.
        plateau_patience : int, optional
            The number of epochs with no improvement after which learning rate will be reduced, by default 10.
        patience : int, optional
            The number of epochs with no improvement after which training will be stopped, by default 20.
        epochs : int, optional
            The number of epochs to train the model, by default 1000.
        verbose : int, optional
            Verbosity mode, by default 1.
        """

        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                mode='min',
                monitor='val_loss',
                min_delta=1e-8,
                patience=patience,
                start_from_epoch=0,
                restore_best_weights=True,
                verbose=verbose,
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                mode='min',
                monitor='val_loss',
                min_lr=1e-4,
                min_delta=1e-8,
                factor=plateau_factor,
                patience=plateau_patience,
                cooldown=plateau_cooldown,
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

        predicts = self.model.predict(x=test_data, steps=test_steps, verbose=verbose)
        decoded, probabilities = np.log(predicts.clip(min=1e-8)), np.array([])

        if ctc_decode:
            sequence_length = [predicts.shape[2]] * predicts.shape[0]
            decoded_paths, probabilities_list = [], []

            progbar = tf.keras.utils.Progbar(target=predicts.shape[1], unit_name='path_decode', verbose=verbose)

            for i in range(predicts.shape[1]):
                progbar.update(i)
                inputs = tf.transpose(predicts[:, i, :, :], perm=[1, 0, 2])

                decoded, log_probabilities = tf.nn.ctc_beam_search_decoder(inputs=inputs,
                                                                           sequence_length=sequence_length,
                                                                           beam_width=beam_width,
                                                                           top_paths=top_paths)

                probabilities_list.append(tf.exp(log_probabilities))

                decoded_pads = []
                for j in range(top_paths):
                    sparse_decoded = tf.sparse.to_dense(decoded[j], default_value=-1)
                    padding = [[0, 0], [0, predicts.shape[2] - tf.reduce_max(tf.shape(sparse_decoded)[1])]]
                    decoded_pads.append(tf.pad(sparse_decoded, paddings=padding, constant_values=-1))

                decoded_paths.append(decoded_pads)
                progbar.update(i + 1)

            decoded = np.transpose(tf.stack(decoded_paths, axis=1), (0, 2, 1, 3))
            probabilities = np.transpose(tf.stack(probabilities_list, axis=1), (2, 0, 1))

            if token_decode:
                decoded_strings = []

                for i in range(decoded.shape[0]):
                    instance_strings = []

                    for j in range(decoded.shape[1]):
                        decoded_string = self.tokenizer.decode(decoded[i, j, :, :])
                        instance_strings.append(decoded_string)

                    decoded_strings.append(instance_strings)

                decoded = np.array(decoded_strings, dtype=object)

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
