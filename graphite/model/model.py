import os
import time
import importlib
import tensorflow as tf


class Model():
    """
    General optical model management.
    """

    def __init__(self,
                 network,
                 tokenizer,
                 artifact_path='mlruns',
                 seed=None):

        tf.random.set_seed(seed)

        self.network = network
        self.tokenizer = tokenizer
        self.base_path = os.path.join(os.path.dirname(__file__), '..', '..')
        self.artifact_path = os.path.join(self.base_path, artifact_path)
        self.seed = seed

        self._network = self._import_network(self.network)
        self._network = self._network(self.tokenizer.shape)

    def compile(self, learning_rate=None, model_uri=None):

        self.model = self._network.compile_model(learning_rate=learning_rate,
                                                 loss_func=self.ctc_loss_func)

        if model_uri is None:
            timestamp = str(int(time.time()))
            model_uri = os.path.join(self.artifact_path, self.network, timestamp, 'model.hdf5')

        self.model_uri = model_uri

        if self.model_uri and os.path.exists(self.model_uri) and os.path.isfile(self.model_uri):
            self.model.load_weights(self.model_uri)

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

        logpath = os.path.dirname(self.model_uri)
        logfile = os.path.join(logpath, 'epochs.log')

        callbacks = [
            tf.keras.callbacks.CSVLogger(
                filename=logfile,
                separator=',',
                append=True,
            ),
            tf.keras.callbacks.TensorBoard(
                log_dir=logpath,
                write_graph=True,
                write_images=True,
                profile_batch=10,
                histogram_freq=10,
                embeddings_freq=10,
                update_freq='epoch',
                write_steps_per_second=True,
            ),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=self.model_uri,
                mode='auto',
                monitor='val_loss',
                save_best_only=True,
                save_weights_only=False,
                save_freq='epoch',
                verbose=verbose,
            ),
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

        output = self.model.fit(
            x=training_data,
            steps_per_epoch=training_steps,
            validation_data=validation_data,
            validation_steps=validation_steps,
            # callbacks=callbacks,
            epochs=epochs,
            verbose=verbose,
        )

        return output

    def _import_network(self, network):

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

        # Reshape inputs to (batch_size, sequence_length, last_dim_size)
        y_true = tf.reshape(y_true, (tf.shape(y_true)[0], -1, tf.shape(y_true)[-1]))
        y_pred = tf.reshape(y_pred, (tf.shape(y_pred)[0], -1, tf.shape(y_pred)[-1]))

        # Remove extra dimensions in y_true if present
        if len(y_true.shape) > 2:
            y_true = tf.squeeze(y_true)

        # Compute lengths for CTC calculation
        label_length = tf.math.count_nonzero(y_true, axis=-1, keepdims=True, dtype='int64')
        input_length = tf.reduce_sum(tf.reduce_sum(y_pred, axis=-1), axis=-1, keepdims=True)

        # Compute CTC loss and average it across the batch
        loss = tf.keras.backend.ctc_batch_cost(y_true, y_pred, input_length, label_length)
        loss = tf.reduce_mean(loss)

        return loss
