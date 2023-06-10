import os
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
        self._network = self._network()

    def compile(self, learning_rate=None, model_uri=None):

        self.model = self._network.compile_model(output_size=self.tokenizer.shape,
                                                 learning_rate=learning_rate or 1e-4,
                                                 ctc_loss_func=self.ctc_loss_func)

        if model_uri:
            self.model.load_weights(model_uri)

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

        if len(y_true.shape) > 2:
            y_true = tf.squeeze(y_true)

        # y_pred.shape = (batch_size, string_length, alphabet_size_1_hot_encoded)
        # output of every model is softmax
        # so sum across alphabet_size_1_hot_encoded give 1
        #               string_length give string length
        input_length = tf.math.reduce_sum(y_pred, axis=-1, keepdims=False)
        input_length = tf.math.reduce_sum(input_length, axis=-1, keepdims=True)

        # y_true strings are padded with 0
        # so sum of non-zero gives number of characters in this string
        label_length = tf.math.count_nonzero(y_true, axis=-1, keepdims=True, dtype='int64')

        loss = tf.keras.backend.ctc_batch_cost(y_true, y_pred, input_length, label_length)

        # average loss across all entries in the batch
        loss = tf.reduce_mean(loss)

        return loss
