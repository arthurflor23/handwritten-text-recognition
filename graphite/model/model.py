import os
import importlib
import tensorflow as tf


class OpticalModel():
    """
    General optical model management.
    """

    def __init__(self,
                 network=None,
                 data_path='mlruns',
                 model_uri=None,
                 seed=None):

        tf.random.set_seed(seed)

        self.network = network
        self.base_path = os.path.join(os.path.dirname(__file__), '..', '..')
        self.data_path = os.path.join(self.base_path, data_path)
        self.model_uri = model_uri
        self.seed = seed

        # if not self.infer:
        #     self.network_class = self._get_network_class()
        #     self.model = self.network_class.get_model()

    def _get_network_class(self):

        module_name = importlib.util.resolve_name(f".networks.{self.network}", __package__)
        module_spec = importlib.util.find_spec(module_name)
        assert module_spec is not None, "network file must be created"

        module = importlib.import_module(module_name, __package__)

        class_name = 'Network'
        assert hasattr(module, class_name), f"`{class_name}` class must be created"

        network_class = getattr(module, class_name)()

        return network_class
