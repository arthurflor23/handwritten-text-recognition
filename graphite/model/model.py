import os
import importlib
import tensorflow as tf


class OpticalModel():
    """
    General optical model management.
    """

    def __init__(self,
                 network,
                 artifact_path='mlruns',
                 seed=None):

        tf.random.set_seed(seed)

        self.network = network
        self.base_path = os.path.join(os.path.dirname(__file__), '..', '..')
        self.artifact_path = os.path.join(self.base_path, artifact_path)
        self.seed = seed

        self._network = self._import_network(self.network)
        self._network = self._network()

    def _import_network(self, network):

        module_name = importlib.util.resolve_name(f".network.{network}", __package__)
        module_spec = importlib.util.find_spec(module_name)
        assert module_spec is not None, "network file must be created"

        module = importlib.import_module(module_name, __package__)

        class_name = 'Network'
        assert hasattr(module, class_name), f"`{class_name}` class must be created"

        network = getattr(module, class_name)

        return network
