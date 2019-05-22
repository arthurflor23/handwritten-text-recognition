"""
Uses generator functions to supply train/test with data.
Image renderings and text are created on the fly each time.
"""

from data.preproc import normalization
import numpy as np
import h5py


class DataGenerator():
    """Generator class with data streaming"""

    def __init__(self, env):

        if env.lazy_loading:
            self.dataset = h5py.File(env.source, "r")
        else:
            with h5py.File(env.source, "r") as hf:
                self.dataset = {
                    "train": {
                        "dt": hf["train"]["dt"][:],
                        "gt": hf["train"]["gt"][:]
                    },
                    "valid": {
                        "dt": hf["valid"]["dt"][:],
                        "gt": hf["valid"]["gt"][:]
                    },
                    "test": {
                        "dt": hf["test"]["dt"][:],
                        "gt": hf["test"]["gt"][:]
                    }
                }

        self.max_text_length = env.max_text_length
        self.batch_size = max(2, env.batch_size)
        self.train_index, self.valid_index, self.test_index = 0, 0, 0

        self.total_train = self.dataset["train"]["gt"][:].shape[0]
        self.total_valid = self.dataset["valid"]["gt"][:].shape[0]
        self.total_test = self.dataset["test"]["gt"][:].shape[0]

        self.train_steps = self.total_train // self.batch_size
        self.valid_steps = self.total_valid // self.batch_size
        self.test_steps = self.total_test // self.batch_size

    def fill_batch(self, partition, total, x, y):
        """Fill batch array (x, y) if necessary (batch_size)"""

        if len(x) < self.batch_size:
            fill = self.batch_size - len(x)
            i = np.random.choice(np.arange(0, total - fill), 1)[0]
            x = np.append(x, self.dataset[partition]["dt"][i:i + fill], axis=0)
            y = np.append(y, self.dataset[partition]["gt"][i:i + fill], axis=0)
        return x, y

    def next_train_batch(self):
        """Get the next batch from train partition (yield)"""

        while True:
            if self.train_index >= self.total_train:
                self.train_index = 0

            index = self.train_index
            until = self.train_index + self.batch_size
            self.train_index += self.batch_size

            x_train = self.dataset["train"]["dt"][index:until]
            y_train = self.dataset["train"]["gt"][index:until]

            x_train, y_train = self.fill_batch("train", self.total_train, x_train, y_train)
            x_train = normalization(x_train,
                                    rotation_range=1.5,
                                    width_shift_range=0.02,
                                    height_shift_range=0.02,
                                    zoom_range=0.02)

            x_train_len = np.asarray([self.max_text_length for i in range(self.batch_size)])
            y_train_len = np.asarray([len(np.trim_zeros(y_train[i])) for i in range(self.batch_size)])

            inputs = {
                "input": x_train,
                "labels": y_train,
                "input_length": x_train_len,
                "label_length": y_train_len
            }
            output = {"CTCloss": np.zeros(self.batch_size)}

            yield (inputs, output)

    def next_valid_batch(self):
        """Get the next batch from validation partition (yield)"""

        while True:
            if self.valid_index >= self.total_valid:
                self.valid_index = 0

            index = self.valid_index
            until = self.valid_index + self.batch_size
            self.valid_index += self.batch_size

            x_valid = self.dataset["valid"]["dt"][index:until]
            y_valid = self.dataset["valid"]["gt"][index:until]

            x_valid, y_valid = self.fill_batch("valid", self.total_valid, x_valid, y_valid)
            x_valid = normalization(x_valid)

            x_valid_len = np.asarray([self.max_text_length for i in range(self.batch_size)])
            y_valid_len = np.asarray([len(np.trim_zeros(y_valid[i])) for i in range(self.batch_size)])

            inputs = {
                "input": x_valid,
                "labels": y_valid,
                "input_length": x_valid_len,
                "label_length": y_valid_len
            }
            output = {"CTCloss": np.zeros(self.batch_size)}

            yield (inputs, output)

    def next_test_batch(self):
        """Return model evaluate parameters"""

        while True:
            if self.test_index >= self.total_test:
                self.test_index = 0

            index = self.test_index
            until = self.test_index + self.batch_size
            self.test_index += self.batch_size

            x_test = self.dataset["test"]["dt"][index:until]
            y_test = self.dataset["test"]["gt"][index:until]

            x_test, y_test = self.fill_batch("test", self.total_test, x_test, y_test)
            x_test = normalization(x_test)

            x_test_len = np.asarray([self.max_text_length for i in range(self.batch_size)])
            y_test_len = np.asarray([len(np.trim_zeros(y_test[i])) for i in range(self.batch_size)])

            yield [x_test, y_test, x_test_len, y_test_len]
