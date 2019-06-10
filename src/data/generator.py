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
        self.batch_size = env.batch_size
        self.max_text_length = env.max_text_length

        with h5py.File(env.source, "r") as hf:
            self.dataset = dict()

            for partition in hf.keys():
                self.dataset[partition] = dict()

                for data_type in hf[partition]:
                    self.dataset[partition][data_type] = hf[partition][data_type][:]

        self.train_index, self.valid_index, self.test_index = 0, 0, 0

        self.total_train = len(self.dataset["train"]["dt"])
        self.total_valid = len(self.dataset["valid"]["dt"])
        self.total_test = len(self.dataset["test"]["dt"])

        self.train_steps = self.total_train // self.batch_size
        self.valid_steps = self.total_valid // self.batch_size
        self.test_steps = self.total_test // self.batch_size

    def fill_batch(self, partition, maximum, x, y, w):
        """Fill batch array (x, y) if required (batch_size)"""

        if len(x) < self.batch_size:
            fill = self.batch_size - len(x)
            i = np.random.choice(np.arange(0, maximum - fill), 1)[0]

            if x is not None:
                x = np.append(x, self.dataset[partition]["dt"][i:i + fill], axis=0)
            if y is not None:
                y = np.append(y, self.dataset[partition]["gt_sparse"][i:i + fill], axis=0)
            if w is not None:
                w = np.append(w, self.dataset[partition]["gt_bytes"][i:i + fill], axis=0)

        return (x, y, w)

    def next_train_batch(self):
        """Get the next batch from train partition (yield)"""

        while True:
            if self.train_index >= self.total_train:
                self.train_index = 0

            index = self.train_index
            until = self.train_index + self.batch_size
            self.train_index += self.batch_size

            x_train = self.dataset["train"]["dt"][index:until]
            y_train = self.dataset["train"]["gt_sparse"][index:until]

            x_train, y_train, _ = self.fill_batch("train", self.total_train, x_train, y_train, w=None)
            x_train = normalization(x_train, rotation_range=0.5, zoom_range=0.01,
                                    height_shift_range=0.01, width_shift_range=0.01)

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
            y_valid = self.dataset["valid"]["gt_sparse"][index:until]

            x_valid, y_valid, _ = self.fill_batch("valid", self.total_valid, x_valid, y_valid, w=None)
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
        """Return model evaluate/predict parameters"""

        while True:
            if self.test_index >= self.total_test:
                self.test_index = 0

            index = self.test_index
            until = self.test_index + self.batch_size
            self.test_index += self.batch_size

            x_test = self.dataset["test"]["dt"][index:until]
            y_test = self.dataset["test"]["gt_sparse"][index:until]
            w_test = self.dataset["test"]["gt_bytes"][index:until]

            x_test, y_test, w_test = self.fill_batch("test", self.total_test, x_test, y_test, w_test)
            x_test = normalization(x_test)

            x_test_len = np.asarray([self.max_text_length for i in range(self.batch_size)])
            y_test_len = np.asarray([len(np.trim_zeros(y_test[i])) for i in range(self.batch_size)])

            yield [x_test, y_test, x_test_len, y_test_len, w_test]
