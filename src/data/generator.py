"""
Uses generator functions to supply train/test with data.
Image renderings and text are created on the fly each time.
"""

import data.preproc as pp
import numpy as np
import h5py


class DataGenerator():
    """Generator class with data streaming"""

    def __init__(self, hdf5_src, batch_size, max_text_length):
        self.batch_size = batch_size
        self.max_text_length = max_text_length

        with h5py.File(hdf5_src, "r") as hf:
            self.dataset = dict()

            for partition in hf.keys():
                self.dataset[partition] = dict()

                for data_type in hf[partition]:
                    self.dataset[partition][data_type] = hf[partition][data_type][:]

        self.full_fill_partition("train")
        self.full_fill_partition("valid")
        self.full_fill_partition("test")

        self.total_train = len(self.dataset["train"]["gt_bytes"])
        self.total_valid = len(self.dataset["valid"]["gt_bytes"])
        self.total_test = len(self.dataset["test"]["gt_bytes"])

        self.train_steps = np.maximum(self.total_train // self.batch_size, 1)
        self.valid_steps = np.maximum(self.total_valid // self.batch_size, 1)
        self.test_steps = np.maximum(self.total_test // self.batch_size, 1)

        self.train_index, self.valid_index, self.test_index = 0, 0, 0

    def full_fill_partition(self, pt):
        """Make full fill partition up to batch size and steps"""

        while len(self.dataset[pt]["dt"]) % self.batch_size:
            i = np.random.choice(np.arange(0, len(self.dataset[pt]["dt"])), 1)[0]

            for sub in ["dt", "gt_sparse", "gt_bytes"]:
                self.dataset[pt][sub] = np.append(self.dataset[pt][sub], [self.dataset[pt][sub][i]], axis=0)

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

            x_train = pp.augmentation(x_train,
                                      rotation_range=1.5,
                                      scale_range=0.05,
                                      height_shift_range=0.025,
                                      width_shift_range=0.05,
                                      erode_range=5,
                                      dilate_range=3)
            x_train = pp.normalization(x_train)

            x_train_len = np.asarray([self.max_text_length for _ in range(self.batch_size)])
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

            x_valid = pp.normalization(x_valid)

            x_valid_len = np.asarray([self.max_text_length for _ in range(self.batch_size)])
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
        """Return model predict parameters"""

        while True:
            if self.test_index >= self.total_test:
                self.test_index = 0

            index = self.test_index
            until = self.test_index + self.batch_size
            self.test_index += self.batch_size

            x_test = self.dataset["test"]["dt"][index:until]
            y_test = self.dataset["test"]["gt_sparse"][index:until]
            w_test = self.dataset["test"]["gt_bytes"][index:until]

            x_test = pp.normalization(x_test)

            x_test_len = np.asarray([self.max_text_length for _ in range(self.batch_size)])
            y_test_len = np.asarray([len(np.trim_zeros(y_test[i])) for i in range(self.batch_size)])

            yield [x_test, y_test, x_test_len, y_test_len, w_test]
