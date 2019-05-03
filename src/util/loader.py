"""Uses generator functions to supply train/test with data.
Image renderings and text are created on the fly each time"""

from util.preproc import padding_list
import numpy as np
import random


class DataGenerator():
    """Generator class with data streaming"""

    def __init__(self, env, train=False):
        self.dictionary = " !\"#&'()*+,-./0123456789:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
        self.batch_size = np.maximum(2, env.batch_size)
        self.training = train

        self.train_npz = env.train
        self.valid_npz = env.valid
        self.test_npz = env.test

        if self.training:
            self.train_index, self.valid_index = 0, 0

            self.train, self.train_gt = self.read_npz(self.train_npz)
            self.train_steps = self.train.shape[0] // self.batch_size

            self.valid, self.valid_gt = self.read_npz(self.valid_npz)
            self.val_steps = self.valid.shape[0] // self.batch_size

        else:
            self.test_index = 0

            self.test, self.test_gt = self.read_npz(self.test_npz)
            self.test_steps = self.test.shape[0] // self.batch_size

    def read_npz(self, npz):
        """Read the partitions and random fill until batch divider"""

        arr = np.load(npz, allow_pickle=True, mmap_mode="r")
        np.random.shuffle(arr["dt"])
        np.random.shuffle(arr["gt"])

        return arr["dt"], arr["gt"]

    def encode_ctc(self, txt, charset):
        """Encode text batch to CTC input (sparse)"""

        return [[float(charset.find(x)) for x in vec] for vec in txt]

    def decode_ctc(self, arr, charset):
        """Decode CTC output batch to text"""

        return ["".join(charset[int(c)] for c in vec if c < len(charset)) for vec in arr]

    def next_train_batch(self):
        """Get the next batch from train partition (yield)"""

        while True:
            if self.train_index >= self.train.shape[0]:
                self.train_index = 0

            index = self.train_index
            until = self.train_index + self.batch_size
            self.train_index += self.batch_size

            x_train = self.train[index:until]
            y_train = self.train_gt[index:until]
            arange = np.arange(x_train.shape[0] - 1)

            while (x_train.shape[0] % self.batch_size) > 0:
                i = random.choice(arange)
                x_train = np.append(x_train, [x_train[i]], axis=0)
                y_train = np.append(y_train, [y_train[i]], axis=0)

            y_train = padding_list(self.encode_ctc(y_train, self.dictionary), value=len(self.dictionary))

            # x_train_len (image rotate height) must be higher y_train_len (max char in line)
            x_train_len = np.ones(self.batch_size) * x_train.shape[2]
            y_train_len = np.ones(self.batch_size) * len(y_train[0])

            inputs = {
                "input": x_train,
                "labels": y_train,
                "input_length": x_train_len,
                "label_length": y_train_len
            }
            output = {"CTCloss": np.zeros(x_train.shape[0])}

            yield (inputs, output)

    def next_valid_batch(self):
        """Get the next batch from validation partition (yield)"""

        while True:
            if self.valid_index >= self.valid.shape[0]:
                self.valid_index = 0

            index = self.valid_index
            until = self.valid_index + self.batch_size
            self.valid_index += self.batch_size

            x_valid = self.valid[index:until]
            y_valid = self.valid_gt[index:until]
            arange = np.arange(x_valid.shape[0] - 1)

            while (x_valid.shape[0] % self.batch_size) > 0:
                i = random.choice(arange)
                x_valid = np.append(x_valid, [x_valid[i]], axis=0)
                y_valid = np.append(y_valid, [y_valid[i]], axis=0)

            y_valid = padding_list(self.encode_ctc(y_valid, self.dictionary), value=len(self.dictionary))

            # x_valid_len (image rotate height) must be higher y_valid_len (max char in line)
            x_valid_len = np.ones(self.batch_size) * x_valid.shape[2]
            y_valid_len = np.ones(self.batch_size) * len(y_valid[0])

            inputs = {
                "input": x_valid,
                "labels": y_valid,
                "input_length": x_valid_len,
                "label_length": y_valid_len
            }
            output = {"CTCloss": np.zeros(x_valid.shape[0])}

            yield (inputs, output)

    def next_test_batch(self):
        """Return model evaluate parameters"""

        while True:
            if self.test_index >= self.test.shape[0]:
                self.test_index = 0

            index = self.test_index
            until = self.test_index + self.batch_size
            self.test_index += self.batch_size

            x_test = self.test[index:until]
            y_test = self.test_gt[index:until]
            arange = np.arange(x_test.shape[0] - 1)

            while (x_test.shape[0] % self.batch_size) > 0:
                i = random.choice(arange)
                x_test = np.append(x_test, [x_test[i]], axis=0)
                y_test = np.append(y_test, [y_test[i]], axis=0)

            y_test = padding_list(self.encode_ctc(y_test, self.dictionary), value=len(self.dictionary))

            # x_test_len (image rotate height) must be higher y_test_len (max char in line)
            x_test_len = np.ones(self.batch_size) * x_test.shape[2]
            y_test_len = np.ones(self.batch_size) * len(y_test[0])

            yield [x_test, y_test, x_test_len, y_test_len]
