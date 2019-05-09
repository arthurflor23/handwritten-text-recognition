"""Uses generator functions to supply train/test with data.
Image renderings and text are created on the fly each time"""

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from util.preproc import padding_list
import numpy as np


class DataGenerator():
    """Generator class with data streaming"""

    def __init__(self, env, train=False):
        self.max_text_length = env.max_text_length
        self.batch_size = max(2, env.batch_size)
        self.charset = "".join([chr(i) for i in range(32, 127)])

        self.train_npz = env.train
        self.valid_npz = env.valid
        self.test_npz = env.test

        self.training = train
        self.generator = ImageDataGenerator(
            fill_mode="constant",
            rotation_range=0.2,
            width_shift_range=1e-2,
            height_shift_range=1e-2,
            shear_range=1e-3,
            zoom_range=1e-3
        )

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
        nb = arr["dt"].shape[0]

        arange = np.arange(nb - 1)
        rp = np.ones(nb, dtype=np.uint8)

        while (np.sum(rp) % self.batch_size) > 0:
            rp[np.random.choice(arange, 1)[0]] += 1

        return np.repeat(arr["dt"], rp, axis=0), np.repeat(arr["gt"], rp, axis=0)

    def encode_ctc(self, txt, charset):
        """Encode text batch to CTC input (sparse)"""

        return [[np.abs(float(charset.find(x))) for x in vec.strip()] for vec in txt]

    def decode_ctc(self, arr, charset):
        """Decode CTC output batch to text"""

        return [("".join(charset[int(c)] for c in vec)).strip() for vec in arr]

    def next_train_batch(self):
        """Get the next batch from train partition (yield)"""

        while True:
            if self.train_index >= self.train.shape[0]:
                self.train_index = 0

            index = self.train_index
            until = self.train_index + self.batch_size
            self.train_index += self.batch_size

            x_train = self.train[index:until]
            y_train = self.encode_ctc(self.train_gt[index:until], self.charset)

            x_train_len = np.asarray([self.max_text_length for i in range(self.batch_size)])
            y_train_len = np.asarray([len(y_train[i]) for i in range(self.batch_size)])

            x_train = self.generator.flow(padding_list(x_train), batch_size=self.batch_size, shuffle=False)[0]
            y_train = padding_list(y_train)

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
            if self.valid_index >= self.valid.shape[0]:
                self.valid_index = 0

            index = self.valid_index
            until = self.valid_index + self.batch_size
            self.valid_index += self.batch_size

            x_valid = self.valid[index:until]
            y_valid = self.encode_ctc(self.valid_gt[index:until], self.charset)

            x_valid_len = np.asarray([self.max_text_length for i in range(self.batch_size)])
            y_valid_len = np.asarray([len(y_valid[i]) for i in range(self.batch_size)])

            inputs = {
                "input": padding_list(x_valid),
                "labels": padding_list(y_valid),
                "input_length": x_valid_len,
                "label_length": y_valid_len
            }
            output = {"CTCloss": np.zeros(self.batch_size)}

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
            y_test = self.encode_ctc(self.test_gt[index:until], self.charset)

            x_test_len = np.asarray([self.max_text_length for i in range(self.batch_size)])
            y_test_len = np.asarray([len(y_test[i]) for i in range(self.batch_size)])

            yield [padding_list(x_test), padding_list(y_test), x_test_len, y_test_len]
