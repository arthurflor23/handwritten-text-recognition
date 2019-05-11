"""Uses generator functions to supply train/test with data.
Image renderings and text are created on the fly each time"""

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from data.preproc import padding_list
import numpy as np


class DataGenerator():
    """Generator class with data streaming"""

    def __init__(self, env):
        self.max_text_length = env.max_text_length
        self.batch_size = max(2, env.batch_size)

        self.generator = ImageDataGenerator(
            fill_mode="constant",
            rotation_range=0.2,
            width_shift_range=0.02,
            height_shift_range=0.02,
            shear_range=1e-2,
            zoom_range=1e-2
        )

        self.train_index, self.valid_index, self.test_index = 0, 0, 0
        arr = np.load(env.source, allow_pickle=True, mmap_mode="r")

        self.train, self.train_gt = self.fill_by_batch(arr["train_dt"], arr["train_gt"])
        self.train_steps = len(self.train) // self.batch_size

        self.valid, self.valid_gt = self.fill_by_batch(arr["valid_dt"], arr["valid_gt"])
        self.val_steps = len(self.valid) // self.batch_size

        self.test, self.test_gt = self.fill_by_batch(arr["test_dt"], arr["test_gt"])
        self.test_steps = len(self.test) // self.batch_size
        del arr

    def fill_by_batch(self, dt_arr, gt_arr):
        """Random fill until batch divider"""

        nb = len(dt_arr)
        arange = np.arange(nb - 1)
        rp = np.ones(nb, dtype=np.uint8)

        while (np.sum(rp) % self.batch_size) > 0:
            rp[np.random.choice(arange, 1)[0]] += 1

        return np.repeat(dt_arr, rp, axis=0), np.repeat(gt_arr, rp, axis=0)

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
            y_valid = self.valid_gt[index:until]

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
            y_test = self.test_gt[index:until]

            x_test_len = np.asarray([self.max_text_length for i in range(self.batch_size)])
            y_test_len = np.asarray([len(y_test[i]) for i in range(self.batch_size)])

            yield [padding_list(x_test), padding_list(y_test), x_test_len, y_test_len]
