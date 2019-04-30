"""Uses generator functions to supply train/test with data.
Image renderings and text are created on the fly each time"""

import os
import random
import numpy as np
from . import preproc


class DataGenerator():
    """Generator class with data streaming"""

    def __init__(self, args, train=False, test=False):
        self.dictionary = " !\"#&'()*+,-./0123456789:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
        self.batch_size = np.maximum(args.batch, 1)
        self.padding_value = 255
        self.nb_features = 64

        self.data_path = args.data
        self.ground_truth_path = args.ground_truth

        if train:
            self.train_list = self.read_and_fill(args.train_file)
            self.train_steps = len(self.train_list) // self.batch_size

            self.val_list = self.read_and_fill(args.validation_file)
            self.val_steps = len(self.val_list) // self.batch_size

            self.training = True
            self.build_train()

        if test:
            self.test_list = self.read_and_fill(args.test_file)
            self.test_steps = len(self.test_list) // self.batch_size

            self.training = False
            self.build_test()

    def read_and_fill(self, partition_file):
        """Read the partitions and random fill until batch divider"""

        arr = open(partition_file).read().splitlines()
        np.random.shuffle(arr)

        while (len(arr) % self.batch_size) > 0:
            arr.append(random.choice(arr))
        return np.array(arr)

    def build_train(self):
        """Read and build the train and validation files of the dataset"""

        self.train_index, self.val_index = 0, 0
        # self.x_train, self.x_train_len = self.fetch_img_by_partition(self.train_list)
        self.y_train, self.y_train_len = self.fetch_txt_by_partition(self.train_list)

        # self.x_val, self.x_val_len = self.fetch_img_by_partition(self.val_list)
        self.y_val, self.y_val_len = self.fetch_txt_by_partition(self.val_list)

    def build_test(self):
        """Read and build the test files of the dataset"""

        self.test_index = 0
        self.x_test, self.x_test_len = self.fetch_img_by_partition(self.test_list)
        self.y_test, self.y_test_len = self.fetch_txt_by_partition(self.test_list)

    def fetch_img_by_partition(self, partition_list):
        """Load image and apply preprocess"""

        inputs = []
        for _, filename in enumerate(partition_list):
            img_path = os.path.join(self.data_path, f"{filename}.png")
            inputs.append(preproc.process_image(img_path, self.nb_features))

        inputs_len = [len(inputs[i]) for i in range(len(inputs))]
        inputs_pad = preproc.padding_list(inputs, value=self.padding_value)

        return np.array(inputs_pad), np.array(inputs_len)

    def fetch_txt_by_partition(self, partition_list):
        """Load text label and apply encode"""

        inputs = []
        for i, filename in enumerate(partition_list):
            txt = open(os.path.join(self.ground_truth_path, f"{filename}.txt")).read().strip()
            txt_encode = [float(self.dictionary.find(c)) for c in txt]
            inputs.append(txt_encode)

        inputs_len = [len(inputs[i]) for i in range(len(inputs))]
        inputs_pad = preproc.padding_list(inputs, value=len(self.dictionary))

        return np.array(inputs_pad), np.array(inputs_len)

    def next_train(self):
        """Get the next batch from train partition (yield)"""

        while True:
            if self.train_index >= len(self.train_list):
                self.train_index = 0

            index = self.train_index
            until = self.train_index + self.batch_size

            x_train, x_train_len = self.fetch_img_by_partition(self.train_list[index:until])
            self.train_index += self.batch_size

            inputs = {
                "input": x_train,
                "labels": self.y_train[index:until],
                "input_length": x_train_len,
                "label_length": self.y_train_len[index:until]
            }
            output = {"CTCloss": np.zeros(len(x_train))}

            yield (inputs, output)

    def next_val(self):
        """Get the next batch from validation partition (yield)"""

        while True:
            if self.val_index >= len(self.val_list):
                self.val_index = 0

            index = self.val_index
            until = self.val_index + self.batch_size

            x_val, x_val_len = self.fetch_img_by_partition(self.val_list[index:until])
            self.val_index += self.batch_size

            inputs = {
                "input": x_val,
                "labels": self.y_val[index:until],
                "input_length": x_val_len,
                "label_length": self.y_val_len[index:until]
            }
            output = {"CTCloss": np.zeros(len(x_val))}

            yield (inputs, output)

    def next_eval(self):
        return [self.x_test, self.y_test, self.x_test_len, self.y_test_len]

    def next_pred(self):
        return [self.x_test, self.x_test_len]
