"""Load the environment variables"""

import os

CHAR_LIST = " !\"#&'()*+,-./0123456789:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"


class Path():
    """Path class"""

    def __init__(self, origin, target="./output"):
        self.output = os.path.join(target, origin)
        self.output_log = os.path.join(self.output, "log")

        self.ground_truth = os.path.join(origin, "ground_truth")
        self.data = os.path.join(origin, "lines")
        self.preproc = os.path.join(origin, "lines_preproc")
        self.partitions = os.path.join(origin, "partitions")

        self.train_file = os.path.join(self.partitions, "train.txt")
        self.validation_file = os.path.join(self.partitions, "validation.txt")
        self.test_file = os.path.join(self.partitions, "test.txt")
