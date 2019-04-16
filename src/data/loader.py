"""Load the environment variables."""

import os
import json


class Loader():
    """Loader class."""

    def __init__(self, input_dir):
        dirname = os.path.dirname(__file__)
        config = os.path.join(dirname, "..", "config.json")

        with open(config, "r") as file:
            self.env = json.load(file)

        self.dataset = os.path.basename(input_dir)
        self.input_dir = input_dir
        self.preproc_dir = os.path.join(input_dir, self.env["PREPROC_DIR"])

        self.train, self.train_gt = None, None
        self.validation, self.validation_gt = None, None
        self.test, self.test_gt = None, None

    def imread_partitions(self):
        """Load all partitions."""

        self.imread_train()
        # self.imread_test()

    def imread_train(self):
        """Load the train and validation (if exists) partitions."""
        print("train")

    def imread_test(self):
        """Load the test partition."""
