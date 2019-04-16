"""Data class to load the variables environment."""

import os


class Data():
    """Data class."""

    def __init__(self, input_dir):
        self.dataset = os.path.basename(input_dir)
        self.input_dir = input_dir
        self.preproc_dir = os.path.join(input_dir, "lines_preproc")

        self.train, self.train_gt = None, None
        self.validation, self.validation_gt = None, None
        self.test, self.test_gt = None, None

    def imread_partitions(self):
        """Load all partitions."""
    

    def imread_train(self):
        """Load the train and validation (if exists) partitions."""
        

    def imread_test(self):
        """Load the test partition."""

