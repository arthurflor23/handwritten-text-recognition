import os


class Data():

    def __init__(self, input_dir):
        self.dataset = os.path.basename(input_dir)
        self.input_dir = input_dir
        self.preproc_dir = os.path.join(input_dir, "lines_preproc")

        self.train, self.train_gt = None, None
        self.validation, self.validation_gt = None, None
        self.test, self.test_gt = None, None
