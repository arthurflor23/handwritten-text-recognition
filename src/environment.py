"""Set paths and environment variables"""

from os.path import join


class Path():

    def __init__(self, dataset, epochs=None, batch=None):
        self.source = join("..", "data", dataset)
        self.raw_source = join("..", "data", f"raw_{dataset}")
        self.output = join("..", "output", dataset)

        self.data = join(self.source, "lines")
        self.ground_truth = join(self.source, "ground_truth")
        self.partitions = join(self.source, "partitions")

        self.train_file = join(self.partitions, "train.txt")
        self.validation_file = join(self.partitions, "validation.txt")
        self.test_file = join(self.partitions, "test.txt")

        self.epochs = epochs
        self.batch = batch
