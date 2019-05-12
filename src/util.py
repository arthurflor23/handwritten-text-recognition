"""Set paths and environment variables"""

import os


class Environment():

    def __init__(self, args):
        self.data = os.path.join("..", "data")
        self.source = os.path.join(self.data, f"{args.dataset}.hdf5")
        self.raw_source = os.path.join("..", "raw", args.dataset)

        self.output = os.path.join("..", "output", args.dataset)
        self.output_tasks = os.path.join(self.output, "tasks")

        self.epochs = args.epochs
        self.batch_size = args.batch_size
        self.worker_mode = args.worker_mode
        self.full_mode = args.full_mode

        self.charset = "".join([chr(i) for i in range(32, 127)])
        self.input_size = (1024, 128, 1)
        self.max_text_length = 128
