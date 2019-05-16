"""Set paths and environment variables"""

import os


class Environment():

    def __init__(self, args):
        self.data = os.path.join("..", "data")
        self.source = os.path.join(self.data, f"{args.dataset}.hdf5")
        self.raw_source = os.path.join("..", "raw", args.dataset)
        self.output = os.path.join("..", "output", args.dataset)

        self.epochs = args.epochs
        self.batch_size = args.batch_size
        self.lazy_loading = args.lazy_loading
        self.gated = args.gated

        self.charset = "".join([chr(i) for i in range(32, 127)])
        self.input_size = (1024, 128, 1)
        self.max_text_length = 128
