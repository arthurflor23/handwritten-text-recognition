"""Set paths and environment variables"""

import os


class Environment():

    def __init__(self, args):
        self.raw_source = os.path.join("..", "raw", args.dataset)
        self.source = os.path.join("..", "data", f"{args.dataset}_{args.level}.hdf5")
        self.output = os.path.join("..", "output", f"{args.dataset}_{args.arch}_{args.level}")

        self.arch = args.arch
        self.level = args.level
        self.epochs = args.epochs
        self.batch_size = args.batch_size

        if self.level == "paragraph":
            self.input_size = (1024, 1280, 1)
            self.max_text_length = 1280
        else:
            self.input_size = (1024, 128, 1)
            self.max_text_length = 128

        self.charset = "".join([chr(i) for i in range(32, 127)])
