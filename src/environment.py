"""Set paths and environment variables"""

import os


class Environment():

    def __init__(self, args):
        profile = f"{args.level}_{args.dataset}"
        self.source = os.path.join("..", "data", f"{profile}.hdf5")
        self.output = os.path.join("..", "output", profile)
        self.raw_source = os.path.join("..", "raw", args.dataset)

        self.level = args.level
        self.epochs = args.epochs
        self.batch_size = args.batch_size

        self.lazy_loading = args.lazy_loading
        self.gated = args.gated
        self.charset = "".join([chr(i) for i in range(32, 127)])

        if self.level == "paragraph":
            self.input_size = (1200, 960, 1)
            self.max_text_length = 1200
        else:
            self.input_size = (1024, 128, 1)
            # self.input_size = (800, 64, 1)
            self.max_text_length = 128
