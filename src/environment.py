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
        self.charset = "!\"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[]_abcdefghijklmnopqrstuvwxyz{|} "

        if self.level == "paragraph":
            self.input_size = (1024, 1280, 1)
            self.max_text_length = 1280
        else:
            self.input_size = (1024, 128, 1)
            self.max_text_length = 128
