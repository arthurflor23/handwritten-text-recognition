"""Add the environment paths to the args input"""

from os.path import join


def setup_path(args):
    """Set paths to the args"""

    setattr(args, "source", join("..", "data", args.dataset))
    setattr(args, "raw_source", join("..", "data", f"raw_{args.dataset}"))
    setattr(args, "output", join("..", args.output, args.dataset))

    setattr(args, "data", join(args.source, "lines"))
    setattr(args, "ground_truth", join(args.source, "ground_truth"))
    setattr(args, "partitions", join(args.source, "partitions"))

    setattr(args, "train_file", join(args.partitions, "train.txt"))
    setattr(args, "validation_file", join(args.partitions, "validation.txt"))
    setattr(args, "test_file", join(args.partitions, "test.txt"))

    return args
