"""Add the environment paths to the args input"""

from os.path import join


def setup_path(args):
    """Set paths to the args"""

    setattr(args, "SOURCE", join("..", "data", args.dataset))
    setattr(args, "RAW_SOURCE", join("..", "data", f"raw_{args.dataset}"))
    setattr(args, "OUTPUT", join("..", args.output, args.dataset))

    setattr(args, "DATA", join(args.SOURCE, "lines"))
    setattr(args, "GROUND_TRUTH", join(args.SOURCE, "ground_truth"))
    setattr(args, "PARTITIONS", join(args.SOURCE, "partitions"))

    setattr(args, "TRAIN_FILE", join(args.PARTITIONS, "train.txt"))
    setattr(args, "VALIDATION_FILE", join(args.PARTITIONS, "validation.txt"))
    setattr(args, "TEST_FILE", join(args.PARTITIONS, "test.txt"))

    return args
