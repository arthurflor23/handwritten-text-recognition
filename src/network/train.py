import os
import sys
import argparse
from model import HTR


try:
    from settings import Environment
except ImportError:
    sys.path[0] = os.path.join(sys.path[0], "..")
    from settings import Environment


def train(args):
    """Get the input parameter and call normalization methods."""

    # print(args.dataset_dir)
    # print(args.output_dir)
    # print(args.train_steps)
    # print(args.learning_rate)

    env = Environment(args.dataset_dir, args.output_dir)
    model = HTR(input_shape=env.img_size)
    model.summary()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--train_steps", type=float, default=500)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    args = parser.parse_args()
    train(args)
