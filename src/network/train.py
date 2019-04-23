import os
import sys
import argparse
import numpy as np
import tensorflow as tf
from model import HTR


try:
    from settings import Environment
except ImportError:
    sys.path[0] = os.path.join(sys.path[0], "..")
    from settings import Environment


def imread_train(env):
    """Load images and targets from train/validation partition"""

    def imread(partition):
        src_data, target_data = [], []

        with open(partition, "r") as file:
            data_list = [x.strip() for x in file.readlines()]

        for item in data_list:
            src_data.append(np.load(os.path.join(env.preproc_dir, f"{item}.npy")))

            with open(os.path.join(env.gt_dir, f"{item}.txt"), "r") as file:
                lines = [x.strip() for x in file.readlines()]
                target_data.append(' '.join(lines))

        return src_data, target_data

    train_data, target_data = imread(env.train_file)
    validation_data, validation_target_data = imread(env.validation_file)

    return train_data, target_data, (validation_data, validation_target_data)


def train(args):
    """Get the input parameter and call normalization methods."""

    env = Environment(args.dataset_dir, args.output_dir)
    train_data, target_data, validation_data = imread_train(env)

    model = HTR(input_shape=env.img_size)
    model.summary()

    # checkpoint = tf.keras.callbacks.ModelCheckpoint(nn.fn_checkpoint, monitor=const.MONITOR, save_best_only=True, verbose=1)
    logger = tf.keras.callbacks.CSVLogger("../temp.log", append=True)

    model.fit(
        x=train_data,
        y=target_data,
        batch_size=64,
        epochs=1,
        verbose=1,
        callbacks=[logger],
        validation_data=validation_data,
        shuffle=True,
        steps_per_epoch=None,
        validation_steps=None,
        use_multiprocessing=True
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()
    train(args)
