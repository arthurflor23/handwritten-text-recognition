"""ads"""

from settings import Environment, HTR
import argparse
import numpy as np
import tensorflow as tf
import os


def imread_train(env):
    """Load images and targets from train/validation partition"""

    def imread(partition):
        src_data, target_data = [], []

        with open(partition, "r") as file:
            data_list = [x.strip() for x in file.readlines()]

        for item in data_list:
            dt = np.load(os.path.join(env.preproc_dir, f"{item}.npy"))
            dt = np.reshape(dt, dt.shape+(1,))
            dt = np.reshape(dt, (1,)+dt.shape)
            src_data.append(dt)

            with open(os.path.join(env.gt_dir, f"{item}.txt"), "r") as file:
                lines = [x.strip() for x in file.readlines()]
                target_data.append(' '.join(lines))

        return src_data, target_data
    
    train_data, target_data = imread(env.train_file)
    validation_data, validation_target_data = imread(env.validation_file)

    return train_data, target_data, (validation_data, validation_target_data)


# def imread_yield(env, partition):
#     with open(partition, "r") as file:
#         data_list = [x.strip() for x in file.readlines()]

#     for item in data_list:
#         data = np.load(os.path.join(env.preproc_dir, f"{item}.npy"))
#         data = np.reshape(data, data.shape+(1,))
#         data = np.reshape(data, (1,)+data.shape)

#         with open(os.path.join(env.gt_dir, f"{item}.txt"), "r") as file:
#             lines = [x.strip() for x in file.readlines()]
#             target_data = np.array([' '.join(lines)])
#             target_data = np.reshape(target_data, target_data.shape+(1,))
#             target_data = np.reshape(target_data, (1,)+target_data.shape)
        
#         yield data, target_data

def main():
    """Get the input parameter and call normalization methods."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()
    
    env = Environment(args.dataset_dir, args.output_dir)
    train_data, target_data, validation_data = imread_train(env)

    model = HTR()
    model.summary()

    # logger = tf.keras.callbacks.CSVLogger("./temp.log")

    # model.fit(
    #     x=train_data,
    #     y=target_data,
    #     validation_data=validation_data,
    #     callbacks=[logger])


if __name__ == '__main__':
    main()
