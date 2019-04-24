"""Train the model with the dataset parameter name."""

import argparse
import os
import numpy as np

from settings import model, environment as env


# def imread_train(env):
#     """Load images and targets from train/validation partition"""

#     def imread(partition):
#         src_data, target_data = [], []

#         with open(partition, "r") as file:
#             data_list = [x.strip() for x in file.readlines()]

#         for item in data_list:
#             data = np.load(os.path.join(env.preproc_dir, f"{item}.npy"))
#             data = np.reshape(data, data.shape + (1,))
#             data = np.reshape(data, (1,) + data.shape)
#             src_data.append(data)

#             with open(os.path.join(env.gt_dir, f"{item}.txt"), "r") as file:
#                 lines = [x.strip() for x in file.readlines()]
#                 target_data.append(' '.join(lines))

#         return src_data, target_data

#     train_data, target_data = imread(env.train_file)
#     validation_data, validation_target_data = imread(env.validation_file)

#     return train_data, target_data, (validation_data, validation_target_data)


def to_sparse(texts):
    """put ground truth texts into sparse tensor for ctc_loss"""

    # last entry must be max(labelList[i])
    shape = [len(texts), 0]
    indices = []
    values = []

    # go over all texts
    for (batch_element, text) in enumerate(texts):
        # convert to string of label (i.e. class-ids)
        label_str = [env.CHAR_LIST.index(c) for c in text]
        # sparse tensor must have size of max. label-string
        if len(label_str) > shape[1]:
            shape[1] = len(label_str)
        # put each label into sparse tensor
        for (i, label) in enumerate(label_str):
            indices.append([batch_element, i])
            values.append(label)

    return (indices, values, shape)


def imread_yield(partition, preproc_dir, gt_dir):

    with open(partition, "r") as file:
        data_list = [x.strip() for x in file.readlines()]

    import cv2

    for item in data_list:
        data = cv2.imread(os.path.join(preproc_dir, f"{item}.png"), cv2.IMREAD_GRAYSCALE)
        data = cv2.resize(data, dsize=(64, 800))

        # data = np.load(os.path.join(preproc_dir, f"{item}.npy"))
        data = np.reshape(data, data.shape + (1,))
        data = np.reshape(data, (1,) + data.shape)

        with open(os.path.join(gt_dir, f"{item}.txt"), "r") as file:
            lines = [x.strip() for x in file.readlines()]
            target_data = np.array([' '.join(lines)])
            # target_data = np.reshape(target_data, target_data.shape + (1,))
            target_data = np.reshape(target_data, (1,) + target_data.shape)

        yield ({"the_input": data,
                "the_label": target_data,
                "input_length": np.array([len(data[0][0]) * len(data[0][1])]),
                "label_length": np.array([len(target_data[0][0])])
                }, {'ctc': target_data})

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

    path = env.Path(args.dataset_dir, args.output_dir)
    # train_data, target_data, validation_data = imread_train(env)

    htr = model.HTR()
    htr.model.summary()

    # callbacks = htr.get_callbacks()
    # logger = tf.keras.callbacks.CSVLogger("./temp.log")

    htr.model.fit_generator(
        generator=imread_yield(path.train_file, path.data, path.ground_truth),
        steps_per_epoch=50,
        epochs=1,
        verbose=1,
        # callbacks=[cb_list],
        # validation_data=validdata.next_batch(),
        # validation_steps=50,
        # validation_freq=1,
        class_weight=None,
        max_queue_size=10,
        workers=1,
        use_multiprocessing=True,
        initial_epoch=0,
    )


if __name__ == '__main__':
    main()
