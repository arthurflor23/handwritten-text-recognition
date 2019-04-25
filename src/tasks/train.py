"""Train the model with the dataset parameter name"""

import sys
import os
import argparse

try:
    sys.path[0] = os.path.join(sys.path[0], "..")
    from environment import setup_path
    from network import data, model
except ImportError as exc:
    sys.exit(f"Import error in '{__file__}': {exc}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    args = setup_path(args)

    # train_data, target_data, validation_data = imread_train(env)
    # htr = model.HTR()
    # htr.model.summary()

    # htr.model.fit_generator(
    #     generator=data_generator(path.train_file, path.preproc, path.ground_truth),
    #     steps_per_epoch=1,
    #     epochs=1,
    #     verbose=1,
    #     use_multiprocessing=True)

    # htr.model.fit_generator(
    #     generator=img_gen.next_train(),
    #     steps_per_epoch=(words_per_epoch - val_words) // minibatch_size,
    #     epochs=stop_epoch,
    #     validation_data=img_gen.next_val(),
    #     validation_steps=val_words // minibatch_size,
    #     callbacks=[viz_cb, img_gen],
    #     initial_epoch=start_epoch)


if __name__ == '__main__':
    main()
