"""Train the model with the dataset parameter name"""

import sys
import os
import argparse
import datetime as time

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
    run_name = time.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    batch_size = 32
    data_gen = data.Generator(
        args=args,
        input_shape=model.INPUT_SIZE,
        batch_size=batch_size,
    )

    # callbacks = callbacks(data)

    htr = model.HTR(data_gen)
    htr.model.summary()

    htr.model.fit_generator(
        generator=data_gen.next_train(),
        steps_per_epoch=data_gen.train_steps,
        epochs=1,
        verbose=1,
        use_multiprocessing=True,
        shuffle=True,
    )

    # model.fit_generator(
    #     generator=data_generator(path.train_file, path.preproc, path.ground_truth),
    #     steps_per_epoch=1,
    #     epochs=1,
    #     verbose=1,
    #     use_multiprocessing=True,
    #     shuffle=True,
    # )


if __name__ == '__main__':
    main()
