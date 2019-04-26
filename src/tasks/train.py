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
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch", type=int, default=20)
    args = parser.parse_args()

    args = setup_path(args)

    data_gen = data.Generator(args, model.INPUT_SIZE, args.batch)
    htr = model.HTR(data_gen)

    htr.model.fit_generator(
        generator=data_gen.next_train(),
        steps_per_epoch=data_gen.train_steps,
        epochs=args.epochs,
        verbose=1,
        callbacks=htr.get_callbacks(),
        validation_data=data_gen.next_val(),
        validation_steps=data_gen.val_steps,
        use_multiprocessing=True
    )


if __name__ == '__main__':
    main()
