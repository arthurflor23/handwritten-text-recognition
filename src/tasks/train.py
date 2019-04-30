"""Train the model with the dataset parameter name"""

import sys
import os
import argparse

try:
    sys.path[0] = os.path.join(sys.path[0], "..")
    from environment import setup_path
    from data.generator import DataGenerator
    from network.network import HTRNetwork
except ImportError as exc:
    sys.exit(f"Import error in '{__file__}': {exc}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--batch", type=int, required=True)
    args = parser.parse_args()
    args = setup_path(args)

    dtgen = DataGenerator(args)
    htr = HTRNetwork(dtgen)
    htr.model.summary()

    htr.model.fit_generator(
        generator=dtgen.next_train(),
        epochs=args.epochs,
        steps_per_epoch=dtgen.train_steps,
        validation_data=dtgen.next_val(),
        validation_steps=dtgen.val_steps,
        callbacks=htr.callbacks,
        verbose=1
    )
