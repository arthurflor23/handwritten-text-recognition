"""Evaluate and test the model with the dataset parameter name"""

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--batch", type=int, required=True)
    args = parser.parse_args()
    args = setup_path(args)

    dtgen = DataGenerator(args)
    htr = HTRNetwork(args.output, dtgen)

    eval = htr.model.evaluate_generator(
        generator=dtgen.next_test_batch(),
        steps=dtgen.test_steps,
        metrics=["loss", "ler", "ser"],
        verbose=1
    )

    eval_corpus = [
        f"Number of images:     {len(dtgen.test_list)}",
        f"Number of features:   {dtgen.nb_features}\n",
        f"Test dataset loss:    {eval[0][0]:.2f}",
        f"Label error rate:     {sum(eval[1])/len(eval[1]):.2f}",
        f"Sequence error rate   {eval[2]:.2f}",
    ]

    with open(os.path.join(args.output, "evaluate.txt"), "w") as ev:
        ev.write("\n".join(eval_corpus))

    pred = htr.model.predict_generator(
        generator=dtgen.next_test_batch(),
        steps=dtgen.test_steps,
        verbose=1,
        decode_func=dtgen.decode_ctc
    )

    pred_corpus = ["L || P"] + [f"{l} || {p}" for (p, l) in zip(pred[0], pred[1])]

    with open(os.path.join(args.output, "predict.txt"), "w") as ev:
        ev.write("\n".join(pred_corpus))
