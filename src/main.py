"""
Provides options via the command line to perform project tasks.
"""

from environment import Path
from network.network import HTRNetwork
from data.generator import DataGenerator
import importlib
import argparse
import os


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--transform", action="store_true", default=False)
    parser.add_argument("--train", action="store_true", default=False)
    parser.add_argument("--eval", action="store_true", default=False)
    parser.add_argument("--test", action="store_true", default=False)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch", type=int, default=2)
    args = parser.parse_args()

    env = Path(args.dataset, args.epochs, args.batch)

    if args.transform:
        """Transform dataset to the project standard"""

        print(f"The {args.dataset} dataset will be transformed for the project...")
        package = f"dt_transform.{os.path.basename(env.source)}"
        transform = importlib.import_module(package)

        if not os.path.exists(env.raw_source):
            os.rename(env.source, env.raw_source)

        transform.partitions(env)
        transform.ground_truth(env)
        transform.data(env)
        print(f"Transformation finished.")

    elif args.train:
        """Train model with dataset parameter name"""

        dtgen = DataGenerator(env, train=True)
        htr = HTRNetwork(env.output, dtgen)
        htr.summary_to_file()

        htr.model.fit_generator(
            generator=dtgen.next_train_batch(),
            epochs=env.epochs,
            steps_per_epoch=dtgen.train_steps,
            validation_data=dtgen.next_val_batch(),
            validation_steps=dtgen.val_steps,
            callbacks=htr.callbacks,
            verbose=1)

    elif args.eval:
        """Evaluate model with dataset parameter name"""

        dtgen = DataGenerator(env)
        htr = HTRNetwork(env.output, dtgen)

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

        with open(os.path.join(env.output, "evaluate.txt"), "w") as ev:
            ev.write("\n".join(eval_corpus))

    elif args.test:
        """Test model with dataset parameter name"""

        dtgen = DataGenerator(env)
        htr = HTRNetwork(env.output, dtgen)

        pred = htr.model.predict_generator(
            generator=dtgen.next_test_batch(),
            steps=dtgen.test_steps,
            verbose=1,
            decode_func=dtgen.decode_ctc
        )

        pred_corpus = ["L || P"] + [f"{l} || {p}" for (p, l) in zip(pred[0], pred[1])]

        with open(os.path.join(env.output, "predict.txt"), "w") as ev:
            ev.write("\n".join(pred_corpus))
