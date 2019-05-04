"""
Provides options via the command line to perform project tasks.
    * Transform dataset to this project standard
    * Visualize sample of the transformed dataset with opencv
    * Train model with dataset parameter name
    * Evaluate model with dataset parameter name
    * Test model with dataset parameter name
"""

import os
import importlib
import argparse
import numpy as np
import cv2

from util.environment import Env
from util.loader import DataGenerator
from network.network import HTRNetwork
from util.preproc import preproc


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--transform", action="store_true", default=False)
    parser.add_argument("--cv2", action="store_true", default=False)
    parser.add_argument("--train", action="store_true", default=False)
    parser.add_argument("--eval", action="store_true", default=False)
    parser.add_argument("--test", action="store_true", default=False)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    env = Env(args)

    if args.transform:
        assert os.path.exists(env.raw_source)

        print(f"The {args.dataset} dataset will be transformed...")
        package = f"dt_transform.{args.dataset}"
        transform = importlib.import_module(package)

        if not os.path.exists(env.source):
            os.makedirs(env.source, exist_ok=True)

        transform.dataset(env, preproc)
        print(f"Transformation finished.")

    elif args.train:
        dtgen = DataGenerator(env, train=True)
        htr = HTRNetwork(env, dtgen)
        htr.model.summary()
        htr.summary_to_file()

        htr.model.fit_generator(
            generator=dtgen.next_train_batch(),
            epochs=env.epochs,
            steps_per_epoch=dtgen.train_steps,
            validation_data=dtgen.next_valid_batch(),
            validation_steps=dtgen.val_steps,
            callbacks=htr.callbacks,
            verbose=1)

    elif args.eval:
        dtgen = DataGenerator(env)
        htr = HTRNetwork(env, dtgen)

        eval = htr.model.evaluate_generator(
            generator=dtgen.next_test_batch(),
            steps=dtgen.test_steps,
            metrics=["loss", "ler", "ser"],
            verbose=1
        )

        eval_corpus = [
            f"Test dataset images:  {dtgen.test.shape[0]}\n",
            f"Test dataset loss:    {sum(eval[0][1])/len(eval[0][1]):.2f}",
            f"Label error rate:     {sum(eval[1])/len(eval[1]):.2f}",
            f"Sequence error rate:  {eval[2]:.2f}",
        ]

        with open(os.path.join(env.output, "evaluate.txt"), "w") as ev:
            ev.write("\n".join(eval_corpus))

    elif args.test:
        dtgen = DataGenerator(env)
        htr = HTRNetwork(env, dtgen)

        pred = htr.model.predict_generator(
            generator=dtgen.next_test_batch(),
            steps=dtgen.test_steps,
            verbose=1,
            decode_func=dtgen.decode_ctc
        )

        pred_corpus = ["Label | Predict"] + [f"{l} | {p}" for (l, p) in zip(pred[0], pred[1])]

        with open(os.path.join(env.output, "predict.txt"), "w") as ev:
            ev.write("\n".join(pred_corpus))

    elif args.cv2:
        sample = np.load(env.train, allow_pickle=True, mmap_mode="r")

        for x in range(23):
            print(sample["dt"].shape)
            print(sample["gt"][x])

            cv2.imshow("img", sample["dt"][x])
            cv2.waitKey(0)
