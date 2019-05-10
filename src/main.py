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
from util.preproc import preproc, encode_ctc, decode_ctc


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--transform", action="store_true", default=False)
    parser.add_argument("--cv2", action="store_true", default=False)
    parser.add_argument("--train", action="store_true", default=False)
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

        os.makedirs(env.data, exist_ok=True)
        transform.dataset(env, preproc, encode_ctc)
        print(f"Transformation finished.")

    elif args.cv2:
        sample = np.load(env.source, allow_pickle=True, mmap_mode="r")

        for x in range(23):
            print(sample["train_dt"].shape)
            print(sample["train_gt"][x])
            cv2.imshow("img", sample["train_dt"][x][:,:,0])
            cv2.waitKey(0)

    elif args.train:
        os.makedirs(env.output_tasks, exist_ok=True)

        dtgen = DataGenerator(env)
        htr = HTRNetwork(env, dtgen)

        htr.model.summary()
        htr.summary_to_file()

        h = htr.model.fit_generator(generator=dtgen.next_train_batch(),
                                    epochs=env.epochs,
                                    steps_per_epoch=dtgen.train_steps,
                                    validation_data=dtgen.next_valid_batch(),
                                    validation_steps=dtgen.val_steps,
                                    callbacks=htr.callbacks,
                                    shuffle=False,
                                    verbose=1)

        train_corpus = "\n".join([
            f"Epochs:                       {env.epochs}",
            f"Batch:                        {env.batch_size}\n",
            f"Train dataset images:         {dtgen.train.shape[0]}",
            f"Validation dataset images:    {dtgen.valid.shape[0]}\n",
            f"Train dataset loss:           {min(h.history['loss']):.4f}",
            f"Validation dataset loss:      {min(h.history['val_loss']):.4f}"
        ])

        with open(os.path.join(env.output_tasks, "train.txt"), "w") as lg:
            print(train_corpus)
            lg.write(train_corpus)

    elif args.test:
        os.makedirs(env.output_tasks, exist_ok=True)

        dtgen = DataGenerator(env)
        htr = HTRNetwork(env, dtgen)

        pred, eval = htr.model.predict_generator(generator=dtgen.next_test_batch(),
                                                 steps=dtgen.test_steps,
                                                 verbose=1,
                                                 decode_func=decode_ctc)

        eval_corpus = "\n".join([
            f"Test dataset images:  {dtgen.test.shape[0]}\n",
            f"Test dataset loss:    {eval[0]:.4f}",
            f"Character error rate: {eval[1]:.4f}",
            f"Word error rate:      {eval[2]:.4f}",
            f"Line error rate:      {eval[3]:.4f}"
        ])

        pred_corpus = "\n".join(
            ["Label\t|\tPredict"] + [f"{lb}\t|\t{pd}" for (lb, pd) in zip(pred[0], pred[1])]
        )

        with open(os.path.join(env.output_tasks, "evaluate.txt"), "w") as lg:
            print(eval_corpus)
            lg.write(eval_corpus)

        with open(os.path.join(env.output_tasks, "test.txt"), "w") as lg:
            print(f"\n{pred_corpus}")
            lg.write(pred_corpus)
