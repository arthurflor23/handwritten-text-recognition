"""
Provides options via the command line to perform project tasks.
    * Transform dataset to this project standard
    * Visualize sample of the transformed dataset
    * Train model with dataset parameter name
    * Evaluate model with dataset parameter name
    * Test model with dataset parameter name
"""

import os
import importlib
import argparse
import h5py
import cv2

from data import preproc as pp
from data.generator import DataGenerator
from network import architecture as arch
from network.model import HTRModel
from environment import Environment


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--level", type=str, required=True)
    parser.add_argument("--transform", action="store_true", default=False)
    parser.add_argument("--cv2", action="store_true", default=False)
    parser.add_argument("--train", action="store_true", default=False)
    parser.add_argument("--test", action="store_true", default=False)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lazy_loading", type=int, default=1)
    parser.add_argument("--gated", type=int, default=1)
    args = parser.parse_args()

    env = Environment(args)

    if args.transform:
        assert os.path.exists(env.raw_source)

        print(f"The {args.dataset} dataset will be transformed...")
        package = f"transform.{args.dataset}"
        mod = importlib.import_module(package)

        trans = mod.Transform(env, pp.preproc, pp.encode_ctc)
        transform_func = getattr(trans, env.level)
        transform_func()
        print(f"Transformation finished.")

    elif args.cv2:
        with h5py.File(env.source, "r") as hf:
            dt = hf["train"]["dt"][0:23]
            gt = hf["train"]["gt"][0:23]

        for x in range(len(dt)):
            print(f"Image shape: {dt[x].shape}")
            print(f"Ground truth encoded:\n{gt[x]}\n")
            cv2.imshow("img", dt[x][:,:,0])
            cv2.waitKey(0)

    elif args.train or args.test:
        dtgen = DataGenerator(env)

        if env.gated:
            inputs, outputs, optimizer = arch.gated_cnn_1dlstm(env)
        else:
            inputs, outputs, optimizer = arch.cnn_1dlstm(env)

        model = HTRModel(inputs=inputs, outputs=outputs, charset=env.charset)
        model.compile(optimizer=optimizer)
        cp_weights = model.set_callbacks(env.output)

        if os.path.isfile(cp_weights):
            model.load_checkpoint(cp_weights)

        model.summary(output=env.output)

        if args.train:
            h = model.fit_generator(generator=dtgen.next_train_batch(),
                                    epochs=env.epochs,
                                    steps_per_epoch=dtgen.train_steps,
                                    validation_data=dtgen.next_valid_batch(),
                                    validation_steps=dtgen.valid_steps,
                                    callbacks=model.callbacks,
                                    shuffle=False,
                                    verbose=1)

            loss = h.history['loss']
            val_loss = h.history['val_loss']

            min_loss = min(loss)
            min_val_loss = min(val_loss)

            min_loss_i = loss.index(min_loss)
            min_val_loss_i = val_loss.index(min_val_loss)

            train_corpus = "\n".join([
                f"Total training images:   {dtgen.total_train}",
                f"Total validation images: {dtgen.total_valid}",
                f"Batch:                   {env.batch_size}\n",
                f"Total epochs:            {len(loss)}",
                f"Last loss:               {loss[-1]:.4f} ({len(loss)} epoch)",
                f"Last val_loss:           {val_loss[-1]:.4f} ({len(val_loss)} epoch)\n",
                f"Best validation loss:",
                f"Minimum val_loss:        {min_val_loss:.4f} ({min_val_loss_i} epoch)",
                f"Respective loss:         {loss[min_val_loss_i]:.4f} ({min_val_loss_i} epoch)"
            ])

            with open(os.path.join(env.output, "train.txt"), "w") as lg:
                print(f"\n{train_corpus}")
                lg.write(train_corpus)

        else:
            pred, eval = model.predict_generator(generator=dtgen.next_test_batch(),
                                                 steps=dtgen.test_steps,
                                                 verbose=1,
                                                 decode_func=pp.decode_ctc)

            eval_corpus = "\n".join([
                f"Total test images:    {dtgen.total_test}\n",
                f"Test loss:            {eval[0]:.4f}\n",
                f"Metrics:",
                f"Character Error Rate: {eval[1]:.4f}",
                f"Word Error Rate:      {eval[2]:.4f}",
                f"Line Error Rate:      {eval[3]:.4f}"
            ])

            pred_corpus = "\n".join(
                ["Label\t|\tPredict"] + [f"{l}\t|\t{p}" for (l, p) in zip(pred[0], pred[1])]
            )

            with open(os.path.join(env.output, "evaluate.txt"), "w") as lg:
                print(f"\n{eval_corpus}")
                lg.write(eval_corpus)

            with open(os.path.join(env.output, "predict.txt"), "w") as lg:
                lg.write(pred_corpus)
