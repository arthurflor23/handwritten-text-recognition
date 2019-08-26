"""
Provides options via the command line to perform project tasks.
* `--dataset`: dataset name (bentham, iam, rimes, saintgall)
* `--arch`: network to be used (puigcerver, bluche, flor)
* `--transform`: transform dataset to the HDF5 file
* `--cv2`: visualize sample from transformed dataset
* `--train`: train model with the dataset argument
* `--test`: evaluate and predict model with the dataset argument
* `--epochs`: number of epochs
* `--batch_size`: number of batches
"""

import os
import importlib
import argparse
import h5py
import cv2
import time

from data import preproc as pp
from data.generator import DataGenerator
from network import architecture
from network.model import HTRModel


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--arch", type=str, default="flor")
    parser.add_argument("--transform", action="store_true", default=False)
    parser.add_argument("--cv2", action="store_true", default=False)
    parser.add_argument("--train", action="store_true", default=False)
    parser.add_argument("--test", action="store_true", default=False)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()

    raw_source = os.path.join("..", "raw", args.dataset)
    hdf5_src = os.path.join("..", "data", f"{args.dataset}.hdf5")
    output = os.path.join("..", "output", f"{args.dataset}_{args.arch}")

    input_size = (1024, 128, 1)
    max_text_length = 128
    charset = "".join([chr(i) for i in range(32, 127)])

    if args.transform:
        assert os.path.exists(raw_source)

        print(f"The {args.dataset} dataset will be transformed...")
        mod = importlib.import_module(f"transform.{args.dataset}")

        transform = mod.Transform(source=raw_source,
                                  target=hdf5_src,
                                  input_size=input_size,
                                  charset=charset,
                                  max_text_length=max_text_length,
                                  preproc=pp.preproc,
                                  encode=pp.encode_ctc)
        transform.line()
        print(f"Transformation finished.")

    elif args.cv2:
        with h5py.File(hdf5_src, "r") as hf:
            dt = hf["train"]["dt"][:]
            gt_bytes = hf["train"]["gt_bytes"][:]
            gt_sparse = hf["train"]["gt_sparse"][:]

        for x in range(len(dt)):
            print(f"Image shape: {dt[x].shape}")
            print(f"Ground truth: {gt_bytes[x].decode()}")
            print(f"Ground truth sparsed:\n{gt_sparse[x]}\n")

            cv2.imshow("img", dt[x])
            cv2.waitKey(0)

    elif args.train or args.test:
        dtgen = DataGenerator(hdf5_src=hdf5_src,
                              batch_size=args.batch_size,
                              max_text_length=max_text_length)

        network_func = getattr(architecture, args.arch)
        ioo = network_func(input_size=input_size, output_size=len(charset) + 1)

        model = HTRModel(inputs=ioo[0], outputs=ioo[1], charset=charset)
        model.compile(optimizer=ioo[2])

        checkpoint = "checkpoint_weights.hdf5"
        model.load_checkpoint(output, checkpoint)

        if args.train:
            model.summary(output, "summary.txt")
            callbacks = model.callbacks(logdir=output, hdf5_target=checkpoint)

            start_time = time.time()
            h = model.fit_generator(generator=dtgen.next_train_batch(),
                                    epochs=args.epochs,
                                    steps_per_epoch=dtgen.train_steps,
                                    validation_data=dtgen.next_valid_batch(),
                                    validation_steps=dtgen.valid_steps,
                                    callbacks=callbacks,
                                    shuffle=True,
                                    verbose=1)
            total_time = time.time() - start_time

            loss = h.history['loss']
            val_loss = h.history['val_loss']

            min_val_loss = min(val_loss)
            min_val_loss_i = val_loss.index(min_val_loss)

            train_corpus = "\n".join([
                f"Total train images:       {dtgen.total_train}",
                f"Total validation images:  {dtgen.total_valid}",
                f"Batch:                    {args.batch_size}\n",
                f"Total time:               {(total_time / 60):.0f} min",
                f"Average time per epoch:   {(total_time / len(loss)):.0f} sec\n",
                f"Total epochs:             {len(loss)}",
                f"Best epoch                {min_val_loss_i + 1}\n",
                f"Training loss:            {loss[min_val_loss_i]:.4f}",
                f"Validation loss:          {min_val_loss:.4f}"
            ])

            with open(os.path.join(output, "train.txt"), "w") as lg:
                print(f"\n{train_corpus}")
                lg.write(train_corpus)

        else:
            predict, evaluate = model.predict_generator(generator=dtgen.next_test_batch(),
                                                        steps=dtgen.test_steps,
                                                        metrics=["loss", "cer", "wer"],
                                                        norm_accentuation=False,
                                                        norm_punctuation=False,
                                                        verbose=1)

            eval_corpus = "\n".join([
                f"Total test images:    {dtgen.total_test}\n",
                f"Metrics:",
                f"Test Loss:            {evaluate[0]:.4f}",
                f"Character Error Rate: {evaluate[1]:.4f}",
                f"Word Error Rate:      {evaluate[2]:.4f}"
            ])

            with open(os.path.join(output, "evaluate.txt"), "w") as lg:
                print(f"\n{eval_corpus}")
                lg.write(eval_corpus)

            pred_corpus = "\n".join([f"L {l}\nS {p}\n" for (l, p) in zip(predict[0], predict[1])])

            with open(os.path.join(output, "predict.txt"), "w") as lg:
                lg.write(pred_corpus)
