"""
Provides options via the command line to perform project tasks.
* `--dataset`: dataset name (bentham, iam, rimes, saintgall, washington)
* `--arch`: network to be used (puigcerver, bluche, flor)
* `--transform`: transform dataset to the HDF5 file
* `--cv2`: visualize sample from transformed dataset
* `--train`: train model with the dataset argument
* `--test`: evaluate and predict model with the dataset argument
* `--epochs`: number of epochs
* `--batch_size`: number of batches
"""

import os
import argparse
import h5py
import cv2
import time

from data import preproc as pp, evaluation
from data.generator import DataGenerator
from data.reader import Dataset
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

    raw_path = os.path.join("..", "raw", args.dataset)
    hdf5_src = os.path.join("..", "data", f"{args.dataset}.hdf5")
    output_path = os.path.join("..", "output", args.dataset, args.arch)

    input_size = (1024, 128, 1)
    max_text_length = 128
    charset_base = "".join([chr(i) for i in range(32, 127)])

    if args.transform:
        assert os.path.exists(raw_path)
        print(f"The {args.dataset} dataset will be transformed...")

        ds = Dataset(source=raw_path, name=args.dataset)
        ds.read_partitions()
        ds.preprocess_partitions(image_input_size=input_size)

        os.makedirs(os.path.dirname(hdf5_src), exist_ok=True)

        for i in ds.partitions:
            with h5py.File(hdf5_src, "a") as hf:
                hf.create_dataset(f"{i}/dt", data=ds.dataset[i]["dt"], compression="gzip", compression_opts=9)
                hf.create_dataset(f"{i}/gt", data=ds.dataset[i]["gt"], compression="gzip", compression_opts=9)
                print(f"[OK] {i} partition.")

        print(f"Transformation finished.")

    elif args.cv2:
        with h5py.File(hdf5_src, "r") as hf:
            dt = hf["test"]["dt"][:]
            gt = hf["test"]["gt"][:]

        for x in range(len(dt)):
            print(f"Image shape: {dt[x].shape}")
            print(f"Ground truth: {gt[x].decode()}\n")

            cv2.imshow("img", pp.adjust_to_see(dt[x]))
            cv2.waitKey(0)

    elif args.train or args.test:
        os.makedirs(output_path, exist_ok=True)

        dtgen = DataGenerator(hdf5_src=hdf5_src,
                              batch_size=args.batch_size,
                              charset=charset_base,
                              max_text_length=max_text_length)

        network_func = getattr(architecture, args.arch)

        ioo = network_func(input_size=input_size,
                           output_size=(dtgen.tokenizer.vocab_size + 1),
                           learning_rate=0.001)

        model = HTRModel(inputs=ioo[0], outputs=ioo[1])
        model.compile(optimizer=ioo[2])

        checkpoint = "checkpoint_weights.hdf5"
        model.load_checkpoint(target=os.path.join(output_path, checkpoint))

        if args.train:
            model.summary(output_path, "summary.txt")
            callbacks = model.get_callbacks(logdir=output_path, hdf5=checkpoint, verbose=1)

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

            time_epoch = (total_time / len(loss))
            total_item = (dtgen.total_train + dtgen.total_valid)

            t_corpus = "\n".join([
                f"Total train images:      {dtgen.total_train}",
                f"Total validation images: {dtgen.total_valid}",
                f"Batch:                   {dtgen.batch_size}\n",
                f"Total time:              {(total_time / 60):.2f} min",
                f"Time per epoch:          {(time_epoch / 60):.2f} min",
                f"Time per item:           {(time_epoch / total_item):.8f} sec\n",
                f"Total epochs:            {len(loss)}",
                f"Best epoch               {min_val_loss_i + 1}\n",
                f"Training loss:           {loss[min_val_loss_i]:.8f}",
                f"Validation loss:         {min_val_loss:.8f}"
            ])

            with open(os.path.join(output_path, "train.txt"), "w") as lg:
                lg.write(t_corpus)
                print(t_corpus)

        elif args.test:
            start_time = time.time()
            predicts = model.predict_generator(generator=dtgen.next_test_batch(),
                                               steps=dtgen.test_steps,
                                               use_multiprocessing=True,
                                               verbose=1)

            predicts = [dtgen.tokenizer.decode(x) for x in predicts]
            total_time = time.time() - start_time

            with open(os.path.join(output_path, "predict.txt"), "w") as lg:
                for pd, gt in zip(predicts, dtgen.dataset["test"]["gt"]):
                    lg.write(f"TE_L {gt}\nTE_P {pd}\n")

            evaluate = evaluation.ocr_metrics(predicts=predicts,
                                              ground_truth=dtgen.dataset["test"]["gt"],
                                              norm_accentuation=False,
                                              norm_punctuation=False)

            e_corpus = "\n".join([
                f"Total test images:    {dtgen.total_test}",
                f"Total time:           {(total_time / 60):.2f} min",
                f"Time per item:        {(total_time / dtgen.total_test):.8f} sec\n",
                f"Metrics:",
                f"Character Error Rate: {evaluate[0]:.8f}",
                f"Word Error Rate:      {evaluate[1]:.8f}",
                f"Sequence Error Rate:  {evaluate[2]:.8f}"
            ])

            with open(os.path.join(output_path, "evaluate.txt"), "w") as lg:
                lg.write(e_corpus)
                print(e_corpus)
