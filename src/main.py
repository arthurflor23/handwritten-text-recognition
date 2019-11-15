"""
Provides options via the command line to perform project tasks.
* `--source`: dataset/model name (bentham, iam, rimes, saintgall, washington)
* `--arch`: network to be used (puigcerver, bluche, flor)
* `--transform`: transform dataset to the HDF5 file
* `--cv2`: visualize sample from transformed dataset
* `--image`: predict a single image with the source parameter
* `--train`: train model with the source argument
* `--test`: evaluate and predict model with the source argument
* `--kaldi_assets`: save all assets for use with kaldi
* `--epochs`: number of epochs
* `--batch_size`: number of batches
"""

import os
import argparse
import h5py
import cv2
import time
import numpy as np

from data import preproc as pp, evaluation
from data.generator import DataGenerator, Tokenizer
from data.reader import Dataset
from kaldiio import WriteHelper
from network import architecture
from network.model import HTRModel


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, required=True)
    parser.add_argument("--arch", type=str, default="flor")

    parser.add_argument("--transform", action="store_true", default=False)
    parser.add_argument("--cv2", action="store_true", default=False)
    parser.add_argument("--image", type=str, default="")

    parser.add_argument("--train", action="store_true", default=False)
    parser.add_argument("--test", action="store_true", default=False)
    parser.add_argument("--kaldi_assets", action="store_true", default=False)

    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()

    raw_path = os.path.join("..", "raw", args.source)
    source = os.path.join("..", "data", f"{args.source}.hdf5")
    output_path = os.path.join("..", "output", args.source, args.arch)

    input_size = (1024, 128, 1)
    max_text_length = 128
    charset_base = "".join([chr(i) for i in range(32, 127)])

    if args.transform:
        assert os.path.exists(raw_path)
        print(f"The {args.source} dataset will be transformed...")

        ds = Dataset(source=raw_path, name=args.source)
        ds.read_partitions()
        ds.preprocess_partitions(image_input_size=input_size)

        os.makedirs(os.path.dirname(source), exist_ok=True)

        for i in ds.partitions:
            with h5py.File(source, "a") as hf:
                hf.create_dataset(f"{i}/dt", data=ds.dataset[i]['dt'], compression="gzip", compression_opts=9)
                hf.create_dataset(f"{i}/gt", data=ds.dataset[i]['gt'], compression="gzip", compression_opts=9)
                print(f"[OK] {i} partition.")

        print(f"Transformation finished.")

    elif args.cv2:
        with h5py.File(source, "r") as hf:
            dt = hf['test']['dt'][:]
            gt = hf['test']['gt'][:]

        predict_file = os.path.join(output_path, "predict.txt")
        predicts = [""] * len(dt)

        if os.path.isfile(predict_file):
            with open(predict_file, "r") as lg:
                predicts = [line[5:] for line in lg if line.startswith("TE_P")]

        for x in range(len(dt)):
            print(f"Image shape:\t{dt[x].shape}")
            print(f"Ground truth:\t{gt[x].decode()}")
            print(f"Predict:\t{predicts[x]}\n")

            cv2.imshow("img", pp.adjust_to_see(dt[x]))
            cv2.waitKey(0)

    elif args.image:
        image = cv2.imread(args.image, cv2.IMREAD_GRAYSCALE)
        image = pp.preproc(image, img_size=input_size)

        x_test = pp.normalization([image])
        x_test_len = np.asarray([max_text_length])

        network_func = getattr(architecture, args.arch)

        tokenizer = Tokenizer(chars=charset_base, max_text_length=max_text_length)
        ioo = network_func(input_size=input_size, output_size=(tokenizer.vocab_size + 1))

        model = HTRModel(inputs=ioo[0], outputs=ioo[1], top_paths=10)
        model.compile(optimizer=ioo[2])

        model.load_checkpoint(target=os.path.join(output_path, "checkpoint_weights.hdf5"))

        predicts, probabilities = model.predict_on_batch([x_test, x_test_len])
        predicts = [[tokenizer.decode(x) for x in y] for y in predicts]

        print("\n####################################")

        for (pd1, pb1) in zip(predicts, probabilities):
            print("\nProb.  - Predict")

            for (pd2, pb2) in zip(pd1, pb1):
                print(f"{pb2:.4f} - {pd2}")

        print("\n####################################")

        cv2.imshow("Image", pp.adjust_to_see(image))
        cv2.waitKey(0)

    else:
        os.makedirs(output_path, exist_ok=True)

        dtgen = DataGenerator(source=source,
                              batch_size=args.batch_size,
                              charset=charset_base,
                              max_text_length=max_text_length,
                              predict=args.test)

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
                                    steps_per_epoch=dtgen.steps['train'],
                                    validation_data=dtgen.next_valid_batch(),
                                    validation_steps=dtgen.steps['valid'],
                                    callbacks=callbacks,
                                    shuffle=True,
                                    verbose=1)
            total_time = time.time() - start_time

            loss = h.history['loss']
            val_loss = h.history['val_loss']

            min_val_loss = min(val_loss)
            min_val_loss_i = val_loss.index(min_val_loss)

            time_epoch = (total_time / len(loss))
            total_item = (dtgen.size['train'] + dtgen.size['valid'])

            t_corpus = "\n".join([
                f"Total train images:      {dtgen.size['train']}",
                f"Total validation images: {dtgen.size['valid']}",
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
                                               steps=dtgen.steps['test'],
                                               use_multiprocessing=True,
                                               verbose=1)

            predicts = [dtgen.tokenizer.decode(x) for x in predicts[:,0]]
            total_time = time.time() - start_time

            with open(os.path.join(output_path, "predict.txt"), "w") as lg:
                for pd, gt in zip(predicts, dtgen.dataset['test']['gt']):
                    lg.write(f"TE_L {gt}\nTE_P {pd}\n")

            evaluate = evaluation.ocr_metrics(predicts=predicts,
                                              ground_truth=dtgen.dataset['test']['gt'],
                                              norm_accentuation=False,
                                              norm_punctuation=False)

            e_corpus = "\n".join([
                f"Total test images:    {dtgen.size['test']}",
                f"Total time:           {(total_time / 60):.2f} min",
                f"Time per item:        {(total_time / dtgen.size['test']):.8f} sec\n",
                f"Metrics:",
                f"Character Error Rate: {evaluate[0]:.8f}",
                f"Word Error Rate:      {evaluate[1]:.8f}",
                f"Sequence Error Rate:  {evaluate[2]:.8f}"
            ])

            with open(os.path.join(output_path, "evaluate.txt"), "w") as lg:
                lg.write(e_corpus)
                print(e_corpus)

        elif args.kaldi_assets:
            predicts = model.predict_generator(generator=dtgen.next_test_batch(),
                                               steps=dtgen.steps['test'],
                                               use_multiprocessing=True,
                                               raw_returns=True,
                                               verbose=1)

            kaldi_path = os.path.join(output_path, "kaldi")
            os.makedirs(kaldi_path, exist_ok=True)

            ark_file_name = os.path.join(kaldi_path, "conf_mats.ark")
            scp_file_name = os.path.join(kaldi_path, "conf_mats.scp")

            with WriteHelper(f"ark,scp:{ark_file_name},{scp_file_name}") as writer:
                for i in range(len(predicts)):
                    writer(str(i), predicts[i])

            # soon...
