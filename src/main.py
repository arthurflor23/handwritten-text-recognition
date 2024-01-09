"""
Provides options via the command line to perform project tasks.
* `--source`: dataset/model name (bentham, iam, rimes, saintgall, washington)
* `--arch`: network to be used (puigcerver, bluche, flor)
* `--transform`: transform dataset to the HDF5 file
* `--cv2`: visualize sample from transformed dataset
* `--kaldi_assets`: save all assets for use with kaldi
* `--image`: predict a single image with the source parameter
* `--train`: train model with the source argument
* `--test`: evaluate the predict model with the source argument
* `--evaluate`: evaluate the model outputs with an arbitrary directory
* `--norm_accentuation`: discard accentuation marks in the evaluation
* `--norm_punctuation`: discard punctuation marks in the evaluation
* `--epochs`: number of epochs
* `--batch_size`: number of batches
"""

import os
import cv2
import h5py
import datetime
import argparse

from data import preproc as pp, evaluation
from data.generator import DataGenerator, Tokenizer
from data.reader import Dataset

from network.model import HTRModel
from language.model import LanguageModel


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, required=True)
    parser.add_argument("--arch", type=str, default="flor")

    parser.add_argument("--transform", action="store_true", default=False)
    parser.add_argument("--cv2", action="store_true", default=False)
    parser.add_argument("--image", type=str, default="")

    parser.add_argument("--train", action="store_true", default=False)
    parser.add_argument("--test", action="store_true", default=False)

    parser.add_argument("--evaluate", action="store_true", default=False)
    parser.add_argument("--predictions_path", default=None)

    parser.add_argument("--kaldi_assets", action="store_true", default=False)
    parser.add_argument("--lm", action="store_true", default=False)
    parser.add_argument("--N", type=int, default=2)

    parser.add_argument("--norm_accentuation", action="store_true", default=False)
    parser.add_argument("--norm_punctuation", action="store_true", default=False)

    parser.add_argument("--epochs", type=int, default=10000)
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()

    raw_path = os.path.join("..", "raw", args.source)
    source_path = os.path.join("..", "data", f"{args.source}.hdf5")
    output_path = os.path.join("..", "output", args.source, args.arch)
    target_path = os.path.join(output_path, "checkpoint_weights.hdf5")

    input_size = (1024, 128, 1)
    max_text_length = 256

    charset_base = (" !\"#$%'(),-.0123456789:;?@\\]"
                    "ABCDEFGHIJKLMNOPQRSTUVWXZ"
                    "abcdefghijklmnopqrstuvwxyz"
                    "ªºÀÁÉÚàáâãçèéêíóôõú")

    if args.transform:
        print(f"{args.source} dataset will be transformed...")
        ds = Dataset(source=raw_path, name=args.source)
        ds.read_partitions()
        ds.save_partitions(source_path, input_size, max_text_length)

    elif args.cv2:
        with h5py.File(source_path, "r") as hf:
            dt = hf['test']['dt'][:256]
            gt = hf['test']['gt'][:256]

        predict_file = os.path.join(output_path, "predict.txt")
        predicts = [''] * len(dt)

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
        tokenizer = Tokenizer(chars=charset_base, max_text_length=max_text_length)

        img = pp.preprocess(args.image, input_size=input_size)
        x_test = pp.normalization([img])

        model = HTRModel(architecture=args.arch,
                         input_size=input_size,
                         vocab_size=tokenizer.vocab_size,
                         beam_width=30,
                         top_paths=10)

        model.compile(learning_rate=0.001)
        model.load_checkpoint(target=target_path)

        predicts, probabilities = model.predict(x_test, ctc_decode=True)
        predicts = [[tokenizer.decode(x) for x in y] for y in predicts]

        print("\n####################################")
        for i, (pred, prob) in enumerate(zip(predicts, probabilities)):
            print("\nProb.  - Predict")

            for (pd, pb) in zip(pred, prob):
                print(f"{pb:.4f} - {pd}")

            cv2.imshow(f"Image {i + 1}", cv2.imread(args.image))
        print("\n####################################")
        cv2.waitKey(0)

    else:
        assert os.path.isfile(source_path) or os.path.isfile(target_path)
        os.makedirs(output_path, exist_ok=True)

        dtgen = DataGenerator(source=source_path,
                              batch_size=args.batch_size,
                              charset=charset_base,
                              max_text_length=max_text_length,
                              predict=(not args.kaldi_assets) and args.test)

        model = HTRModel(architecture=args.arch,
                         input_size=input_size,
                         vocab_size=dtgen.tokenizer.vocab_size,
                         beam_width=30,
                         reduce_tolerance=20,
                         stop_tolerance=30)

        model.compile(learning_rate=0.001)
        model.load_checkpoint(target=target_path)

        if args.kaldi_assets:
            predicts, _ = model.predict(x=dtgen.next_test_batch(), steps=dtgen.steps['test'], ctc_decode=False)

            lm = LanguageModel(output_path, args.N)
            lm.generate_kaldi_assets(dtgen, predicts)

        elif args.lm:
            lm = LanguageModel(output_path, args.N)
            ground_truth = [x.decode() for x in dtgen.dataset['test']['gt']]

            start_time = datetime.datetime.now()

            predicts, _ = model.predict(x=dtgen.next_test_batch(), steps=dtgen.steps['test'], ctc_decode=False)
            lm.generate_kaldi_assets(dtgen, predicts)

            lm.kaldi(predict=False)
            predicts = lm.kaldi(predict=True)

            total_time = datetime.datetime.now() - start_time

            with open(os.path.join(output_path, "predict_kaldi.txt"), "w") as lg:
                for pd, gt in zip(predicts, ground_truth):
                    lg.write(f"TE_L {gt}\nTE_P {pd}\n")

            evaluate = evaluation.ocr_metrics(predicts=predicts,
                                              ground_truth=ground_truth,
                                              norm_accentuation=args.norm_accentuation,
                                              norm_punctuation=args.norm_punctuation)

            e_corpus = "\n".join([
                f"Total test images:    {dtgen.size['test']}",
                f"Total time:           {total_time}",
                f"Time per item:        {total_time / dtgen.size['test']}\n",
                "Metrics:",
                f"Character Error Rate: {evaluate[0]:.8f}",
                f"Word Error Rate:      {evaluate[1]:.8f}",
                f"Sequence Error Rate:  {evaluate[2]:.8f}"
            ])

            sufix = ("_norm" if args.norm_accentuation or args.norm_punctuation else "") + \
                    ("_accentuation" if args.norm_accentuation else "") + \
                    ("_punctuation" if args.norm_punctuation else "")

            with open(os.path.join(output_path, f"evaluate_kaldi{sufix}.txt"), "w") as lg:
                lg.write(e_corpus)
                print(e_corpus)

        elif args.train:
            model.summary(output_path, "summary.txt")
            callbacks = model.get_callbacks(logdir=output_path, checkpoint=target_path, verbose=1)

            start_time = datetime.datetime.now()

            h = model.fit(x=dtgen.next_train_batch(),
                          epochs=args.epochs,
                          steps_per_epoch=dtgen.steps['train'],
                          validation_data=dtgen.next_valid_batch(),
                          validation_steps=dtgen.steps['valid'],
                          callbacks=callbacks,
                          shuffle=True,
                          verbose=1)

            total_time = datetime.datetime.now() - start_time

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
                f"Total time:              {total_time}",
                f"Time per epoch:          {time_epoch}",
                f"Time per item:           {time_epoch / total_item}\n",
                f"Total epochs:            {len(loss)}",
                f"Best epoch               {min_val_loss_i + 1}\n",
                f"Training loss:           {loss[min_val_loss_i]:.8f}",
                f"Validation loss:         {min_val_loss:.8f}"
            ])

            with open(os.path.join(output_path, "train.txt"), "w") as lg:
                lg.write(t_corpus)
                print(t_corpus)

        elif args.test:
            start_time = datetime.datetime.now()

            predicts, _ = model.predict(x=dtgen.next_test_batch(),
                                        steps=dtgen.steps['test'],
                                        ctc_decode=True,
                                        verbose=1)

            predicts = [dtgen.tokenizer.decode(x[0]) for x in predicts]
            ground_truth = [x.decode() for x in dtgen.dataset['test']['gt']]

            total_time = datetime.datetime.now() - start_time

            with open(os.path.join(output_path, "predict.txt"), "w") as lg:
                for pd, gt in zip(predicts, ground_truth):
                    lg.write(f"TE_L {gt}\nTE_P {pd}\n")

            evaluate = evaluation.ocr_metrics(predicts=predicts,
                                              ground_truth=ground_truth,
                                              norm_accentuation=args.norm_accentuation,
                                              norm_punctuation=args.norm_punctuation)

            e_corpus = "\n".join([
                f"Total test images:    {dtgen.size['test']}",
                f"Total time:           {total_time}",
                f"Time per item:        {total_time / dtgen.size['test']}\n",
                "Metrics:",
                f"Character Error Rate: {evaluate[0]:.8f}",
                f"Word Error Rate:      {evaluate[1]:.8f}",
                f"Sequence Error Rate:  {evaluate[2]:.8f}"
            ])

            sufix = ("_norm" if args.norm_accentuation or args.norm_punctuation else "") + \
                    ("_accentuation" if args.norm_accentuation else "") + \
                    ("_punctuation" if args.norm_punctuation else "")

            with open(os.path.join(output_path, f"evaluate{sufix}.txt"), "w") as lg:
                lg.write(e_corpus)
                print(e_corpus)

            #####################################################

            ds = Dataset(source=raw_path, name=args.source)
            ds.read_partitions()

            preds_path = os.path.join(output_path, 'predictions')
            os.makedirs(preds_path, exist_ok=True)

            for i in range(len(ds.dataset['test']['path'])):
                filename = os.path.basename(ds.dataset['test']['path'][i]).split('.')[0]
                with open(os.path.join(preds_path, f"{filename}.txt"), "w") as lg:
                    lg.write(predicts[i])

        elif args.evaluate:
            start_time = datetime.datetime.now()

            ds = Dataset(source=raw_path, name=args.source)
            ds.read_partitions()

            if 'path' in ds.dataset['test']:
                preds_path = args.predictions_path or os.path.join(output_path, 'predictions')

                predicts = []
                ground_truth = []

                for i in range(len(ds.dataset['test']['path'])):
                    pred_path = os.path.join(preds_path, os.path.basename(ds.dataset['test']['path'][i]))

                    if os.path.isfile(pred_path):
                        predicts.append(' '.join(open(pred_path).read().splitlines()))
                    else:
                        predicts.append('')

                    ground_truth.append(' '.join(open(ds.dataset['test']['path'][i]).read().splitlines()))

                evaluate = evaluation.ocr_metrics(predicts=predicts, ground_truth=ground_truth)

                e_corpus = "\n".join([
                    f"Total test images:    {dtgen.size['test']}",
                    "Metrics:",
                    f"Character Error Rate: {evaluate[0]:.8f}",
                    f"Word Error Rate:      {evaluate[1]:.8f}",
                    f"Sequence Error Rate:  {evaluate[2]:.8f}"
                ])

                print(e_corpus)
