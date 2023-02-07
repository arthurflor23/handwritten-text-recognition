import argparse
import time
import os
import string
import tarfile
import fastparquet
import shutil
import csv
import cv2
import matplotlib.pyplot as plt

from zipfile import ZipFile
from tqdm import tqdm

from data import preproc as pp
from data.generator import Tokenizer
from data.reader import Dataset

from network.model import HTRModel

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--transform", action="store_true", default=False)
    parser.add_argument("--labels", type=str)

    parser.add_argument("--source", type=str, required=True)
    parser.add_argument("--weights", type=str)
    parser.add_argument("--arch", type=str, default="flor")

    parser.add_argument("--archive", type=bool, default=False)
    parser.add_argument("--csv", type=str, default="")
    parser.add_argument("--parquet", type=str, default="")

    args = parser.parse_args()

    source_path = args.source
    weights_path = args.weights
    dataset_path = "../data/" + os.path.basename(source_path) + ".hdf5"

    input_size = (1024, 128, 1)
    max_text_length = 50
    charset_base = string.printable[:95]

    start_time = time.time()

    if args.transform:
        ds = Dataset(source="census", name="census", images=args.source, labels=args.labels)
        ds.read_partitions()
        ds.save_partitions(dataset_path, input_size, max_text_length)

    # elif args.train:
    #     assert os.path.isfile(source_path) or os.path.isfile(weights_path)
    #     os.makedirs(output_path, exist_ok=True)
    #
    #     dtgen = DataGenerator(source=source_path,
    #                           batch_size=args.batch_size,
    #                           charset=charset_base,
    #                           max_text_length=max_text_length,
    #                           predict=(not args.kaldi_assets) and args.test)
    #
    #     model = HTRModel(architecture=args.arch,
    #                      input_size=input_size,
    #                      vocab_size=dtgen.tokenizer.vocab_size,
    #                      beam_width=10,
    #                      stop_tolerance=20,
    #                      reduce_tolerance=15)
    #
    #     model.compile(learning_rate=0.001)
    #     model.load_checkpoint(target=target_path)
    #
    #     model.summary(output_path, "summary.txt")
    #     callbacks = model.get_callbacks(logdir=output_path, checkpoint=target_path, verbose=1)
    #
    #     start_time = datetime.datetime.now()
    #
    #     h = model.fit(x=dtgen.next_train_batch(),
    #                   epochs=args.epochs,
    #                   steps_per_epoch=dtgen.steps['train'],
    #                   validation_data=dtgen.next_valid_batch(),
    #                   validation_steps=dtgen.steps['valid'],
    #                   callbacks=callbacks,
    #                   shuffle=True,
    #                   verbose=1)
    #
    #     total_time = datetime.datetime.now() - start_time
    #
    #     loss = h.history['loss']
    #     val_loss = h.history['val_loss']
    #
    #     min_val_loss = min(val_loss)
    #     min_val_loss_i = val_loss.index(min_val_loss)
    #
    #     time_epoch = (total_time / len(loss))
    #     total_item = (dtgen.size['train'] + dtgen.size['valid'])
    #
    #     t_corpus = "\n".join([
    #         f"Total train images:      {dtgen.size['train']}",
    #         f"Total validation images: {dtgen.size['valid']}",
    #         f"Batch:                   {dtgen.batch_size}\n",
    #         f"Total time:              {total_time}",
    #         f"Time per epoch:          {time_epoch}",
    #         f"Time per item:           {time_epoch / total_item}\n",
    #         f"Total epochs:            {len(loss)}",
    #         f"Best epoch               {min_val_loss_i + 1}\n",
    #         f"Training loss:           {loss[min_val_loss_i]:.8f}",
    #         f"Validation loss:         {min_val_loss:.8f}"
    #     ])
    #
    #     with open(os.path.join(output_path, "train.txt"), "w") as lg:
    #         lg.write(t_corpus)
    #         print(t_corpus)

    # The default mode is inference.
    else:
        final_predicts = []

        if args.archive:
            # Slurm allocates storage space for a job that isn't subject to the same file size and count restrictions
            # as the rest of the storage, so we will unpack the archive onto the tmp folder created.
            archive_path = args.source
            archive_file_type = archive_path.split(".", 1)[1]
            folder_path = os.environ['TMPDIR']
            if archive_file_type == "zip":
                with ZipFile(archive_path) as zip_file:
                    for member in zip_file.namelist():
                        filename = os.path.basename(member)
                        # skip directories
                        if not filename:
                            continue

                        # copy file (taken from zipfile's extract)
                        source = zip_file.open(member)
                        target = open(os.path.join(folder_path, filename), "wb")
                        with source, target:
                            shutil.copyfileobj(source, target)

            elif archive_file_type == "tar" or archive_file_type == "tar.gz":
                tar = tarfile.open(archive_path)
                for member in tar.getmembers():
                    if member.isreg():  # skip if the TarInfo is not files
                        member.name = os.path.basename(member.name)  # remove the path by reset it
                        tar.extract(member, folder_path)  # extract

            else:
                print("Invalid File type, accepted file types are zip, tar, and tar.gz")

        else:
            folder_path = args.source

        tokenizer = Tokenizer(chars=charset_base, max_text_length=max_text_length)
        model = HTRModel(architecture=args.arch,
                         input_size=input_size,
                         vocab_size=tokenizer.vocab_size,
                         beam_width=10)

        model.compile()
        model.load_checkpoint(target=weights_path)
        images = [x for x in os.listdir(folder_path) if x.split(".")[-1] == "jpg" or x.split(".")[-1] == "jp2"]
        total = len(images)
        pbar = tqdm(total=total)

        for image_name in images:
            image_path = os.path.join(folder_path, image_name)

            os.environ["OPENCV_IO_ENABLE_JASPER"] = "true"

            if cv2.haveImageReader(image_path):
                img = cv2.imread(image_path)
            else:
                img = plt.imread(image_path)

            # first check if image is a blank snippet
            vertical_crop = int(img.shape[0] * 0.1375)
            horizontal_crop = int(img.shape[1] * 0.2375)

            cropped_image = img[vertical_crop:-vertical_crop, horizontal_crop:-horizontal_crop]
            gray = cv2.cvtColor(cropped_image, cv2.COLOR_RGB2GRAY)

            thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
                                           19, 5)

            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

            white_pixels = cv2.countNonZero(closing)
            total_pixels = closing.shape[0] * closing.shape[1]
            white_percent = (white_pixels / total_pixels) * 100

            if white_percent > 92:
                final_predicts.append([image_name, "<BLANK>"])
                pbar.update(1)
                continue

            img = pp.preprocess(image_path, input_size=input_size)
            x_test = pp.normalization([img])

            predicts, probabilities = model.predict(x_test, ctc_decode=True)
            predicts = [[tokenizer.decode(x) for x in y] for y in predicts]

            if white_percent > 90:
                final_predicts.append([image_name, predicts[0][0] + "<BORDERLINE>", probabilities[0][0]])
            else:
                final_predicts.append([image_name, predicts[0][0], probabilities[0][0]])
            pbar.update(1)

        if args.csv:
            csv_name = os.path.basename(args.source) + ".csv"
            csv_path = os.path.join(args.csv, csv_name)
            with open(csv_path, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerows(final_predicts)
        elif args.parquet:
            parquet_path = os.path.join(args.csv, 'predicts.parquet')
            fastparquet.write(parquet_path, final_predicts)

        finish_time = time.time()
        total_time = finish_time - start_time
        print("Images Processed: ", len(images))
        print("Total Time elapsed: ", total_time / 60, " minutes")
        print("Time per image: ", total_time / len(images), "seconds")
