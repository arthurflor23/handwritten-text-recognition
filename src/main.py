import argparse
import time
import os
import string
import tarfile
import datetime
import fastparquet
import shutil
import csv
import psutil
import gc
import glob
import cv2
import matplotlib.pyplot as plt

from zipfile import ZipFile
from tqdm import tqdm

from data import preproc as pp
from data.generator import Tokenizer
from data.reader import Dataset
from data.generator import DataGenerator

from network.model import HTRModel
from BlankDetector import BlankDetector

BATCH_SIZE = 100

WRITE_BAD_TO_OWN_FILE = False # True

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--transform", action="store_true", default=False)
    parser.add_argument("--labels", type=str)
    parser.add_argument("--train", type=str, default="")

    parser.add_argument("--source", type=str)
    parser.add_argument("--weights", type=str, default="")
    parser.add_argument("--arch", type=str, default="flor")

    parser.add_argument("--archive", type=bool, default=False)
    parser.add_argument("--csv", type=str, default="")
    parser.add_argument("--append", action="store_true", default=False)
    parser.add_argument("--parquet", type=str, default="")

    parser.add_argument("--test", type=int, default=0)

    args = parser.parse_args()

    source_path = args.source
    weights_path = os.path.join("../weights", args.weights)

    input_size = (1024, 128, 1)
    max_text_length = 50
    charset_base = string.printable[:95]

    start_time = time.time()

    if args.transform:

        dataset_path = "../data/" + str.split(os.path.basename(args.labels), ".")[0] + ".hdf5"

        print('Transforming')
        print('Source Path:', source_path)
        print('Architecture:', args.arch)
        print('Labels:', args.labels)
        print('Dataset Path:', dataset_path)

        ds = Dataset(source="census", name="census", images=args.source, labels=args.labels)
        ds.read_partitions()
        ds.save_partitions(dataset_path, input_size, max_text_length)

    elif args.train:
        os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
        source_path = args.train
        log_path = "../logs"
        weights_path = "../weights/" + str.split(os.path.basename(source_path), ".")[0] + "Weights.hdf5"

        dtgen = DataGenerator(source=source_path,
                              batch_size=32,
                              charset=charset_base,
                              max_text_length=max_text_length)

        print('Training')
        print('Dataset:', args.train)
        print('Weights Path:', weights_path)
        print(f"Train images: {dtgen.size['train']}")
        print(f"Validation images: {dtgen.size['valid']}")
        print(f"Test images: {dtgen.size['test']}")

        os.environ["TF_ENABLE_AUTO_GC"] = "1"
        model = HTRModel(architecture=args.arch,
                         input_size=input_size,
                         vocab_size=dtgen.tokenizer.vocab_size,
                         beam_width=8,
                         stop_tolerance=18,
                         reduce_tolerance=10,
                         reduce_factor=0.1)

        model.compile(learning_rate=0.001)

        callbacks = model.get_callbacks(logdir=log_path, checkpoint=weights_path, verbose=1)

        start_time = datetime.datetime.now()

        h = model.fit(x=dtgen.next_train_batch(),
                      epochs=1000,
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

    # The default mode is inference.
    else:
        print('Not training or transforming.')
        print('Source Path:', source_path)
        print('Weights:', weights_path)
        print('Architecture:', args.arch)
        print('Archive:', args.archive)
        print('CSV:', args.csv)
        print('Append:', args.append)
        print('Parquet:', args.parquet)
        print('Test:', args.test)
                
        final_predicts = []
        images = []
        
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
                        images.append(filename)
                        target.close()

            elif archive_file_type == "tar" or archive_file_type == "tar.gz":
                tar = tarfile.open(archive_path)
                for member in tar.getmembers():
                    if member.isreg():  # skip if the TarInfo is not files
                        member.name = os.path.basename(member.name)  # remove the path by reset it
                        tar.extract(member, folder_path)  # extract
                        images.append(member.name)

            else:
                print("Invalid File type, accepted file types are zip, tar, and tar.gz")

        else:
            folder_path = args.source
            images = glob.iglob(f'{folder_path}/*')

        os.environ["TF_ENABLE_AUTO_GC"] = "1"
        os.environ["TF_RUN_EAGER_OP_AS_FUNCTION"] = 'false'
        tokenizer = Tokenizer(chars=charset_base, max_text_length=max_text_length)
        model = HTRModel(architecture=args.arch,
                         input_size=input_size,
                         vocab_size=tokenizer.vocab_size,
                         beam_width=10)

        model.compile()
        if not os.path.exists(weights_path):
            raise AssertionError("Weights don't exist")
        print(f'Loading weights from {weights_path}')
        model.load_checkpoint(target=weights_path)

        blank_detector = BlankDetector("./blank_detector.json")
        supported_extensions = ["jpg", "jpeg", "jpe", "jp2", "png"]
        total = len(os.listdir(folder_path))

        print('Total images:', total)
        print('-----------------')
        time.sleep(0.25)
        
        pbar = tqdm(images, total=total)

        out_path = None
        bad_path = os.path.join(args.csv, 'predicts_bad.csv')
        if args.csv:
            if args.csv.split(".")[-1] != "csv":
                out_path = os.path.join(args.csv, "predicts.csv")
            else:
                out_path = args.csv
                bad_path = args.csv.replace(".csv", "_bad.csv")
        elif args.parquet:
            out_path = os.path.join(args.csv, 'predicts.parquet')

        for i, image_path in enumerate(pbar):
            if image_path.split(".")[-1] not in supported_extensions:
                continue
            image_name = image_path.split(os.sep)[-1]
            pbar.set_description(f'{image_name}')
            image_path = os.path.join(folder_path, image_name)

            try:
                if os.name == "nt":
                    img = plt.imread(image_path)
                else:
                    img = cv2.imread(image_path)
            except:
                try:
                    img = plt.imread(image_path)
                except:
                    # [image_name, predicts[0][0], probabilities[0][0], predicted_blank]
                    failed_to_open_value = "<FAILED_TO_OPEN>"
                    if WRITE_BAD_TO_OWN_FILE:
                        with open(bad_path, 'a') as f:
                            f.write(f"{image_name},{failed_to_open_value},0,0\n")                    
                    else:
                        final_predicts.append([image_name, failed_to_open_value, 0, 0])
                    continue

            if img is None:
                non_image_value = "<IMAGE_WAS_NONE>"
                if WRITE_BAD_TO_OWN_FILE:
                    with open(bad_path, 'a') as f:
                        f.write(f"{image_name},{non_image_value},0,0\n")
                else:
                    final_predicts.append([image_name, non_image_value, 0, 0])
                continue

            predicted_blank = blank_detector.predictBlank(img)

            img = pp.preprocess(image_path, input_size=input_size)
            x_test = pp.normalization([img])
            predicts, probabilities = model.predict(x_test, ctc_decode=True)
            predicts = [[tokenizer.decode(x) for x in y] for y in predicts]
            final_predicts.append([image_name, predicts[0][0], probabilities[0][0], predicted_blank])
            _ = gc.collect()

            if i != 0 and i % BATCH_SIZE == 0:
                print('RAM Used (GB):', psutil.virtual_memory()[3] / 1000000000)
                if args.csv:
                    with open(out_path, 'a+', newline='') as csvfile:
                        writer = csv.writer(csvfile)
                        writer.writerows(final_predicts)
                        csvfile.close()
                elif args.parquet:
                    fastparquet.write(out_path, final_predicts)
                final_predicts = []

        if args.csv:
            with open(out_path, 'a+', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerows(final_predicts)
                csvfile.close()
        elif args.parquet:
            fastparquet.write(out_path, final_predicts)

        finish_time = time.time()
        total_time = finish_time - start_time
        print("Images Processed: ", len(images))
        print("Total Time elapsed: ", total_time / 60, " minutes")
        print("Time per image: ", total_time / len(images), "seconds")
