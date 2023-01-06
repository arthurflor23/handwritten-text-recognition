import argparse
import time
import os
import string
import tarfile

from zipfile import Zipfile

from data import preproc as pp
from data.generator import Tokenizer

from network.model import HTRModel

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, required=True)
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--arch", type=str, default="flor")

    parser.add_argument("--archive", type=bool, default=False)
    parser.add_argument("--csv", type=str, default="")
    parser.add_argument("--parquet", type=str, default="")

    args = parser.parse_args()

    source_path = args.source
    weights_path = args.weights

    input_size = (1024, 128, 1)
    max_text_length = 50
    charset_base = string.printable[:95]

    start_time = time.time()

    if args.source:

        if args.archive:
            # Slurm allocates storage space for a job that isn't subject to the same file size and count restrictions
            # as the rest of the storage, so we will unpack the archive onto the tmp folder created.
            archive_path = args.source
            archive_file_type = archive_path.split(".", 1)[1]
            folder_path = "tmp"
            if archive_file_type == "zip":
                Zipfile(archive_path).extractall(folder_path)

            elif archive_file_type == "tar" or archive_file_type == "tar.gz":
                tarfile.open(archive_path).extractall(folder_path)

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

        images = [os.path.join(folder_path, x) for x in os.listdir(folder_path) if x.split(".")[-1] == "jpg" or x.split(".")[-1] == "jp2"]
        for image_name in images:
            img = pp.preprocess(image_name, input_size=input_size)
            x_test = pp.normalization([img])

            predicts, probabilities = model.predict(x_test, ctc_decode=True)
            predicts = [[tokenizer.decode(x) for x in y] for y in predicts]


        finish_time = time.time()
        total_time = finish_time - start_time
        print("Images Processed: ", len(images))
        print("Total Time elapsed: ", total_time / 60, " minutes")
        print("Time per image: ", total_time / len(images), "seconds")
