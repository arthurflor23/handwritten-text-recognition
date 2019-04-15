from glob import glob
import argparse
import shutil
import os


def rename_partitions(dataset_dir):
    partitions_dir = os.path.join(
        dataset_dir, "largeWriterIndependentTextLineRecognitionTask")
    new_partitions_dir = os.path.join(dataset_dir, "partitions")

    if not os.path.exists(new_partitions_dir):
        set_file = os.path.join(partitions_dir, "trainset.txt")
        new_set_file = os.path.join(partitions_dir, "train.txt")
        os.rename(set_file, new_set_file)

        set_file = os.path.join(partitions_dir, "validationset1.txt")
        new_set_file = os.path.join(partitions_dir, "validation.txt")
        os.rename(set_file, new_set_file)

        set_file = os.path.join(partitions_dir, "testset.txt")
        new_set_file = os.path.join(partitions_dir, "test.txt")
        os.rename(set_file, new_set_file)

        os.rename(partitions_dir, new_partitions_dir)


def extract_lines(dataset_dir):
    lines_dir = os.path.join(dataset_dir, "lines")
    paths = next(os.walk(lines_dir))[1]

    for path in paths:
        path = os.path.join(lines_dir, path)
        glob_filter = os.path.join(path, "**", "*.*")
        subpaths = [x for x in glob(glob_filter, recursive=True)]

        for img in subpaths:
            new_img = os.path.join(lines_dir, os.path.basename(img))
            shutil.move(img, new_img)

        shutil.rmtree(path)


def gt_generate(dataset_dir):
    gt_dir = os.path.join(dataset_dir, "gt")

    if not os.path.exists(gt_dir):
        lines = os.path.join(dataset_dir, "ascii", "lines.txt")

        with open(lines) as f:
            content = f.readlines()
            content = [x.strip() for x in content]
            os.mkdir(gt_dir)

            for line in content:
                if (not line or line[0] == "#"):
                    continue

                line_splited = line.strip().split(' ')
                assert len(line_splited) >= 9

                file_name = line_splited[0]
                gt = line_splited[len(line_splited)-1].replace("|", " ")

                f = open(os.path.join(gt_dir, f"{file_name}.txt"), "w+")
                f.write(gt)
                f.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    args = parser.parse_args()

    dataset_dir = os.path.join(args.input_dir, os.path.basename(__file__)[:-3])
    rename_partitions(dataset_dir)
    extract_lines(dataset_dir)
    gt_generate(dataset_dir)


if __name__ == '__main__':
    main()
