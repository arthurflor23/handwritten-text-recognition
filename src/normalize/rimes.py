"""Normalize Rimes dataset."""

import argparse
import sys
import os
import shutil
import xml.etree.ElementTree as ET
import cv2

try:
    from settings.environment import Environment
except ImportError:
    sys.path[0] = os.path.join(sys.path[0], "..")
    from settings.environment import Environment


def norm_partitions(origin, env):
    """Normalize and create 'partitions' folder."""

    if os.path.exists(env.partitions_dir):
        shutil.rmtree(env.partitions_dir)
    os.makedirs(env.partitions_dir)

    def generate(set_file, train_file, validation_file=None):
        root = ET.parse(set_file).getroot()
        lines = []

        with open(train_file, "w") as train_f:
            for page_tag in root:
                basename = os.path.basename(page_tag.attrib["FileName"])
                basename = basename.split(".")[0]

                for i, _ in enumerate(page_tag.iter("Line")):
                    lines.append(f"{basename}-{i}\n")

            if validation_file:
                index = int(len(lines) * 0.9)
                train = lines[:index]
                validation = lines[index:]

                with open(validation_file, "w") as validation_f:
                    train_f.write(''.join(train))
                    validation_f.write(''.join(validation))
            else:
                train_f.write(''.join(lines))

    set_file = os.path.join(origin, "training_2011.xml")
    generate(set_file, env.train_file, env.validation_file)

    set_file = os.path.join(origin, "eval_2011_annotated.xml")
    generate(set_file, env.test_file)


def norm_gt(origin, env):
    """Normalize and create 'gt' folder (Ground Truth)."""

    if os.path.exists(env.gt_dir):
        shutil.rmtree(env.gt_dir)
    os.makedirs(env.gt_dir)

    def generate(set_file):
        root = ET.parse(set_file).getroot()

        for page_tag in root:
            basename = os.path.basename(page_tag.attrib["FileName"])
            basename = basename.split(".")[0]

            for i, line_tag in enumerate(page_tag.iter("Line")):
                new_set_file = os.path.join(env.gt_dir, f"{basename}-{i}.txt")

                with open(new_set_file, "w+") as file:
                    file.write(line_tag.attrib["Value"].strip())

    generate(os.path.join(origin, "training_2011.xml"))
    generate(os.path.join(origin, "eval_2011_annotated.xml"))


def norm_data(origin, env):
    """Normalize and create 'lines' folder."""

    if os.path.exists(env.data_dir):
        shutil.rmtree(env.data_dir)
    os.makedirs(env.data_dir)

    def generate(origin_dir, root):
        for page_tag in root:
            basename = os.path.basename(page_tag.attrib["FileName"])
            pagename = basename.split(".")[0]
            page = cv2.imread(os.path.join(origin_dir, basename))

            for i, line_tag in enumerate(page_tag.iter("Line")):
                bottom = abs(int(line_tag.attrib["Bottom"]))
                left = abs(int(line_tag.attrib["Left"]))
                right = abs(int(line_tag.attrib["Right"]))
                top = abs(int(line_tag.attrib["Top"]))

                line = page[top:bottom, left:right]
                line_dir = os.path.join(
                    env.data_dir, f"{pagename}-{i}.{env.extension}")

                cv2.imwrite(line_dir, line)

    origin_dir = os.path.join(origin, "training_2011", "images")
    set_file = os.path.join(origin, "training_2011.xml")
    root = ET.parse(set_file).getroot()
    generate(origin_dir, root)

    origin_dir = os.path.join(origin, "eval_2011", "images")
    set_file = os.path.join(origin, "eval_2011_annotated.xml")
    root = ET.parse(set_file).getroot()
    generate(origin_dir, root)


def main():
    """Get the input parameter and call normalization methods."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, required=True)
    args = parser.parse_args()

    env = Environment(args.dataset_dir)
    src_backup = f"{args.dataset_dir}_backup"

    if not os.path.exists(src_backup):
        os.rename(args.dataset_dir, src_backup)

    norm_partitions(src_backup, env)
    norm_gt(src_backup, env)
    norm_data(src_backup, env)


if __name__ == '__main__':
    main()
