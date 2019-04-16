"""Normalize Rimes dataset."""

import argparse
import os
import shutil
import xml.etree.ElementTree as ET
import cv2


def norm_partitions(origin, target):
    """Normalize and create 'partitions' folder."""

    def generate(set_file, new_set_file):
        root = ET.parse(set_file).getroot()

        with open(new_set_file, "w+") as f:
            for page_tag in root:
                basename = os.path.basename(page_tag.attrib["FileName"])
                basename = basename.split(".")[0]

                for i in range(len(page_tag.iter("Line"))):
                    f.write(f"{basename}-{i}\n")
            f.close()

    target_dir = os.path.join(target, "partitions")

    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)
    os.makedirs(target_dir)

    set_file = os.path.join(origin, "training_2011.xml")
    new_set_file = os.path.join(target_dir, "train.txt")
    generate(set_file, new_set_file)

    set_file = os.path.join(origin, "eval_2011_annotated.xml")
    new_set_file = os.path.join(target_dir, "test.txt")
    generate(set_file, new_set_file)


def norm_gt(origin, target):
    """Normalize and create 'gt' folder (Ground Truth)."""

    def generate(set_file):
        root = ET.parse(set_file).getroot()

        for page_tag in root:
            basename = os.path.basename(page_tag.attrib["FileName"])
            basename = basename.split(".")[0]

            for i, line_tag in enumerate(page_tag.iter("Line")):
                new_set_file = os.path.join(target_dir, f"{basename}-{i}.txt")

                with open(new_set_file, "w+") as file:
                    file.write(line_tag.attrib["Value"].strip())
                    file.close()

    target_dir = os.path.join(target, "gt")

    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)
    os.makedirs(target_dir)

    set_file = os.path.join(origin, "training_2011.xml")
    generate(set_file)

    set_file = os.path.join(origin, "eval_2011_annotated.xml")
    generate(set_file)


def norm_lines(origin, target):
    """Normalize and create 'lines' folder."""

    def generate(origin_dir, set_file):
        root = ET.parse(set_file).getroot()

        for page_tag in root:
            basename = os.path.basename(page_tag.attrib["FileName"])
            page = cv2.imread(os.path.join(origin_dir, basename))
            basename = basename.split(".")[0]

            for i, line_tag in enumerate(page_tag.iter("Line")):
                bottom = abs(int(line_tag.attrib["Bottom"]))
                left = abs(int(line_tag.attrib["Left"]))
                right = abs(int(line_tag.attrib["Right"]))
                top = abs(int(line_tag.attrib["Top"]))

                line = page[top:bottom, left:right]
                new_set_file = os.path.join(target_dir, f"{basename}-{i}.png")
                cv2.imwrite(new_set_file, line)

    target_dir = os.path.join(target, "lines")

    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)
    os.makedirs(target_dir)

    origin_dir = os.path.join(origin, "training_2011", "images")
    set_file = os.path.join(origin, "training_2011.xml")
    generate(origin_dir, set_file)

    origin_dir = os.path.join(origin, "eval_2011", "images")
    set_file = os.path.join(origin, "eval_2011_annotated.xml")
    generate(origin_dir, set_file)


def main():
    """Get the input parameter and call normalization methods."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    args = parser.parse_args()

    src = args.data_dir
    src_backup = f"{src}_backup"

    if not os.path.exists(src_backup):
        os.rename(src, src_backup)

    norm_partitions(src_backup, src)
    norm_gt(src_backup, src)
    norm_lines(src_backup, src)


if __name__ == '__main__':
    main()
