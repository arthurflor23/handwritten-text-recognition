"""Normalize Rimes dataset"""

import os
import shutil
import xml.etree.ElementTree as ET
import cv2


def partitions(origin, path):
    """Normalize and create 'partitions' folder"""

    if os.path.exists(path.partitions):
        shutil.rmtree(path.partitions)
    os.makedirs(path.partitions)

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
    generate(set_file, path.train_file, path.validation_file)

    set_file = os.path.join(origin, "eval_2011_annotated.xml")
    generate(set_file, path.test_file)


def ground_truth(origin, path):
    """Normalize and create 'gt' folder (Ground Truth)"""

    if os.path.exists(path.ground_truth):
        shutil.rmtree(path.ground_truth)
    os.makedirs(path.ground_truth)

    def generate(set_file):
        root = ET.parse(set_file).getroot()

        for page_tag in root:
            basename = os.path.basename(page_tag.attrib["FileName"])
            basename = basename.split(".")[0]

            for i, line_tag in enumerate(page_tag.iter("Line")):
                new_set_file = os.path.join(path.ground_truth, f"{basename}-{i}.txt")

                with open(new_set_file, "w+") as file:
                    file.write(line_tag.attrib["Value"].strip())

    generate(os.path.join(origin, "training_2011.xml"))
    generate(os.path.join(origin, "eval_2011_annotated.xml"))


def data(origin, path):
    """Normalize and create 'lines' folder"""

    if os.path.exists(path.data):
        shutil.rmtree(path.data)
    os.makedirs(path.data)

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
                    path.data, f"{pagename}-{i}.png")

                cv2.imwrite(line_dir, line)

    origin_dir = os.path.join(origin, "training_2011", "images")
    set_file = os.path.join(origin, "training_2011.xml")
    root = ET.parse(set_file).getroot()
    generate(origin_dir, root)

    origin_dir = os.path.join(origin, "eval_2011", "images")
    set_file = os.path.join(origin, "eval_2011_annotated.xml")
    root = ET.parse(set_file).getroot()
    generate(origin_dir, root)
