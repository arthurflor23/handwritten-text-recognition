"""Transform Rimes dataset"""

import xml.etree.ElementTree as ET
import numpy as np
import html
import cv2
import os


class Dataset():

    def __init__(self, partitions):
        self.partitions = partitions

    def get_partitions(self, source):
        """Process of read partitions data/ground truth"""

        paths = self._get_partitions(source)
        dataset = dict()

        for i in self.partitions:
            dataset[i] = {"dt": [], "gt_bytes": [], "gt_sparse": []}

            for item in paths[i]:
                img = cv2.imread(os.path.join(source, item[0]), cv2.IMREAD_GRAYSCALE)
                img = np.array(img[item[2][0]:item[2][1], item[2][2]:item[2][3]], dtype=np.uint8)

                dataset[i]["dt"].append(img)
                dataset[i]["gt_sparse"].append(item[1])

        return dataset

    def _get_partitions(self, source):
        """Read the partitions file"""

        def generate(xml, subpath, partition, validation=False):
            xml = ET.parse(os.path.join(source, xml)).getroot()
            dt = []

            for page_tag in xml:
                page_path = page_tag.attrib["FileName"]

                for i, line_tag in enumerate(page_tag.iter("Line")):
                    text = html.unescape(line_tag.attrib["Value"])
                    text = " ".join(text.split())

                    if len(text) > 3:
                        bound = [abs(int(line_tag.attrib["Top"])), abs(int(line_tag.attrib["Bottom"])),
                                 abs(int(line_tag.attrib["Left"])), abs(int(line_tag.attrib["Right"]))]
                        dt.append([os.path.join(subpath, page_path), text, bound])

            if validation:
                index = int(len(dt) * 0.9)
                partition["valid"] = dt[index:]
                partition["train"] = dt[:index]
            else:
                partition["test"] = dt

        partition = dict()
        generate("training_2011.xml", "training_2011", partition, validation=True)
        generate("eval_2011_annotated.xml", "eval_2011", partition, validation=False)

        return partition
