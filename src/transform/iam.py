"""Transform IAM dataset"""

import cv2
import os


class Dataset():

    def __init__(self, partitions):
        self.partitions = partitions

    def get_partitions(self, source):
        """Process of read partitions data/ground truth"""

        pt_path = os.path.join(source, "largeWriterIndependentTextLineRecognitionTask")
        paths = {
            "train": open(os.path.join(pt_path, "trainset.txt")).read().splitlines(),
            "valid": open(os.path.join(pt_path, "validationset1.txt")).read().splitlines(),
            "test": open(os.path.join(pt_path, "testset.txt")).read().splitlines()
        }

        lines = open(os.path.join(source, "ascii", "lines.txt")).read().splitlines()
        gt_dict = dict()

        for line in lines:
            if (not line or line[0] == "#"):
                continue

            splitted = line.split()
            gt_dict[splitted[0]] = splitted[-1].replace("|", " ")

        dataset = dict()

        for i in self.partitions:
            dataset[i] = {"dt": [], "gt": []}

            for line in paths[i]:
                split = line.split("-")
                img_path = os.path.join(split[0], f"{split[0]}-{split[1]}", f"{split[0]}-{split[1]}-{split[2]}.png")
                img_path = os.path.join(source, "lines", img_path)

                if len(gt_dict[line]) > 5:
                    dataset[i]["dt"].append(cv2.imread(img_path, cv2.IMREAD_GRAYSCALE))
                    dataset[i]["gt"].append(gt_dict[line])

        return dataset
