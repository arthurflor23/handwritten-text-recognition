"""Transform Saint Gall dataset"""

from glob import glob
import cv2
import os


class Dataset():

    def __init__(self, partitions):
        self.partitions = partitions

    def get_partitions(self, source):
        """Process of read partitions data/ground truth"""

        pt_path = os.path.join(source, "sets")
        paths = {"train": open(os.path.join(pt_path, "train.txt")).read().splitlines(),
                 "valid": open(os.path.join(pt_path, "valid.txt")).read().splitlines(),
                 "test": open(os.path.join(pt_path, "test.txt")).read().splitlines()}

        lines = open(os.path.join(source, "ground_truth", "transcription.txt")).read().splitlines()
        gt_dict = dict()

        for line in lines:
            splitted = line.split()
            splitted[1] = splitted[1].replace("-", "").replace("|", " ")
            gt_dict[splitted[0]] = splitted[1]

        img_path = os.path.join(source, "data", "line_images_normalized")
        dataset = dict()

        for i in self.partitions:
            dataset[i] = {"dt": [], "gt": [], "gt_sparse": []}

            for line in paths[i]:
                glob_filter = os.path.join(img_path, f"{line}*")
                img_list = [x for x in glob(glob_filter, recursive=True)]

                for line in img_list:
                    index = os.path.splitext(os.path.basename(line))[0]

                    if len(gt_dict[index]) > 3:
                        dataset[i]["dt"].append(cv2.imread(line, cv2.IMREAD_GRAYSCALE))
                        dataset[i]["gt"].append(gt_dict[index])

        return dataset
