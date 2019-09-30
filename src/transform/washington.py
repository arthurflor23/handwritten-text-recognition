"""Transform Washington dataset"""

import cv2
import os


class Dataset():

    def __init__(self, partitions):
        self.partitions = partitions

    def get_partitions(self, source):
        """Process of read partitions data/ground truth"""

        pt_path = os.path.join(source, "sets", "cv1")
        paths = {"train": open(os.path.join(pt_path, "train.txt")).read().splitlines(),
                 "valid": open(os.path.join(pt_path, "valid.txt")).read().splitlines(),
                 "test": open(os.path.join(pt_path, "test.txt")).read().splitlines()}

        lines = open(os.path.join(source, "ground_truth", "transcription.txt")).read().splitlines()
        gt_dict = dict()

        for line in lines:
            splitted = line.split()
            splitted[1] = splitted[1].replace("-", "").replace("|", " ")
            splitted[1] = splitted[1].replace("s_pt", ".").replace("s_cm", ",")
            splitted[1] = splitted[1].replace("s_mi", "-").replace("s_qo", ":")
            splitted[1] = splitted[1].replace("s_sq", ";").replace("s_et", "V")
            splitted[1] = splitted[1].replace("s_bl", "(").replace("s_br", ")")
            splitted[1] = splitted[1].replace("s_qt", "'").replace("s_", "")
            gt_dict[splitted[0]] = splitted[1]

        img_path = os.path.join(source, "data", "line_images_normalized")
        dataset = dict()

        for i in self.partitions:
            dataset[i] = {"dt": [], "gt_bytes": [], "gt_sparse": []}

            for line in paths[i]:
                if len(gt_dict[line]) > 3:
                    dataset[i]["dt"].append(cv2.imread(os.path.join(img_path, f"{line}.png"), cv2.IMREAD_GRAYSCALE))
                    dataset[i]["gt_sparse"].append(gt_dict[line])

        return dataset
