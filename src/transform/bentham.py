"""Transform Bentham dataset"""

import html
import cv2
import os


class Dataset():

    def __init__(self, partitions):
        self.partitions = partitions

    def get_partitions(self, source):
        """Process of read partitions data/ground truth"""

        source = os.path.join(source, "BenthamDatasetR0-GT")
        pt_path = os.path.join(source, "Partitions")
        paths = {"train": open(os.path.join(pt_path, "TrainLines.lst")).read().splitlines(),
                 "valid": open(os.path.join(pt_path, "ValidationLines.lst")).read().splitlines(),
                 "test": open(os.path.join(pt_path, "TestLines.lst")).read().splitlines()}

        transcriptions = os.path.join(source, "Transcriptions")
        gt = os.listdir(transcriptions)
        gt_dict = dict()

        for index, x in enumerate(gt):
            text = " ".join(open(os.path.join(transcriptions, x)).read().splitlines())
            text = html.unescape(text).replace("<gap/>", "")
            gt_dict[os.path.splitext(x)[0]] = " ".join(text.split())

        img_path = os.path.join(source, "Images", "Lines")
        dataset = dict()

        for i in self.partitions:
            dataset[i] = {"dt": [], "gt_bytes": [], "gt_sparse": []}

            for line in paths[i]:
                if len(gt_dict[line]) > 3:
                    dataset[i]["dt"].append(cv2.imread(os.path.join(img_path, f"{line}.png"), cv2.IMREAD_GRAYSCALE))
                    dataset[i]["gt_sparse"].append(gt_dict[line])

        return dataset
