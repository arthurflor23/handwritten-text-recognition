"""Dataset reader and process"""

import os
import html
import xml.etree.ElementTree as ET

from data import preproc as pp
from functools import partial
from glob import glob
from multiprocessing import Pool


class Dataset():
    """Dataset class to read images and sentences from base (raw files)"""

    def __init__(self, source, name):
        self.source = source
        self.name = name
        self.dataset = None
        self.partitions = ["train", "valid", "test"]

    def read_partitions(self):
        """Read images and sentences from dataset"""

        self.dataset = getattr(self, f"_{self.name}")()

    def preprocess_partitions(self, image_input_size):
        """Preprocess images and sentences from partitions"""

        for i in self.partitions:
            self.dataset[i]['gt'] = [pp.text_standardize(x).encode() for x in self.dataset[i]['gt']]

            pool = Pool()
            self.dataset[i]['dt'] = pool.map(partial(pp.preproc, img_size=image_input_size), self.dataset[i]['dt'])
            pool.close()
            pool.join()

    def _bentham(self):
        """Bentham dataset reader"""

        source = os.path.join(self.source, "BenthamDatasetR0-GT")
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
            dataset[i] = {"dt": [], "gt": []}

            for line in paths[i]:
                if len(gt_dict[line]) > 5:
                    dataset[i]['dt'].append(os.path.join(img_path, f"{line}.png"))
                    dataset[i]['gt'].append(gt_dict[line])

        return dataset

    def _iam(self):
        """IAM dataset reader"""

        pt_path = os.path.join(self.source, "largeWriterIndependentTextLineRecognitionTask")
        paths = {"train": open(os.path.join(pt_path, "trainset.txt")).read().splitlines(),
                 "valid": (open(os.path.join(pt_path, "validationset1.txt")).read().splitlines() +
                           open(os.path.join(pt_path, "validationset2.txt")).read().splitlines()),
                 "test": open(os.path.join(pt_path, "testset.txt")).read().splitlines()}

        lines = open(os.path.join(self.source, "ascii", "lines.txt")).read().splitlines()
        gt_dict = dict()

        for line in lines:
            if (not line or line[0] == "#"):
                continue

            splitted = line.split()

            if splitted[1] == "ok":
                gt_dict[splitted[0]] = " ".join(splitted[8::]).replace("|", " ")

        dataset = dict()

        for i in self.partitions:
            dataset[i] = {"dt": [], "gt": []}

            for line in paths[i]:
                try:
                    if len(gt_dict[line]) < 5:
                        continue

                    split = line.split("-")

                    folder = f"{split[0]}-{split[1]}"
                    img_file = f"{split[0]}-{split[1]}-{split[2]}.png"
                    img_path = os.path.join(self.source, "lines", split[0], folder, img_file)

                    dataset[i]['dt'].append(img_path)
                    dataset[i]['gt'].append(gt_dict[line])
                except KeyError:
                    pass

        return dataset

    def _rimes(self):
        """Rimes dataset reader"""

        def generate(xml, subpath, paths, validation=False):
            xml = ET.parse(os.path.join(self.source, xml)).getroot()
            dt = []

            for page_tag in xml:
                page_path = page_tag.attrib['FileName']

                for i, line_tag in enumerate(page_tag.iter("Line")):
                    text = html.unescape(line_tag.attrib['Value'])
                    text = " ".join(text.split())

                    if len(text) > 5:
                        bound = [abs(int(line_tag.attrib['Top'])), abs(int(line_tag.attrib['Bottom'])),
                                 abs(int(line_tag.attrib['Left'])), abs(int(line_tag.attrib['Right']))]
                        dt.append([os.path.join(subpath, page_path), text, bound])

            if validation:
                index = int(len(dt) * 0.9)
                paths['valid'] = dt[index:]
                paths['train'] = dt[:index]
            else:
                paths['test'] = dt

        dataset, paths = dict(), dict()
        generate("training_2011.xml", "training_2011", paths, validation=True)
        generate("eval_2011_annotated.xml", "eval_2011", paths, validation=False)

        for i in self.partitions:
            dataset[i] = {"dt": [], "gt": []}

            for item in paths[i]:
                boundbox = [item[2][0], item[2][1], item[2][2], item[2][3]]
                dataset[i]['dt'].append((os.path.join(self.source, item[0]), boundbox))
                dataset[i]['gt'].append(item[1])

        return dataset

    def _saintgall(self):
        """Saint Gall dataset reader"""

        pt_path = os.path.join(self.source, "sets")
        paths = {"train": open(os.path.join(pt_path, "train.txt")).read().splitlines(),
                 "valid": open(os.path.join(pt_path, "valid.txt")).read().splitlines(),
                 "test": open(os.path.join(pt_path, "test.txt")).read().splitlines()}

        lines = open(os.path.join(self.source, "ground_truth", "transcription.txt")).read().splitlines()
        gt_dict = dict()

        for line in lines:
            splitted = line.split()
            splitted[1] = splitted[1].replace("-", "").replace("|", " ")
            gt_dict[splitted[0]] = splitted[1]

        img_path = os.path.join(self.source, "data", "line_images_normalized")
        dataset = dict()

        for i in self.partitions:
            dataset[i] = {"dt": [], "gt": []}

            for line in paths[i]:
                glob_filter = os.path.join(img_path, f"{line}*")
                img_list = [x for x in glob(glob_filter, recursive=True)]

                for line in img_list:
                    line = os.path.splitext(os.path.basename(line))[0]

                    if len(gt_dict[line]) > 5:
                        dataset[i]['dt'].append(os.path.join(img_path, f"{line}.png"))
                        dataset[i]['gt'].append(gt_dict[line])

        return dataset

    def _washington(self):
        """Washington dataset reader"""

        pt_path = os.path.join(self.source, "sets", "cv1")
        paths = {"train": open(os.path.join(pt_path, "train.txt")).read().splitlines(),
                 "valid": open(os.path.join(pt_path, "valid.txt")).read().splitlines(),
                 "test": open(os.path.join(pt_path, "test.txt")).read().splitlines()}

        lines = open(os.path.join(self.source, "ground_truth", "transcription.txt")).read().splitlines()
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

        img_path = os.path.join(self.source, "data", "line_images_normalized")
        dataset = dict()

        for i in self.partitions:
            dataset[i] = {"dt": [], "gt": []}

            for line in paths[i]:
                if len(gt_dict[line]) > 5:
                    dataset[i]['dt'].append(os.path.join(img_path, f"{line}.png"))
                    dataset[i]['gt'].append(gt_dict[line])

        return dataset
