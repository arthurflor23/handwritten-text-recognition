"""Dataset reader and process"""

import os
import html
import h5py
import random
import numpy as np
import multiprocessing
import xml.etree.ElementTree as ET

from glob import glob
from tqdm import tqdm
from data import preproc as pp
from functools import partial


class Dataset():
    """Dataset class to read images and sentences from base (raw files)"""

    def __init__(self, source, name):
        self.source = source
        self.name = name
        self.dataset = None
        self.partitions = ['train', 'valid', 'test']

    def read_partitions(self):
        """Read images and sentences from dataset"""

        dataset = getattr(self, f"_{self.name}")()

        if not self.dataset:
            self.dataset = self._init_dataset()

        for y in self.partitions:
            self.dataset[y]['path'] += dataset[y]['path']
            self.dataset[y]['dt'] += dataset[y]['dt']
            self.dataset[y]['gt'] += dataset[y]['gt']

    def save_partitions(self, target, image_input_size, max_text_length):
        """Save images and sentences from dataset"""

        os.makedirs(os.path.dirname(target), exist_ok=True)
        total = 0

        with h5py.File(target, "w") as hf:
            for pt in self.partitions:
                size = (len(self.dataset[pt]['dt']),) + image_input_size[:2]
                total += size[0]

                dummy_image = np.zeros(size, dtype=np.uint8)
                dummy_sentence = [("c" * max_text_length).encode()] * size[0]

                hf.create_dataset(f"{pt}/dt", data=dummy_image, compression="gzip", compression_opts=9)
                hf.create_dataset(f"{pt}/gt", data=dummy_sentence, compression="gzip", compression_opts=9)

        pbar = tqdm(total=total)
        batch_size = 1024

        for pt in self.partitions:
            for batch in range(0, len(self.dataset[pt]['gt']), batch_size):
                images = []

                with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
                    r = pool.map(partial(pp.preprocess, input_size=image_input_size),
                                 self.dataset[pt]['dt'][batch:batch + batch_size])
                    images.append(r)
                    pool.close()
                    pool.join()

                with h5py.File(target, "a") as hf:
                    hf[f"{pt}/dt"][batch:batch + batch_size] = images
                    hf[f"{pt}/gt"][batch:batch + batch_size] = [s.encode() for s in self.dataset[pt]
                                                                ['gt'][batch:batch + batch_size]]
                    pbar.update(batch_size)

    def _init_dataset(self):
        dataset = dict()

        for i in self.partitions:
            dataset[i] = {"dt": [], "gt": [], "path": []}

        return dataset

    def _shuffle(self, *ls):
        random.seed(42)

        if len(ls) == 1:
            li = list(*ls)
            random.shuffle(li)
            return li

        li = list(zip(*ls))
        random.shuffle(li)
        return zip(*li)

    def _bressay(self):

        dataset = self._init_dataset()
        img_path = os.path.join(self.source, "data", "lines")

        paths = {"train": open(os.path.join(self.source, "sets", "training.txt")).read().splitlines(),
                 "valid": open(os.path.join(self.source, "sets", "validation.txt")).read().splitlines(),
                 "test": open(os.path.join(self.source, "sets", "test.txt")).read().splitlines()}

        for pt in self.partitions:
            for item in paths[pt]:
                glob_filter = os.path.join(img_path, item, "**", "*.png")
                data_list = [x for x in glob(glob_filter, recursive=True)]

                for image_path in data_list:
                    text_path = image_path.replace('.png', '.txt')

                    if os.path.isfile(image_path) and os.path.isfile(text_path):
                        dataset[pt]['path'].append(text_path)
                        dataset[pt]['dt'].append(image_path)
                        dataset[pt]['gt'].append(' '.join(open(text_path).read().splitlines()))

        return dataset

    def _hdsr14_car_a(self):
        """ICFHR 2014 Competition on Handwritten Digit String Recognition in Challenging Datasets dataset reader"""

        dataset = self._init_dataset()
        partition = self._read_orand_partitions(os.path.join(self.source, "ORAND-CAR-2014"), 'a')

        for pt in self.partitions:
            for item in partition[pt]:
                text = " ".join(list(item[1]))
                dataset[pt]['dt'].append(item[0])
                dataset[pt]['gt'].append(text)

        return dataset

    def _hdsr14_car_b(self):
        """ICFHR 2014 Competition on Handwritten Digit String Recognition in Challenging Datasets dataset reader"""

        dataset = self._init_dataset()
        partition = self._read_orand_partitions(os.path.join(self.source, "ORAND-CAR-2014"), 'b')

        for pt in self.partitions:
            for item in partition[pt]:
                text = " ".join(list(item[1]))
                dataset[pt]['dt'].append(item[0])
                dataset[pt]['gt'].append(text)

        return dataset

    def _read_orand_partitions(self, basedir, type_f):
        """ICFHR 2014 Competition on Handwritten Digit String Recognition in Challenging Datasets dataset reader"""

        partition = {"train": [], "valid": [], "test": []}
        folder = f"CAR-{type_f.upper()}"

        for i in ['train', 'test']:
            img_path = os.path.join(basedir, folder, f"{type_f.lower()}_{i}_images")
            txt_file = os.path.join(basedir, folder, f"{type_f.lower()}_{i}_gt.txt")

            with open(txt_file) as f:
                lines = [line.replace("\n", "").split("\t") for line in f]
                lines = [[os.path.join(img_path, x[0]), x[1]] for x in lines]

            partition[i] = lines

        sub_partition = int(len(partition['train']) * 0.1)
        partition['valid'] = partition['train'][:sub_partition]
        partition['train'] = partition['train'][sub_partition:]

        return partition

    def _hdsr14_cvl(self):
        """ICFHR 2014 Competition on Handwritten Digit String Recognition in Challenging Datasets dataset reader"""

        dataset = self._init_dataset()
        partition = {"train": [], "valid": [], "test": []}

        glob_filter = os.path.join(self.source, "cvl-strings", "**", "*.png")
        train_list = [x for x in glob(glob_filter, recursive=True)]

        glob_filter = os.path.join(self.source, "cvl-strings-eval", "**", "*.png")
        test_list = [x for x in glob(glob_filter, recursive=True)]

        sub_partition = int(len(train_list) * 0.1)
        partition['valid'].extend(train_list[:sub_partition])
        partition['train'].extend(train_list[sub_partition:])
        partition['test'].extend(test_list[:])

        for pt in self.partitions:
            for item in partition[pt]:
                text = " ".join(list(os.path.basename(item).split("-")[0]))
                dataset[pt]['dt'].append(item)
                dataset[pt]['gt'].append(text)

        return dataset

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
        dataset = self._init_dataset()

        for i in self.partitions:
            for line in paths[i]:
                dataset[i]['dt'].append(os.path.join(img_path, f"{line}.png"))
                dataset[i]['gt'].append(gt_dict[line])

        return dataset

    def _iam(self):
        """IAM dataset reader"""

        pt_path = os.path.join(self.source, "largeWriterIndependentTextLineRecognitionTask")
        paths = {"train": open(os.path.join(pt_path, "trainset.txt")).read().splitlines(),
                 "valid": open(os.path.join(pt_path, "validationset1.txt")).read().splitlines() +
                 open(os.path.join(pt_path, "validationset2.txt")).read().splitlines(),
                 "test": open(os.path.join(pt_path, "testset.txt")).read().splitlines()}

        lines = open(os.path.join(self.source, "ascii", "lines.txt")).read().splitlines()
        dataset = self._init_dataset()
        gt_dict = dict()

        for line in lines:
            if (not line or line[0] == "#"):
                continue

            split = line.split()
            gt_dict[split[0]] = " ".join(split[8::]).replace("|", " ")

        for i in self.partitions:
            for line in paths[i]:
                try:
                    split = line.split("-")
                    folder = f"{split[0]}-{split[1]}"

                    img_file = f"{split[0]}-{split[1]}-{split[2]}.png"
                    img_path = os.path.join(self.source, "lines", split[0], folder, img_file)

                    dataset[i]['gt'].append(gt_dict[line])
                    dataset[i]['dt'].append(img_path)
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

                    bound = [abs(int(line_tag.attrib['Top'])), abs(int(line_tag.attrib['Bottom'])),
                             abs(int(line_tag.attrib['Left'])), abs(int(line_tag.attrib['Right']))]
                    dt.append([os.path.join(subpath, page_path), text, bound])

            if validation:
                index = int(len(dt) * 0.9)
                paths['valid'] = dt[index:]
                paths['train'] = dt[:index]
            else:
                paths['test'] = dt

        dataset = self._init_dataset()
        paths = dict()

        generate("training_2011.xml", "training_2011", paths, validation=True)
        generate("eval_2011_annotated.xml", "eval_2011", paths, validation=False)

        for i in self.partitions:
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
            split = line.split()
            split[1] = split[1].replace("-", "").replace("|", " ")
            gt_dict[split[0]] = split[1]

        img_path = os.path.join(self.source, "data", "line_images_normalized")
        dataset = self._init_dataset()

        for i in self.partitions:
            for line in paths[i]:
                glob_filter = os.path.join(img_path, f"{line}*")
                img_list = [x for x in glob(glob_filter, recursive=True)]

                for line in img_list:
                    line = os.path.splitext(os.path.basename(line))[0]
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
            split = line.split()
            split[1] = split[1].replace("-", "").replace("|", " ")
            split[1] = split[1].replace("s_pt", ".").replace("s_cm", ",")
            split[1] = split[1].replace("s_mi", "-").replace("s_qo", ":")
            split[1] = split[1].replace("s_sq", ";").replace("s_et", "V")
            split[1] = split[1].replace("s_bl", "(").replace("s_br", ")")
            split[1] = split[1].replace("s_qt", "'").replace("s_GW", "G.W.")
            split[1] = split[1].replace("s_", "")
            gt_dict[split[0]] = split[1]

        img_path = os.path.join(self.source, "data", "line_images_normalized")
        dataset = self._init_dataset()

        for i in self.partitions:
            for line in paths[i]:
                dataset[i]['dt'].append(os.path.join(img_path, f"{line}.png"))
                dataset[i]['gt'].append(gt_dict[line])

        return dataset
