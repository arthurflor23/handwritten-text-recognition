"""Dataset reader and process"""

import os
import html
import h5py
import string
import random
import pandas as pd
import numpy as np
import multiprocessing
import xml.etree.ElementTree as ET

from glob import glob
from tqdm import tqdm
from data import preproc as pp
from functools import partial


class Dataset():
    """Dataset class to read images and sentences from base (raw files)"""

    def __init__(self, source, name, images=None, labels=None):
        self.source = source
        self.name = name
        self.images = images
        self.labels = labels
        self.dataset = None
        self.partitions = ['train', 'valid', 'test']

    def read_partitions(self):
        """Read images and sentences from dataset"""

        dataset = getattr(self, f"_{self.name}")()

        if not self.dataset:
            self.dataset = self._init_dataset()

        for y in self.partitions:
            self.dataset[y]['dt'] += dataset[y]['dt']
            self.dataset[y]['gt'] += dataset[y]['gt']

    def save_partitions(self, target, image_input_size, max_text_length):
        """Save images and sentences from dataset"""

        os.makedirs(os.path.dirname(target), exist_ok=True)
        total = 0

        with h5py.File(target, "w") as hf:
            for pt in self.partitions:
                self.dataset[pt] = self.check_text(self.dataset[pt], max_text_length)
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

                with multiprocessing.Pool(multiprocessing.cpu_count() - 2) as pool:
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
            dataset[i] = {"dt": [], "gt": []}

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

    def _census(self):

        img_path = self.images
        labels_data = pd.read_csv(self.labels)

        labels_data["filename"] = img_path + "/" + labels_data["filename"].astype(str)
        labels_data["string"] = labels_data["string"].astype(str)
        labels_data = labels_data[labels_data['filename'].apply(os.path.exists)]

        train, valid, test = np.split(labels_data.sample(frac=1, random_state=42),
                                      [int(.6 * len(labels_data)), int(.8 * len(labels_data))])

        dataset = self._init_dataset()

        dataset["train"]["dt"] = train.to_numpy()[:, 0].tolist()
        dataset["train"]["gt"] = train.to_numpy()[:, 1].tolist()

        dataset["valid"]["dt"] = valid.to_numpy()[:, 0].tolist()
        dataset["valid"]["gt"] = valid.to_numpy()[:, 1].tolist()

        dataset["test"]["dt"] = test.to_numpy()[:, 0].tolist()
        dataset["test"]["gt"] = test.to_numpy()[:, 1].tolist()

        return dataset

    @staticmethod
    def check_text(data, max_text_length=128):
        """Checks if the text has more characters instead of punctuation marks"""

        dt = {'gt': list(data['gt']), 'dt': list(data['dt'])}

        for i in reversed(range(len(dt['gt']))):
            text = pp.text_standardize(dt['gt'][i])
            strip_punc = text.strip(string.punctuation).strip()
            no_punc = text.translate(str.maketrans("", "", string.punctuation)).strip()

            length_valid = (len(text) > 0) and (len(text) < max_text_length)
            text_valid = (len(strip_punc) > 0) and (len(no_punc) > 0)

            if (not length_valid) or (not text_valid):
                dt['gt'].pop(i)
                dt['dt'].pop(i)
                continue

        return dt
