"""Transform Rimes dataset"""

from multiprocessing import Pool
from functools import partial
import xml.etree.ElementTree as ET
import html
import h5py
import cv2
import os


class Transform():

    def __init__(self,
                 source,
                 target,
                 input_size,
                 charset,
                 max_text_length,
                 preproc, encode):
        self.source = source
        self.target = target
        self.input_size = input_size
        self.charset = charset
        self.max_text_length = max_text_length
        self.preproc = preproc
        self.encode = encode

    def paragraph(self):
        """Make process of paragraph hdf5"""

        partition = self._get_partitions(page=True)
        self._build(self._extract, partition, "train")
        self._build(self._extract, partition, "valid")
        self._build(self._extract, partition, "test")

    def line(self):
        """Make process of line hdf5"""

        partition = self._get_partitions(page=False)
        self._build(self._extract, partition, "train")
        self._build(self._extract, partition, "valid")
        self._build(self._extract, partition, "test")

    def _build(self, func, partition, group):
        """Preprocessing and build line tasks"""

        pool = Pool()
        dt, gt, gt_sparse = zip(*pool.map(partial(func), partition[group]))
        pool.close()
        pool.join()

        self._save(group=group, dt=dt, gt=gt, gt_sparse=gt_sparse)

    def _extract(self, item):
        """Extract lines from the pages"""

        dt = cv2.imread(os.path.join(self.source, item[0]), cv2.IMREAD_GRAYSCALE)
        dt = dt[item[2][0]:item[2][1], item[2][2]:item[2][3]]

        dt = self.preproc(img=dt, img_size=self.input_size, read_first=False)
        gt_sparse = self.encode(text=item[1], charset=self.charset, mtl=self.max_text_length)

        return dt, item[1], gt_sparse

    def _get_partitions(self, page=False):
        """Read the partitions file"""

        def generate(xml, subpath, partition, validation=False):
            xml = ET.parse(os.path.join(self.source, xml)).getroot()
            dt = []

            for page_tag in xml:
                page_path = page_tag.attrib["FileName"]
                text_page = []

                for i, line_tag in enumerate(page_tag.iter("Line")):
                    text_line = " ".join(html.unescape(line_tag.attrib["Value"]).split())

                    if len(text_line) > 0:
                        if page:
                            text_page.append(text_line)
                            continue

                        bound = [abs(int(line_tag.attrib["Top"])), abs(int(line_tag.attrib["Bottom"])),
                                 abs(int(line_tag.attrib["Left"])), abs(int(line_tag.attrib["Right"]))]
                        dt.append([os.path.join(subpath, page_path), text_line, bound])

                if page and len(text_page) > 0:
                    dt.append([os.path.join(subpath, page_path), " ".join(text_page), [0,-1,0,-1]])

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

    def _save(self, group, dt, gt, gt_sparse):
        """Save hdf5 file"""

        os.makedirs(os.path.dirname(self.target), exist_ok=True)

        with h5py.File(self.target, "a") as hf:
            hf.create_dataset(f"{group}/dt", data=dt, compression="gzip", compression_opts=9)
            hf.create_dataset(f"{group}/gt_bytes", data=[n.encode() for n in gt], compression="gzip", compression_opts=9)
            hf.create_dataset(f"{group}/gt_sparse", data=gt_sparse, compression="gzip", compression_opts=9)
            print(f"[OK] {group} partition.")
