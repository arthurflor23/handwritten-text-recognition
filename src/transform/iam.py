"""Transform IAM dataset"""

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

        partition = self._get_partitions()
        partition["train"] = set([i[:-3] for i in partition["train"]])
        partition["valid"] = set([i[:-3] for i in partition["valid"]])
        partition["test"] = set([i[:-3] for i in partition["test"]])

        self._build_paragraphs(partition, "train")
        self._build_paragraphs(partition, "valid")
        self._build_paragraphs(partition, "test")

    def line(self):
        """Make process of line hdf5"""

        partition = self._get_partitions()
        lines = open(os.path.join(self.source, "ascii", "lines.txt")).read().splitlines()
        gt_dict = dict()

        for line in lines:
            if (not line or line[0] == "#"):
                continue

            splited = line.split()
            text = " ".join(splited[-1].replace("|", " ").split())
            gt_dict[splited[0]] = text

        self._build_lines(gt_dict, partition, "train")
        self._build_lines(gt_dict, partition, "valid")
        self._build_lines(gt_dict, partition, "test")

    def _build_lines(self, gt_dict, partition, group):
        """Preprocessing and build line tasks"""

        dt, gt = [], []

        for line in partition[group]:

            if len(gt_dict[line]) > 0:
                split = line.split("-")
                path = os.path.join(split[0], f"{split[0]}-{split[1]}", f"{split[0]}-{split[1]}-{split[2]}.png")
                path = os.path.join(self.source, "lines", path)

                dt.append(path)
                gt.append(gt_dict[line])

        pool = Pool()
        dt = pool.map(partial(self.preproc, img_size=self.input_size, read_first=True), dt)
        gt_sparse = pool.map(partial(self.encode, charset=self.charset, mtl=self.max_text_length), gt)
        pool.close()
        pool.join()

        self._save(group=group, dt=dt, gt=gt, gt_sparse=gt_sparse)

    def _build_paragraphs(self, partition, group):
        """Preprocessing and build paragraph tasks"""

        xml = os.path.join(self.source, "xml")
        form = os.path.join(self.source, "forms")
        pool = Pool()

        dt, gt, gt_sparse = zip(*pool.map(partial(self._extract, xml_path=xml, form_path=form), partition[group]))
        pool.close()
        pool.join()

        self._save(group=group, dt=dt, gt=gt, gt_sparse=gt_sparse)

    def _extract(self, x, xml_path, form_path):
        """Extract paragraphs from the pages"""

        dt, gt_sparse = [], []
        xml = f"{os.path.join(xml_path, x)}.xml"
        png = f"{os.path.join(form_path, x)}.png"

        root = ET.parse(xml).getroot()
        top, bottom = 99999, 0
        text = []

        for page_tag in root:
            for i, line_tag in enumerate(page_tag.iter("line")):
                text_line = " ".join(html.unescape(line_tag.attrib["text"]).split())

                if len(text_line) > 0:
                    text.append(text_line)
                    top = min(top, abs(int(line_tag.attrib["asy"])))
                    bottom = max(bottom, abs(int(line_tag.attrib["dsy"])))

        gt = " ".join(text)
        page = cv2.imread(png, cv2.IMREAD_GRAYSCALE)
        page = page[top:bottom, 0:-1]

        dt.append(self.preproc(img=page, img_size=self.input_size, read_first=False))
        gt_sparse.append(self.encode(text=gt, charset=self.charset, mtl=self.max_text_length))

        return dt, gt, gt_sparse

    def _get_partitions(self):
        """Read the partitions file"""

        pt_path = os.path.join(self.source, "largeWriterIndependentTextLineRecognitionTask")
        partition = {
            "train": open(os.path.join(pt_path, "trainset.txt")).read().splitlines(),
            "valid": open(os.path.join(pt_path, "validationset1.txt")).read().splitlines(),
            "test": open(os.path.join(pt_path, "testset.txt")).read().splitlines()
        }
        return partition

    def _save(self, group, dt, gt, gt_sparse):
        """Save hdf5 file"""

        os.makedirs(os.path.dirname(self.target), exist_ok=True)

        with h5py.File(self.target, "a") as hf:
            hf.create_dataset(f"{group}/dt", data=dt, compression="gzip", compression_opts=9)
            hf.create_dataset(f"{group}/gt_bytes", data=[n.encode() for n in gt], compression="gzip", compression_opts=9)
            hf.create_dataset(f"{group}/gt_sparse", data=gt_sparse, compression="gzip", compression_opts=9)
            print(f"[OK] {group} partition.")
