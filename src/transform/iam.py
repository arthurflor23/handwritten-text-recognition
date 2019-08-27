"""Transform IAM dataset"""

from multiprocessing import Pool
from functools import partial
import string
import h5py
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

    def line(self):
        """Make process of line hdf5"""

        partition = self._get_partitions()
        lines = open(os.path.join(self.source, "ascii", "lines.txt")).read().splitlines()
        gt_dict = dict()

        for line in lines:
            if (not line or line[0] == "#"):
                continue

            splitted = line.split()
            text = splitted[-1].replace("|", " ")

            for i in string.punctuation.replace("'", ""):
                text = text.replace(i, f" {i} ")

            gt_dict[splitted[0]] = " ".join(text.split())

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
