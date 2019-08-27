"""Transform Bentham dataset"""

from multiprocessing import Pool
from functools import partial
import string
import html
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
        self.source = os.path.join(source, "BenthamDatasetR0-GT")
        self.target = target
        self.input_size = input_size
        self.charset = charset
        self.max_text_length = max_text_length
        self.preproc = preproc
        self.encode = encode

    def line(self):
        """Make process of line hdf5"""

        partition = self._get_partitions()
        path = os.path.join(self.source, "Transcriptions")

        gt = os.listdir(path=path)
        gt_dict = dict()

        for index, x in enumerate(gt):
            text = " ".join(open(os.path.join(path, x)).read().splitlines())
            text = html.unescape(text).replace("<gap/>", "")

            for i in string.punctuation.replace("'", ""):
                text = text.replace(i, f" {i} ")

            gt_dict[os.path.splitext(x)[0]] = " ".join(text.split())

        self._build_lines(gt_dict, partition, "train")
        self._build_lines(gt_dict, partition, "valid")
        self._build_lines(gt_dict, partition, "test")

    def _build_lines(self, gt_dict, partition, group):
        """Preprocessing and build line tasks"""

        path = os.path.join(self.source, "Images", "Lines")
        dt, gt = [], []

        for line in partition[group]:
            if len(gt_dict[line]) > 0:
                dt.append(os.path.join(path, f"{line}.png"))
                gt.append(gt_dict[line])

        pool = Pool()
        dt = pool.map(partial(self.preproc, img_size=self.input_size, read_first=True), dt)
        gt_sparse = pool.map(partial(self.encode, charset=self.charset, mtl=self.max_text_length), gt)
        pool.close()
        pool.join()

        self._save(group=group, dt=dt, gt=gt, gt_sparse=gt_sparse)

    def _get_partitions(self):
        """Read the partitions file"""

        pt_path = os.path.join(self.source, "Partitions")
        partition = {
            "train": open(os.path.join(pt_path, "TrainLines.lst")).read().splitlines(),
            "valid": open(os.path.join(pt_path, "ValidationLines.lst")).read().splitlines(),
            "test": open(os.path.join(pt_path, "TestLines.lst")).read().splitlines()
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
