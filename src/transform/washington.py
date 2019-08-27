"""Transform Washington dataset"""

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
        gt = os.path.join(self.source, "ground_truth")
        lines = open(os.path.join(gt, "transcription.txt")).read().splitlines()
        gt_dict = dict()

        for line in lines:
            splitted = line.split()
            splitted[1] = splitted[1].replace("-", "").replace("|", " ")
            splitted[1] = splitted[1].replace("s_pt", ".").replace("s_cm", ",")
            splitted[1] = splitted[1].replace("s_mi", "-").replace("s_qo", ":")
            splitted[1] = splitted[1].replace("s_sq", ";").replace("s_et", "V")
            splitted[1] = splitted[1].replace("s_bl", "(").replace("s_br", ")")
            splitted[1] = splitted[1].replace("s_qt", "'").replace("s_", "")
            text = " ".join(splitted[1].split())

            for i in string.punctuation.replace("'", ""):
                text = text.replace(i, f" {i} ")

            gt_dict[splitted[0]] = " ".join(text.split())

        self._build_lines(gt_dict, partition, "train")
        self._build_lines(gt_dict, partition, "valid")
        self._build_lines(gt_dict, partition, "test")

    def _build_lines(self, gt_dict, partition, group):
        """Preprocessing and build line tasks"""

        path = os.path.join(self.source, "data", "line_images_normalized")
        dt, gt = [], []

        for line in partition[group]:
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

        pt_path = os.path.join(self.source, "sets", "cv1")
        partition = {
            "train": open(os.path.join(pt_path, "train.txt")).read().splitlines(),
            "valid": open(os.path.join(pt_path, "valid.txt")).read().splitlines(),
            "test": open(os.path.join(pt_path, "test.txt")).read().splitlines()
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
