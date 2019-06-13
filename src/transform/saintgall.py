"""Transform Saint Gall dataset"""

from multiprocessing import Pool
from functools import partial
from glob import glob
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

    def paragraph(self):
        """Make process of paragraph hdf5"""

        print("\nSaint Gall dataset doesn't support paragraph transformation.\n")

    def line(self):
        """Make process of line hdf5"""

        partition = self._get_partitions()
        gt = os.path.join(self.source, "ground_truth")
        lines = open(os.path.join(gt, "transcription.txt")).read().splitlines()
        gt_dict = dict()

        for line in lines:
            if (not line or line[0] == "#"):
                continue

            splited = line.split()
            text = " ".join(splited[1].replace("-", "").replace("|", " ").split())
            gt_dict[splited[0]] = text

        self._build_lines(gt_dict, partition, "train")
        self._build_lines(gt_dict, partition, "valid")
        self._build_lines(gt_dict, partition, "test")

    def _build_lines(self, gt_dict, partition, group):
        """Preprocessing and build line tasks"""

        path = os.path.join(self.source, "data", "line_images_normalized")
        dt, gt = [], []

        for line in partition[group]:
            glob_filter = os.path.join(path, f"{line}*")
            img_list = [x for x in glob(glob_filter, recursive=True)]

            for img_path in img_list:
                index = os.path.splitext(os.path.basename(img_path))[0]

                if len(gt_dict[index]) > 0:
                    dt.append(img_path)
                    gt.append(gt_dict[index])

        pool = Pool()
        dt = pool.map(partial(self.preproc, img_size=self.input_size, read_first=True), dt)
        gt_sparse = pool.map(partial(self.encode, charset=self.charset, mtl=self.max_text_length), gt)
        pool.close()
        pool.join()

        self._save(group=group, dt=dt, gt=gt, gt_sparse=gt_sparse)

    def _get_partitions(self):
        """Read the partitions file"""

        pt_path = os.path.join(self.source, "sets")
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
