"""Transform Saint Gall dataset"""

from multiprocessing import Pool
from functools import partial
from glob import glob
import h5py
import os


class Transform():

    def __init__(self, env, preproc, encode):
        self.env = env
        self.preproc = preproc
        self.encode = encode

    def paragraph(self):
        """Make process of paragraph hdf5"""

        print("\nSaint Gall dataset doesn't support paragraph transformation.\n")

    def line(self):
        """Make process of line hdf5"""

        partition = self._get_partitions()
        gt = os.path.join(self.env.raw_source, "ground_truth")
        lines = open(os.path.join(gt, "transcription.txt")).read().splitlines()
        gt_dict = dict()

        for line in lines:
            if (not line or line[0] == "#"):
                continue

            splited = line.strip().split(" ")
            assert len(splited) >= 3

            name = splited[0].strip()
            text = splited[1].replace("-", "").replace("|", " ").strip()
            gt_dict[name] = text

        self._build_lines(gt_dict, partition, "train")
        self._build_lines(gt_dict, partition, "valid")
        self._build_lines(gt_dict, partition, "test")

    def _build_lines(self, gt_dict, partition, group):
        """Preprocessing and build line tasks"""

        path = os.path.join(self.env.raw_source, "data", "line_images_normalized")
        dt, gt = [], []

        for line in partition[group]:
            glob_filter = os.path.join(path, f"{line}*")
            img_list = [x for x in glob(glob_filter, recursive=True)]

            for path in img_list:
                index = os.path.splitext(os.path.basename(path))[0]
                text_line = gt_dict[index].strip()

                if len(text_line) > 0:
                    dt.append(path)
                    gt.append(text_line)

        pool = Pool()
        dt = pool.map(partial(self.preproc, img_size=self.env.input_size, read_first=True), dt)
        gt = pool.map(partial(self.encode, charset=self.env.charset, mtl=self.env.max_text_length), gt)
        pool.close()
        pool.join()

        self._save(group=group, dt=dt, gt=gt)
        del dt, gt

    def _get_partitions(self):
        """Read the partitions file"""

        pt_path = os.path.join(self.env.raw_source, "sets")
        partition = {
            "train": open(os.path.join(pt_path, "train.txt")).read().splitlines(),
            "valid": open(os.path.join(pt_path, "valid.txt")).read().splitlines(),
            "test": open(os.path.join(pt_path, "test.txt")).read().splitlines()
        }
        return partition

    def _save(self, group, dt, gt):
        """Save hdf5 file"""

        os.makedirs(os.path.dirname(self.env.source), exist_ok=True)

        with h5py.File(self.env.source, "a") as hf:
            hf.create_dataset(f"{group}/dt", data=dt, compression="gzip", compression_opts=9)
            hf.create_dataset(f"{group}/gt", data=gt, compression="gzip", compression_opts=9)
            print(f"[OK] {group} partition.")
