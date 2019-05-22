"""Transform Bentham dataset"""

from multiprocessing import Pool
from functools import partial
import html
import h5py
import os


class Transform():

    def __init__(self, env, preproc, encode):
        self.env = env
        self.env.raw_source = os.path.join(self.env.raw_source, "BenthamDatasetR0-GT")

        self.preproc = preproc
        self.encode = encode

    def paragraph(self):
        """Make process of paragraph hdf5"""

        print("\nBentham dataset doesn't support paragraph transformation.\n")

    def line(self):
        """Make process of line hdf5"""

        partition = self._get_partitions()
        path = os.path.join(self.env.raw_source, "Transcriptions")

        gt = os.listdir(path=path)
        gt_dict = dict()

        for index, x in enumerate(gt):
            text = " ".join(open(os.path.join(path, x)).read().splitlines())
            text = html.unescape(" ".join(text.split())).replace("<gap/>", "")
            gt_dict[os.path.splitext(x)[0]] = text

        self._build_lines(gt_dict, partition, "train")
        self._build_lines(gt_dict, partition, "valid")
        self._build_lines(gt_dict, partition, "test")

    def _build_lines(self, gt_dict, partition, group):
        """Preprocessing and build line tasks"""

        path = os.path.join(self.env.raw_source, "Images", "Lines")
        dt, gt = [], []

        for line in partition[group]:

            if len(gt_dict[line]) > 0:
                dt.append(os.path.join(path, f"{line}.png"))
                gt.append(gt_dict[line])

        pool = Pool()
        dt = pool.map(partial(self.preproc, img_size=self.env.input_size, read_first=True), dt)
        gt = pool.map(partial(self.encode, charset=self.env.charset, mtl=self.env.max_text_length), gt)
        pool.close()
        pool.join()

        self._save(group=group, dt=dt, gt=gt)

    def _get_partitions(self):
        """Read the partitions file"""

        pt_path = os.path.join(self.env.raw_source, "Partitions")
        partition = {
            "train": open(os.path.join(pt_path, "TrainLines.lst")).read().splitlines(),
            "valid": open(os.path.join(pt_path, "ValidationLines.lst")).read().splitlines(),
            "test": open(os.path.join(pt_path, "TestLines.lst")).read().splitlines()
        }
        return partition

    def _save(self, group, dt, gt):
        """Save hdf5 file"""

        os.makedirs(os.path.dirname(self.env.source), exist_ok=True)

        with h5py.File(self.env.source, "a") as hf:
            hf.create_dataset(f"{group}/dt", data=dt, compression="gzip", compression_opts=9)
            hf.create_dataset(f"{group}/gt", data=gt, compression="gzip", compression_opts=9)
            print(f"[OK] {group} partition.")
