"""Transform Rimes dataset"""

from multiprocessing import Pool
from functools import partial
import xml.etree.ElementTree as ET
import h5py
import cv2
import os


def dataset(env, preproc, encode):
    """Load and save hdf5 file of the ground truth and images (preprocessed)"""

    def transform(group, xml, partition, valid=None):
        with h5py.File(env.source, "a") as hf:
            xml = os.path.join(env.raw_source, xml)
            dt, gt = build_data(env, xml, partition, preproc, encode)

            if valid is None:
                hf.create_dataset(f"{group}/dt", data=dt, compression="gzip", compression_opts=9)
                hf.create_dataset(f"{group}/gt", data=gt, compression="gzip", compression_opts=9)
                del dt, gt
            else:
                index = int(len(dt) * 0.9)
                train_dt, train_gt = dt[:index], gt[:index]
                valid_dt, valid_gt = dt[index:], gt[index:]
                del dt, gt

                hf.create_dataset(f"{group}/dt", data=train_dt, compression="gzip", compression_opts=9)
                hf.create_dataset(f"{group}/gt", data=train_gt, compression="gzip", compression_opts=9)
                del train_dt, train_gt

                hf.create_dataset(f"{valid}/dt", data=valid_dt, compression="gzip", compression_opts=9)
                hf.create_dataset(f"{valid}/gt", data=valid_gt, compression="gzip", compression_opts=9)
                del valid_dt, valid_gt

    transform(group="train", xml="training_2011.xml", partition="training_2011", valid="valid")
    transform(group="test", xml="eval_2011_annotated.xml", partition="eval_2011")


def build_data(env, xml, partition, preproc, encode):
    """Preprocess images with pool function"""

    root = ET.parse(xml).getroot()
    dt, gt = [], []

    for page_tag in root:
        basename = page_tag.attrib["FileName"]
        page_path = os.path.join(env.raw_source, partition, basename)
        page = cv2.imread(page_path, cv2.IMREAD_GRAYSCALE)

        for i, line_tag in enumerate(page_tag.iter("Line")):
            text_line = line_tag.attrib["Value"].strip()

            if len(text_line) > 0:
                gt.append(text_line)
                dt.append(page[abs(int(line_tag.attrib["Top"])):abs(int(line_tag.attrib["Bottom"])),
                               abs(int(line_tag.attrib["Left"])):abs(int(line_tag.attrib["Right"]))])

    pool = Pool()
    dt = pool.map(partial(preproc, img_size=env.input_size, read_first=False), dt)
    gt = pool.map(partial(encode, charset=env.charset, mtl=env.max_text_length), gt)
    pool.close()
    pool.join()

    return dt, gt
