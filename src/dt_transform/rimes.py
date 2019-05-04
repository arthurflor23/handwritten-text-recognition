"""Transform Rimes dataset"""

from multiprocessing import Pool
from functools import partial
import xml.etree.ElementTree as ET
import numpy as np
import cv2
import os


def dataset(env, preproc_func):
    """Load and save npz file of the ground truth and images (preprocessed)"""

    xml = os.path.join(env.raw_source, "training_2011.xml")
    dt, gt = build_data_from(xml, "training_2011", env.train, env.valid)

    index = int(len(dt) * 0.9)
    np.savez_compressed(env.valid, dt=np.array(dt[index:]), gt=np.array(gt[index:]))
    np.savez_compressed(env.train, dt=dt[:index], gt=gt[:index])

    xml = os.path.join(env.raw_source, "eval_2011_annotated.xml")
    dt, gt = build_data_from(env, xml, "eval_2011", preproc_func)
    np.savez_compressed(env.test, dt=dt, gt=gt)


def build_data_from(env, xml, partition, preproc_func):
    """Preprocess images with pool function"""

    root = ET.parse(xml).getroot()
    dt, gt = [], []

    for page_tag in root:
        basename = page_tag.attrib["FileName"]
        page_path = os.path.join(env.raw_source, partition, basename)
        page = cv2.imread(page_path, cv2.IMREAD_GRAYSCALE)

        for i, line_tag in enumerate(page_tag.iter("Line")):
            bottom = abs(int(line_tag.attrib["Bottom"]))
            left = abs(int(line_tag.attrib["Left"]))
            right = abs(int(line_tag.attrib["Right"]))
            top = abs(int(line_tag.attrib["Top"]))

            line = page[top:bottom, left:right]
            gt.append(line_tag.attrib["Value"].strip())
            dt.append(line)

    pool = Pool()
    dt = pool.map(partial(preproc_func, img_size=env.model_input_size, read_first=False), dt)
    pool.close()
    pool.join()

    return dt, gt
