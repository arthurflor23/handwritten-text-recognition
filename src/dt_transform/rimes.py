"""Transform Rimes dataset"""

from multiprocessing import Pool
from functools import partial
import xml.etree.ElementTree as ET
import numpy as np
import cv2
import os


def dataset(env, preproc, encode):
    """Load and save npz file of the ground truth and images (preprocessed)"""

    xml = os.path.join(env.raw_source, "eval_2011_annotated.xml")
    test_dt, test_gt = build_data(env, xml, "eval_2011", preproc, encode)

    xml = os.path.join(env.raw_source, "training_2011.xml")
    train_dt, train_gt = build_data(env, xml, "training_2011", preproc, encode)
    index = int(len(train_dt) * 0.9)

    np.savez_compressed(
        env.source,
        train_dt=train_dt[:index],
        train_gt=train_gt[:index],
        valid_dt=train_dt[index:],
        valid_gt=train_gt[index:],
        test_dt=test_dt,
        test_gt=test_gt,
    )


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
        del page
    del root

    pool = Pool()
    dt = pool.map(partial(preproc, img_size=env.model_input_size, read_first=False), dt)
    gt = pool.map(partial(encode, charset=env.charset), gt)
    pool.close()
    pool.join()

    return dt, gt
