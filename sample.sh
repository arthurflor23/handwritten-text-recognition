#!/bin/bash

INPUT_DIR="./data/"
DATASET="iam"

python ./src/normalize/$DATASET.py --input_dir $INPUT_DIR

# OUTPUT_DIR="./data_proc/"

# python ./src/preprocess.py --input_dir $INPUT_DIR \
#                            --gt_dir $GT_DIR \
#                            --subsets_dir $SUBSETS_DIR \
#                            --output_dir $OUTPUT_DIR