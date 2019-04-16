#!/bin/bash

### ---------------------------------------- ###
### define dataset name and data environment ###
### ---------------------------------------- ###
PROJECT_ROOT=$(dirname $(readlink -f "$0"))

# DATASET="bentham"
# DATASET="iam"
DATASET="saintgall"
# DATASET="rimes"

DATASET_DIR="$PROJECT_ROOT/data/$DATASET"
OUTPUT_DIR="$PROJECT_ROOT/output"


### ----------------------------------------------- ###
### structure the raw dataset to the design pattern ###
### ----------------------------------------------- ###
# python $PROJECT_ROOT/src/normalize/$DATASET.py --data_dir $DATASET_DIR


### ---------------------- ###
### preprocess the dataset ###
### ---------------------- ###
python $PROJECT_ROOT/src/data/preproc.py --input_dir $DATASET_DIR


### ----------- ###
### train model ###
### ----------- ###
# python $PROJECT_ROOT/src/train.py --input_dir $DATASET_DIR \
                                #   --output_dir $OUTPUT_DIR
                                #   --train_steps 500 \
                                #   --eval_steps 30 \
                                #   --learning_rate 0.01


### ---------- ###
### test model ###
### ---------- ###
# python $PROJECT_ROOT/src/test.py --input_dir $DATASET_DIR \
#                                  --output_dir $OUTPUT_DIR