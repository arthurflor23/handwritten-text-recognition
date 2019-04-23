#!/bin/bash

### ---------------------------------------- ###
### define dataset name and data environment ###
### ---------------------------------------- ###
# DATASET="bentham"
# DATASET="iam"
DATASET="saintgall"
# DATASET="rimes"

DATASET_DIR="../data/$DATASET"
OUTPUT_DIR="../output"


### ----------------------------------------------- ###
### structure the raw dataset to the design pattern ###
### ----------------------------------------------- ###
# python norm/$DATASET.py --dataset_dir $DATASET_DIR


### ---------------------- ###
### preprocess the dataset ###
### ---------------------- ###
# python preproc/preproc.py --dataset_dir $DATASET_DIR


### ----------- ###
### train model ###
### ----------- ###
python network/train.py --dataset_dir $DATASET_DIR \
                        --output_dir $OUTPUT_DIR \
                        # --train_steps 500 \
                        # --learning_rate 0.01


### ---------- ###
### test model ###
### ---------- ###
# python network/test.py --dataset_dir $DATASET_DIR \
#                        --output_dir $OUTPUT_DIR