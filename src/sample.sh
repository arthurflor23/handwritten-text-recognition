#!/bin/bash

### ---------------------------------------- ###
### define dataset name and data environment ###
### ---------------------------------------- ###
# DATASET="bentham"
# DATASET="iam"
# DATASET="saintgall"
# DATASET="rimes"
DATASET="temp"

DATASET_PATH="../data/$DATASET"
OUTPUT_PATH="../output"


### ----------------------------------------------- ###
### structure the raw dataset to the design pattern ###
### ----------------------------------------------- ###
# python tasks/normalize.py --data_source $DATASET_PATH


### ----------- ###
### train model ###
### ----------- ###
python tasks/train.py --data_source $DATASET_PATH \
                      --data_output $OUTPUT_PATH \
                      --epochs 1 \
                      --batch 32


### ---------- ###
### test model ###
### ---------- ###
# python tasks/test.py --data_source $DATASET_PATH --data_output $OUTPUT_DIR
