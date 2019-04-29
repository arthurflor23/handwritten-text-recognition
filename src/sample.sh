#!/bin/bash

### ---------------------------------------- ###
### define dataset name and data environment ###
### ---------------------------------------- ###
OUTPUT="output"

# DATASET="bentham"
# DATASET="iam"
# DATASET="saintgall"
# DATASET="rimes"
DATASET="temp"


### ----------------------------------------------- ###
### structure the raw dataset to the design pattern ###
### ----------------------------------------------- ###
# python tasks/dt_transform.py --dataset $DATASET_PATH


### ----------- ###
### train model ###
### ----------- ###
python tasks/train.py --dataset $DATASET \
                      --output $OUTPUT \
                      --epochs 20 \
                      --batch 32


### ---------- ###
### test model ###
### ---------- ###
# python tasks/test.py --dataset $DATASET_PATH --output $OUTPUT_DIR
