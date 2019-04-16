#!/bin/bash

### ---------------------------------------- ###
### define dataset name and data environment ###
### ---------------------------------------- ###
# DATASET="bentham"
DATASET="iam"
# DATASET="saintgall"
# DATASET="rimes"
DATA_DIR="./data/$DATASET"
OUTPUT_DIR="./output"


### ----------------------------------------------- ###
### structure the raw dataset to the design pattern ###
### ----------------------------------------------- ###
# python ./src/normalize/$DATASET.py --data_dir $DATA_DIR


### ---------------------- ###
### preprocess the dataset ###
### ---------------------- ###
python ./src/preprocess.py --input_dir $DATA_DIR


### ----------- ###
### train model ###
### ----------- ###
# python ./src/train.py --input_dir $DATA_DIR \
#                       --output_dir $OUTPUT_DIR
#                       --train_steps 500 \
#                       --eval_steps 30 \
#                       --learning_rate 0.01


### ---------- ###
### test model ###
### ---------- ###
# python ./src/test.py --input_dir $DATA_DIR \
#                      --output_dir $OUTPUT_DIR