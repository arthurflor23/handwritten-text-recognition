#!/bin/bash

### ---------------------------------------- ###
### define dataset name and data environment ###
### ---------------------------------------- ###
# DATASET="bentham"
DATASET="iam"
# DATASET="saintgall"
# DATASET="rimes"

DATASET_DIR="../data/$DATASET"
OUTPUT_DIR="../output"


### ----------------------------------------------- ###
### structure the raw dataset to the design pattern ###
### ----------------------------------------------- ###
# python normalize/$DATASET.py --dataset_dir $DATASET_DIR


### ---------------------- ###
### preprocess the dataset ###
### ---------------------- ###
python preproc/preprocess.py --dataset_dir $DATASET_DIR


### ----------- ###
### train model ###
### ----------- ###
# python train.py --dataset_dir $DATASET_DIR \
#                 --output_dir $OUTPUT_DIR
#                 --train_steps 500 \
#                 --eval_steps 30 \
#                 --learning_rate 0.01


### ---------- ###
### test model ###
### ---------- ###
# python test.py --dataset_dir $DATASET_DIR \
#                --output_dir $OUTPUT_DIR