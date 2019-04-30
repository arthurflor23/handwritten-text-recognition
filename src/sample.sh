#!/bin/bash

### ---------------------------------------- ###
### define dataset name and data environment ###
### ---------------------------------------- ###
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
python tasks/train.py --dataset $DATASET --epochs 1 --batch 1


### ---------- ###
### test model ###
### ---------- ###
# python tasks/test.py --dataset $DATASET_PATH
