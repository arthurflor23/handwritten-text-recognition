#!/bin/sh

# Example commands for running the package pipelines.
# Demonstrates training and testing for different datasets.

python sarah --source washington --recognition flor --training
python sarah --source washington --recognition flor --test --recognition-run-id -1

python sarah --source iam --recognition flor --training
python sarah --source iam --recognition flor --test --recognition-run-id -1

python sarah --source rimes --recognition flor --training-ratio 0.9 --validation-ratio 0.1 --training
python sarah --source rimes --recognition flor --test --recognition-run-id -1

python sarah --source all-in-one --recognition flor --training
python sarah --source all-in-one --recognition flor --test --recognition-run-id -1

python sarah --source all-in-one --synthesis flor --training-ratio 1.0 --validation-ratio 0.0 --test-ratio 0.0 --training
