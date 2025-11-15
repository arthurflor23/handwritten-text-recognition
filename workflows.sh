#!/bin/sh

# Example commands for running package pipelines with multiple datasets.
# Demonstrates both training and testing modes for recognition and synthesis.
# Shows usage of dataset splitting, run identifiers, and GPU selection.

python sarah --source washington --recognition flor --training --gpu 0
python sarah --source washington --recognition flor --test --recognition-run-id -1

python sarah --source iam --recognition flor --training --gpu 0
python sarah --source iam --recognition flor --test --recognition-run-id -1

python sarah --source rimes --recognition flor --training-ratio 0.9 --validation-ratio 0.1 --training --gpu 0
python sarah --source rimes --recognition flor --test --recognition-run-id -1

python sarah --source all-in-one --recognition flor --training --gpu 0
python sarah --source all-in-one --recognition flor --test --recognition-run-id -1

python sarah --source all-in-one --synthesis flor --training-ratio 1.0 --validation-ratio 0.0 --test-ratio 0.0 --training --gpu 0
python sarah --source all-in-one --synthesis flor --test --synthesis-run-id -1
