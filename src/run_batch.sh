#!/bin/bash

#SBATCH --cpus-per-task=4   # number of processor cores (i.e., tasks)
#SBATCH --mem=4G   # memory per CPU core
#SBATCH -J "Flor-Batch"   # job name
#SBATCH -e ./flor_out/%j-err.txt
#SBATCH -o ./flor_out/%j-out.txt
#SBATCH -p htc


module load python/3.8
source /shared/home/cyclemgmt/FlorHTR_env/bin/activate

python3 main.py "--source" "$1" "--weights" "$2" "--csv" "$3" "--append"
