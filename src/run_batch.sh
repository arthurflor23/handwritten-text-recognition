#!/bin/bash

#SBATCH --cpus-per-task=2   # number of processor cores (i.e., tasks)
#SBATCH --mem=3G   # memory per CPU core
#SBATCH -J "Flor-Batch"   # job name
#SBATCH -e ./flor_out/%j-err.txt
#SBATCH -o ./flor_out/%j-out.txt
#SBATCH -p htc


source /shared/home/cyclemgmt/FlorHTR_env/bin/activate

delete_finished = $5

if [$delete_finished]; then
  python3 -u main.py "--source" "$1" "--weights" "$2" "--csv" "$3" "--append" "--finished" "$4" "--delete_finished"
  exit
fi

python3 -u main.py "--source" "$1" "--weights" "$2" "--csv" "$3" "--append" "--finished" "$4"
