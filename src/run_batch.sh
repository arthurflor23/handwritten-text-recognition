#!/bin/bash
#SBATCH --time=72:00:00   # walltime
#SBATCH --ntasks=4   # number of processor cores (i.e. tasks)
#SBATCH --nodes=1   # number of nodes
#SBATCH --mem-per-cpu=4096M   # memory per CPU core
#SBATCH -J "Flor-HTR"   # job name
#SBATCH --mail-user=lparrish@worldarchives.com
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH -e ../../flor_out/%j-err.txt
#SBATCH -o ../../flor_out/%j-out.txt


module load python/3.6
source /fslgroup/fslg_census/compute/projects/FlorHTR_env/bin/activate

python3 main.py "--source" "$1" "--weights" "$2" "--arch" "flor" "--archive" "True" "--csv" "$3" "--append" "--test" "$4"
