#!/bin/bash
#SBATCH --time=72:00:00   # walltime
#SBATCH --ntasks=8   # number of processor cores (i.e. tasks)
#SBATCH --nodes=1   # number of nodes
#SBATCH --mem-per-cpu=2048M   # memory per CPU core
#SBATCH -J "Flor-HTR"   # job name
#SBATCH --mail-user=lparrish@worldarchives.com
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH -e ../../flor_out/%j-err.txt
#SBATCH -o ../../flor_out/%j-out.txt


module load python/3.6.9
source /fslgroup/fslg_census/compute/projects/FlorHTR_env/bin/activate

python main.py "--source $1 --weights $2 --arch flor --archive True --csv $3"
