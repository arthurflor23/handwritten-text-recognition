#!/bin/bash
#SBATCH --time=96:00:00   # walltime
#SBATCH --ntasks=4   # number of processor cores (i.e. tasks)
#SBATCH --nodes=1   # number of nodes
#SBATCH --mem-per-cpu=2048M   # memory per CPU core
#SBATCH -J "Flor-Main"   # job name
#SBATCH --mail-user=lparrish@worldarchives.com
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH -e ../../flor_out/%j-err.txt
#SBATCH -o ../../flor_out/%j-out.txt

#Usage sbatch run_inference_job.sh Directory ColumnName WeightsName


snippets_path=$1
column=$2
weights="../weights/$3.hdf5"
csv_path="$4/$column"
test=${5:-0}
column_directory="$snippets_path/$column"
column_directory=${column_directory//\"}

find "${column_directory}" -type f -exec sg fslg_census "sbatch run_batch.sh {} ${weights} ${csv_path} ${test}" \;
