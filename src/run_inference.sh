#!/bin/bash
#SBATCH --time=96:00:00   # walltime
#SBATCH --ntasks=8   # number of processor cores (i.e. tasks)
#SBATCH --nodes=1   # number of nodes
#SBATCH --mem-per-cpu=2048M   # memory per CPU core
#SBATCH -J "Flor-Main"   # job name
#SBATCH --mail-user=lparrish@worldarchives.com
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH -e ../../flor_out/%j-err.txt
#SBATCH -o ../../flor_out/%j-out.txt

#***** NOTE: run this using: sg fslg_census "sbatch run_inference.sh job_config_path{String} images_path{String}"
# Parameters:
#   job_config_path - Path to the job configuration file
#   images_path - Path to the directory

config_name=$1
column=$2
weights=$3
csv_path=$4
job_config_path="../../CensusSegmenter/config/job_config/job_${config_name}.yaml"
sorted_snippets=${$(grep "SNIPPETS" "${job_config_path}" | awk '{print $2}')%/*}"/sorted_snippets"
echo sorted_snippets
column_directory="$sorted_snippets/$column"

find"${column_directory}" -type f -exec sg fslg_census "sbatch run_batch.sh {} ${weights} ${csv_path}" \;
