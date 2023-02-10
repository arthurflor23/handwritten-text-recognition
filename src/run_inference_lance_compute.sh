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

column=$1
weights=$2
csv_path="/home/lanceap/compute/1950_Transcription/$column"
sorted_snippets="/home/lanceap/compute/sorted_snippets/"
echo "${sorted_snippets}"
column_directory="$sorted_snippets/$column"
column_directory=${column_directory//\"}

find "${column_directory}" -type f -exec sg fslg_census "sbatch run_batch.sh {} ${weights} ${csv_path}" \;
