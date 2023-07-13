#!/bin/bash
#SBATCH --cpus-per-task=8   # number of processor cores (i.e., tasks)
#SBATCH --mem=50G   # memory per CPU core
#SBATCH -J "Flor-Main"   # job name
#SBATCH -e /shared/home/cyclemgmt/handwritten-text-recognition-master/src/flor_out/%j-err.txt
#SBATCH -o /shared/home/cyclemgmt/handwritten-text-recognition-master/src/flor_out/%j-out.txt
#SBATCH -p htc

#Usage sbatch run_inference_job.sh Directory ColumnName WeightsName


snippets_path=$1
column=$2
weights="../weights/$3.hdf5"
csv_path="$4/$column"
column_directory="$snippets_path/$column"
extract=$5

n=0
if [ $extract = "T" ]; then
  mkdir "$column_directory/Batch_$n"

  for f in $(find $column_directory \ (-name '*.jpg'  -o -name '*.jp2'\ )); do
    cat $f
    if (( $(ls "$column_directory/Batch_$n" | wc -l) >= 20000 )); then
      ((++n));
      cat $n
      mkdir "$column_directory/Batch_$n"
    fi
    cp $f "$column_directory/Batch_$n"
  done
fi

#find "${column_directory}" -type f -exec sbatch run_batch.sh {} $weights $csv_path \;
