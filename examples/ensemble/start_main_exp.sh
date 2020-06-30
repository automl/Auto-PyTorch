#!/bin/bash

RUN_ID=$1        # Starting run id
DATASET_ID=0
limit=7          # 8 Datasets
SEED=$2

until [ $DATASET_ID -gt $limit ]
do
   yes | sbatch examples/ensemble/job_main_exp.sh $RUN_ID $DATASET_ID $SEED
   ((RUN_ID++))

   ((DATASET_ID++))   
done
