#!/bin/bash

RUN_ID=$1       # Starting run id
DATASET_ID=0
limit=7          # 5 Datasets --> 4
SEED=$2

#RUN_ID, DATASET_ID, SEED, ENSEMBLE_SETTING, PORTFOLIO_TYPE, OPTIMIZER

until [ $DATASET_ID -gt $limit ]
do
   yes | sbatch job_main_exp.sh $RUN_ID $DATASET_ID $SEED
   ((RUN_ID++))

   ((DATASET_ID++))   
done
