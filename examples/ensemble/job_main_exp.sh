#!/bin/bash
#SBATCH -p bosch_cpu-cascadelake                        # partition (queue) (test_cpu-ivy, all_cpu-cascadelake, bosch_cpu-cascadelake)
#SBATCH -t 2-01:00                                      # time (D-HH:MM)
#SBATCH -N 1                                            # number of nodes
#SBATCH -c 2                                            # number of cores
#SBATCH -a 1-3                                          # array size
#SBATCH -o logs/cluster/%x.%N.%j.out                    # STDOUT  (the folder log has to be created prior to running or this won't work)
#SBATCH -e logs/cluster/%x.%N.%j.err                    # STDERR  (the folder log has to be created prior to running or this won't work)
#SBATCH -J apt_test                                    # sets the job name. If not specified, the file name will be used as job name
# Print some information about the job to STDOUT
echo "Workingdir: $PWD";
echo "Started at $(date)";
echo "Running job $SLURM_JOB_NAME using $SLURM_JOB_CPUS_PER_NODE cpus per node with given JID $SLURM_JOB_ID on queue $SLURM_JOB_PARTITION"; 

# Activate venv
source env/bin/activate
export PYTHONPATH=$PWD

if [ $SLURM_ARRAY_TASK_ID -gt 1 ]
then
    sleep 10
fi

# Array jobs
python3 -W ignore examples/ensemble/test_ensemble.py --run_id $1 --task_id $SLURM_ARRAY_TASK_ID --num_workers 3 --dataset_id $2 --seed $3 --ensemble_setting ensemble --portfolio_type greedy --num_threads 2 --test false

# Done
echo "DONE";
echo "Finished at $(date)";


# DEBUGGING:
#python3 -W ignore examples/ensemble/test_ensemble.py --run_id 999 --task_id 1 --num_workers 1 --dataset_id 0 --seed 1 --ensemble_setting ensemble --portfolio_type greedy --num_threads 2 --test true
