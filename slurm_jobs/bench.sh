#!/bin/bash
#SBATCH -p bosch_cpu-cascadelake                        # partition (queue) (test_cpu-ivy, all_cpu-cascadelake, bosch_cpu-cascadelake)
#SBATCH --mem 6000                                      # memory pool for all cores
#SBATCH -t 10-00:00                                      # time (D-HH:MM)
#SBATCH -N 1                                            # number of nodes
#SBATCH -c 1                                            # number of cores
#SBATCH -a 1-17500%300                                  # array size
#SBATCH -o logs/shapedmlp_2k/%x.%N.%j.out                    # STDOUT  (the folder log has to be created prior to running or this won't work)
#SBATCH -e logs/shapedmlp_2k/%x.%N.%j.err                    # STDERR  (the folder log has to be created prior to running or this won't work)
#SBATCH -J LCBench                                    # sets the job name. If not specified, the file name will be used as job name
# Print some information about the job to STDOUT
echo "Workingdir: $PWD";
echo "Started at $(date)";
echo "Running job $SLURM_JOB_NAME using $SLURM_JOB_CPUS_PER_NODE cpus per node with given JID $SLURM_JOB_ID on queue $SLURM_JOB_PARTITION"; 

# Activate venv
source env/bin/activate

# Variables
SEED=1
BUDGET=50

# Array jobs
python3 run_bench.py --run_id $SLURM_ARRAY_TASK_ID --offset 0 --seed $SEED --budget $BUDGET --config_root_dir configs/refit/resnet_9 --root_logdir logs/resnet_9 --device_type gpu --dataloader_worker 2

# Done
echo "DONE";
echo "Finished at $(date)";
