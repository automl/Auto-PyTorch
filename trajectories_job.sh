#!/bin/bash
#SBATCH -p bosch_cpu-cascadelake                                     # partition (queue) (test_cpu-ivy, all_cpu-cascadelake, bosch_cpu-cascadelake)
#SBATCH -t 1-00:00                                      # time (D-HH:MM)
#SBATCH -N 1                                            # number of nodes
#SBATCH -c 1                                            # number of cores
#SBATCH -a 1-9                                         # array size
#SBATCH -o logs/cluster/%x.%N.%j.out                        # STDOUT  (the folder log has to be created prior to running or this won't work)
#SBATCH -e logs/cluster/%x.%N.%j.err                        # STDERR  (the folder log has to be created prior to running or this won't work)
#SBATCH -J create_ensemble                                         # sets the job name. If not specified, the file name will be used as job name
# Print some information about the job to STDOUT
echo "Workingdir: $PWD";
echo "Started at $(date)";
echo "Running job $SLURM_JOB_NAME using $SLURM_JOB_CPUS_PER_NODE cpus per node with given JID $SLURM_JOB_ID on queue $SLURM_JOB_PARTITION"; 

# Activate venv
source env/bin/activate
export PYTHONPATH=$PWD

# Array jobs 
python3 create_trajectory.py --test false --run_id $SLURM_ARRAY_TASK_ID

# Done
echo "DONE";
echo "Finished at $(date)";
