#!/bin/bash
#SBATCH -p ml_gpu-rtx2080                               # partition (queue) (test_cpu-ivy, all_cpu-cascadelake, bosch_cpu-cascadelake)
#SBATCH -t 2-00:00                                      # time (D-HH:MM)
#SBATCH -N 1                                            # number of nodes
#SBATCH -c 4                                            # number of cores
#SBATCH -J create_bench                                    # sets the job name. If not specified, the file name will be used as job name
# Print some information about the job to STDOUT
echo "Workingdir: $PWD";
echo "Started at $(date)";
echo "Running job $SLURM_JOB_NAME using $SLURM_JOB_CPUS_PER_NODE cpus per node with given JID $SLURM_JOB_ID on queue $SLURM_JOB_PARTITION"; 

# Activate venv
source /home/zimmerl/LCBench_reader/env/bin/activate
python3 create_benchmark.py

# Done
echo "DONE";
echo "Finished at $(date)";
