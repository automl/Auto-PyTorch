#!/bin/bash
#SBATCH -p bosch_cpu-cascadelake                        # partition (queue) (test_cpu-ivy, all_cpu-cascadelake, bosch_cpu-cascadelake)
#SBATCH --mem 6000                                      # memory pool for all cores
#SBATCH -t 2-00:00                                      # time (D-HH:MM)
#SBATCH -N 1                                            # number of nodes
#SBATCH -c 1                                            # number of cores
#SBATCH -a 1-1000                                       # array size
#SBATCH -o logs/cluster/%x.%N.%j.out                    # STDOUT  (the folder log has to be created prior to running or this won't work)
#SBATCH -e logs/cluster/%x.%N.%j.err                    # STDERR  (the folder log has to be created prior to running or this won't work)
#SBATCH -J openml_lc                                    # sets the job name. If not specified, the file name will be used as job name
# Print some information about the job to STDOUT
echo "Workingdir: $PWD";
echo "Started at $(date)";
echo "Running job $SLURM_JOB_NAME using $SLURM_JOB_CPUS_PER_NODE cpus per node with given JID $SLURM_JOB_ID on queue $SLURM_JOB_PARTITION"; 

# Activate venv
source env/bin/activate

# Array jobs 
python3 run_bench.py --run_id $SLURM_ARRAY_TASK_ID --architecture shapedmlpnet --logging step

# Done
echo "DONE";
echo "Finished at $(date)";
