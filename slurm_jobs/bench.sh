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
BUDGET=25     # 12, 25, 50

# Array jobs
# 1-17500
python3 run_bench.py --run_id $SLURM_ARRAY_TASK_ID --offset 0 --seed $SEED --budget $BUDGET --config_dir configs/refit/bench_configs/bench_2k --root_logdir logs/shapedmlp_2k
# 17500-35000
python3 run_bench.py --run_id $SLURM_ARRAY_TASK_ID --offset 17500 --seed $SEED --budget $BUDGET --config_dir configs/refit/bench_configs/bench_2k --root_logdir logs/shapedmlp_2k
# 35000-52500
python3 run_bench.py --run_id $SLURM_ARRAY_TASK_ID --offset 35000 --seed $SEED --budget $BUDGET --config_dir configs/refit/bench_configs/bench_2k --root_logdir logs/shapedmlp_2k
# 52500-70000
python3 run_bench.py --run_id $SLURM_ARRAY_TASK_ID --offset 52500 --seed $SEED --budget $BUDGET --config_dir configs/refit/bench_configs/bench_2k --root_logdir logs/shapedmlp_2k


# Done
echo "DONE";
echo "Finished at $(date)";
