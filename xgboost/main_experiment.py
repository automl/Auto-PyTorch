import argparse
import pickle
import json
import logging
logging.basicConfig(level=logging.DEBUG)
import os
import random
import time

import hpbandster.core.nameserver as hpns
import hpbandster.core.result as hpres
from hpbandster.optimizers import BOHB as BOHB
from hpbandster.optimizers import RandomSearch as RS
import numpy as np
import openml

from data.loader import Loader

from worker import XGBoostWorker, TabNetWorker


parser = argparse.ArgumentParser(
    description='XGBoost experiment.'
)
parser.add_argument(
    '--run_id',
    type=str,
    help='The run id of the optimization run.',
    default='XGBoost',
)
parser.add_argument(
    '--working_directory',
    type=str,
    help='The working directory where results will be stored.',
    default='.',
)
parser.add_argument(
    '--nic_name',
    type=str,
    help='Which network interface to use for communication.',
    default='ib0',
)
parser.add_argument(
    '--optimizer',
    type=str,
    help='Which optimizer to use for the experiment.',
    default='bohb',
)
parser.add_argument(
    '--model',
    type=str,
    help='Which model to use for the experiment.',
    default='tabnet',
)
parser.add_argument(
    '--task_id',
    type=int,
    help='Minimum budget used during the optimization.',
    default=233109,
)
parser.add_argument(
    '--seed',
    type=int,
    help='Seed used for the experiment.',
    default=11,
)
parser.add_argument(
    '--max_budget',
    type=float,
    help='Maximum budget used during the optimization.',
    default=1,
)
parser.add_argument(
    '--min_budget',
    type=float,
    help='Minimum budget used during the optimization.',
    default=1,
)
parser.add_argument(
    '--n_iterations',
    type=int,
    help='Number of iterations.',
    default=10,
)
parser.add_argument(
    '--n_workers',
    type=int,
    help='Number of workers to run in parallel.',
    default=2,
)
parser.add_argument(
    '--worker',
    help='Flag to turn this into a worker process',
    action='store_true',
)

args = parser.parse_args()

np.random.seed(args.seed)
random.seed(args.seed)

host = hpns.nic_name_to_host(args.nic_name)
loader = Loader(task_id=args.task_id)

#
# check_leak_status(loader.get_splits())
# check_split_stratification(loader.get_splits())

nr_classes = int(openml.datasets.get_dataset(loader.get_dataset_id()).qualities['NumberOfClasses'])

if nr_classes != 2:
    param = {
        'objective': 'multi:softmax',
        'num_class': nr_classes + 1,
        'disable_default_eval_metric': 1,
        'seed': args.seed,
        'nthread': 2,
    }
else:
    param = {
        'objective': 'binary:logistic',
        'disable_default_eval_metric': 1,
        'seed': args.seed,
        'nthread': 2,
    }

worker_choices = {
    'tabnet': TabNetWorker,
    'xgboost': XGBoostWorker,
}

model_worker = worker_choices[args.model]
if args.worker:
    time.sleep(5)  # short artificial delay to make sure the nameserver is already running
    worker = model_worker(
        run_id=args.run_id,
        host=host,
        param=param,
        splits=loader.get_splits(),
        categorical_information=loader.categorical_information,
    )
    while True:
        try:
            worker.load_nameserver_credentials(
                working_directory=args.working_directory,
            )
            break
        except RuntimeError:
            pass
    worker.run(background=False)
    exit(0)

print(f'Experiment started with task id: {args.task_id}')
run_directory = os.path.join(
    args.working_directory,
    f'{args.task_id}',
    f'{args.seed}',
)
os.makedirs(run_directory, exist_ok=True)

NS = hpns.NameServer(
    run_id=args.run_id,
    host=host,
    port=0,
    working_directory=args.working_directory,
)
ns_host, ns_port = NS.start()

worker = model_worker(
    run_id=args.run_id,
    host=host,
    param=param,
    splits=loader.get_splits(),
    categorical_information=loader.categorical_information,
    nameserver=ns_host,
    nameserver_port=ns_port
)
worker.run(background=True)
result_logger = hpres.json_result_logger(directory=run_directory, overwrite=False)

optimizer_choices = {
    'bohb': BOHB,
    'random_search': RS,
}

optimizer = optimizer_choices[args.optimizer]

bohb = optimizer(
    configspace = model_worker.get_default_configspace(seed=args.seed),
    run_id = args.run_id,
    host=host,
    nameserver=ns_host,
    nameserver_port=ns_port,
    min_budget=args.min_budget,
    max_budget=args.max_budget,
    result_logger=result_logger,
)

res = bohb.run(
    n_iterations=args.n_iterations,
    min_n_workers=args.n_workers
)

bohb.shutdown(shutdown_workers=True)
NS.shutdown()

with open(os.path.join(run_directory, 'results.pkl'), 'wb') as fh:
    pickle.dump(res, fh)

id2config = res.get_id2config_mapping()
incumbent = res.get_incumbent_id()

all_runs = res.get_all_runs()

best_config = id2config[incumbent]['config']
print('Best found configuration:', best_config)
print('A total of %i unique configurations where sampled.' % len(id2config.keys()))
print('A total of %i runs where executed.' % len(res.get_all_runs()))
print('Total budget corresponds to %.1f full function evaluations.'%(sum([r.budget for r in all_runs])/args.max_budget))
print('Total budget corresponds to %.1f full function evaluations.'%(sum([r.budget for r in all_runs])/args.max_budget))
print('The run took  %.1f seconds to complete.'%(all_runs[-1].time_stamps['finished'] - all_runs[0].time_stamps['started']))

loader = Loader(task_id=args.task_id, val_fraction=0)
worker = model_worker(
    args.run_id,
    param=param,
    splits=loader.get_splits(),
    categorical_information=loader.categorical_information,
    nameserver='127.0.0.1',
)
refit_result = worker.refit(best_config)

with open(os.path.join(run_directory, 'refit_result.json'), 'w') as file:
    json.dump(refit_result, file)
