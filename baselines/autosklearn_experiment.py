import argparse
import json
import os
import random

import autosklearn.classification
from autosklearn.metrics import balanced_accuracy
import numpy as np
import openml
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import train_test_split


def create_dir(
        path: str,
):
    """Create the directory/subdirectories for the given path.

    Given a path, check the directory/subdirectories that are
    part of the path, perform a few checks and create the parts
    that are missing.

    Parameters:
    -----------
    path: str
        The path to be created.
    """
    if os.path.exists(path):
        if not os.path.isdir(path):
            os.makedirs(path)
    else:
        os.makedirs(path)


parser = argparse.ArgumentParser(
    description='autosklearn_gb'
)
parser.add_argument(
    '--run_id',
    help='Unique id to identify the AutoSklearn run.',
    default='autosklearn_gb',
    type=str,
)
parser.add_argument(
    '--tmp_dir',
    help='Temporary node storage.',
    default='path/temporary_storage',
    type=str,
)
parser.add_argument(
    '--working_dir',
    help='Working directory where to store the results.',
    default='path/working_dir',
    type=str,
)
parser.add_argument(
    '--task_id',
    help='Task id so that the dataset can be retrieved from OpenML.',
    default=233088,
    type=int,
)
parser.add_argument(
    '--nr_workers',
    help='Number of workers.',
    default=10,
    type=int,
)
parser.add_argument(
    '--seed',
    help='Seed number.',
    default=11,
    type=int,
)

args = parser.parse_args()
np.random.seed(args.seed)
random.seed(args.seed)

task = openml.tasks.get_task(task_id=args.task_id)
dataset = task.get_dataset()
X, y, categorical_indicator, _ = dataset.get_data(
    dataset_format='array',
    target=dataset.default_target_attribute,
)
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=args.seed,
    stratify=y,
)

output_directory = os.path.join(
    args.working_dir,
    f'{args.seed}',
    f'{args.task_id}',
    'output',
)
result_directory = os.path.join(
    args.working_dir,
    f'{args.seed}',
    f'{args.task_id}',
    'results',
)

feat_types = ['Categorical' if feature else 'Numerical' for feature in categorical_indicator]
resampling_strategy = StratifiedShuffleSplit
resampling_strategy_arguments = {'test_size': 0.25, 'random_state': args.seed, 'n_splits': 1}
# This is a stratified split, so this should work better.
# validation_policy = {'holdout': {'train_size': 0.75, 'shuffle': True}}

if __name__ == '__main__':
    gb_autosklearn = autosklearn.classification.AutoSklearnClassifier(
        include_estimators=['gradient_boosting'],
        include_preprocessors=['no_preprocessing'],
        time_left_for_this_task=324000,
        ensemble_size=1,
        seed=args.seed,
        memory_limit=12000,
        output_folder=output_directory,
        tmp_folder=os.path.join(args.tmp_dir, 'autosklearn'),
        resampling_strategy=resampling_strategy,
        resampling_strategy_arguments=resampling_strategy_arguments,
        initial_configurations_via_metalearning=0,
        metric=balanced_accuracy,
        n_jobs=args.nr_workers,
        smac_scenario_args={'runcount_limit': 840},
    )
    gb_autosklearn.fit(X_train.copy(), y_train.copy(), dataset_name=dataset.name)
    print(gb_autosklearn.sprint_statistics())
    gb_autosklearn.refit(X_train.copy(), y_train.copy())
    y_test_pred = gb_autosklearn.predict(X_test)
    y_train_pred = gb_autosklearn.predict(X_train)

    train_acc = balanced_accuracy_score(
        y_train,
        y_train_pred,
    )
    test_acc = balanced_accuracy_score(
        y_test,
        y_test_pred,
    )

    information = {
        'train_acc': train_acc,
        'test_acc': test_acc,
    }
    create_dir(result_directory)
    with open(os.path.join(result_directory, 'refit_result.json'), 'w') as file:
        json.dump(information, file)
