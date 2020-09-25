import argparse
import json
import os
import random
import time

from autoPyTorch import (
    AutoNetClassification,
    HyperparameterSearchSpaceUpdates,
)
import autoPyTorch.pipeline.nodes as autonet_nodes
from autoPyTorch.components.metrics.additional_logs import test_result
from utilities import return_best_config

import openml
import numpy as np
from sklearn.model_selection import train_test_split
import torch


def str2bool(v):
    if isinstance(v, bool):
        return (v, )
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return (True, )
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return (False, )
    elif v.lower() == 'conditional':
        return (True, False)
    else:
        raise argparse.ArgumentTypeError('No valid value given.')


# training settings
parser = argparse.ArgumentParser(description='Configuration for the experiment')
parser.add_argument(
    '--run_id',
    help='Unique id to identify the run.',
    default='BOHB_Autonet',
    type=str,
)
parser.add_argument(
    '--array_id',
    help='Array id to tread one job array as a HPB run.',
    default=-1,
    type=int,
)
parser.add_argument(
    '--learning_rate',
    help='Learning rate for the optimizer',
    default=0.01,
    type=float,
)
parser.add_argument(
    '--random_seed',
    help='Random seed for the given experiment. It will be used for all workers',
    default=11,
    type=int,
)
parser.add_argument(
    '--working_dir',
    help='Working directory to store live data.',
    default='.',
    type=str,
)
parser.add_argument(
    '--nr_workers',
    help='Number of workers for the given experiment.',
    default=1,
    type=int,
)
parser.add_argument(
    '--task_id',
    help='Task id so that the dataset can be retrieved from OpenML.',
    default=233088,
    type=int,
)
parser.add_argument(
    '--num_threads',
    help='Number of threads to use for the experiment.',
    default=1,
    type=int,
)
parser.add_argument(
    '--run_type',
    help='If this is a run for hyperparameter optimization or to obtain the final accuracy.',
    type=str,
    choices=['hpo_run', 'final_run'],
    default='final_run',
)
parser.add_argument(
    '--model_name',
    help='Name of the model if final run is chosen and there is a specific hp configuration for all tasks.',
    default='plain_network',
    type=str,
)

# Regularization method settings
parser.add_argument(
    '--use_swa',
    help='If stochastic weight averaging should be used.',
    type=str2bool,
    nargs='?',
    const=(True, ),
    default=(False, ),
)
parser.add_argument(
    '--use_se',
    help='If snapshot ensembling should be used.',
    type=str2bool,
    nargs='?',
    const=(True, ),
    default=(False, ),
)
parser.add_argument(
    '--use_lookahead',
    help='If the lookahead optimizing technique should be used.',
    type=str2bool,
    nargs='?',
    const=(True, ),
    default=(False, ),
)
parser.add_argument(
    '--use_weight_decay',
    help='If weight decay regularization should be used.',
    type=str2bool,
    nargs='?',
    const=(True, ),
    default=(False, ),
)
parser.add_argument(
    '--use_batch_normalization',
    help='If batch normalization regularization should be used.',
    type=str2bool,
    nargs='?',
    const=(True, ),
    default=(False, ),
)
parser.add_argument(
    '--use_skip_connection',
    help='If skip connections should be used. Turns the network into a residual network.',
    type=str2bool,
    nargs='?',
    const=(True, ),
    default=(False, ),
)
parser.add_argument(
    '--use_dropout',
    help='If dropout regularization should be used.',
    type=str2bool,
    nargs='?',
    const=(True, ),
    default=(False, ),
)
parser.add_argument(
    '--mb_choice',
    help='Multibranch network regularization. Only active when skip_connection is active.',
    type=str,
    choices=['none', 'shake-shake', 'shake-drop', 'all'],
    default='none',
)
parser.add_argument(
    '--use_adversarial_training',
    help='If adversarial training should be used.',
    type=str2bool,
    nargs='?',
    const=(True, ),
    default=(False, ),
)
parser.add_argument(
    '--example_augmentation',
    help='If methods that augment examples should be used',
    type=str,
    choices=['mixup', 'cutout', 'cutmix', 'standard', 'all'],
    default='standard',
)

# network settings
parser.add_argument(
    '--nr_units',
    help='Number of units per layer. To be used in the fixed architecture.',
    default=64,
    type=int,
)

args = parser.parse_args()
search_space_updates = HyperparameterSearchSpaceUpdates()

example_augmentation_choices = ['mixup', 'cutout', 'cutmix', 'standard']  if args.example_augmentation == 'all' else [args.example_augmentation]
multibranch_choices = ['none', 'shake-shake', 'shake-drop'] if args.mb_choice == 'all' else [args.mb_choice]

# Fixed architecture space


# Applying all of the hyperparameter updates as given.
search_space_updates.append(
    node_name="NetworkSelector",
    hyperparameter="shapedresnet:max_units",
    value_range=[args.nr_units],
    log=False,
)
search_space_updates.append(
    node_name="NetworkSelector",
    hyperparameter="shapedresnet:resnet_shape",
    value_range=["brick"],
)
search_space_updates.append(
    node_name="NetworkSelector",
    hyperparameter="shapedresnet:num_groups",
    value_range=[2],
    log=False,
)
search_space_updates.append(
    node_name="NetworkSelector",
    hyperparameter="shapedresnet:blocks_per_group",
    value_range=[2],
    log=False,
)
search_space_updates.append(
    node_name="CreateDataLoader",
    hyperparameter="batch_size",
    value_range=[128],
    log=False,
)
search_space_updates.append(
    node_name="NetworkSelector",
    hyperparameter="shapedresnet:use_dropout",
    value_range=[*args.use_dropout],
)
search_space_updates.append(
    node_name="NetworkSelector",
    hyperparameter="shapedresnet:multibranch_choices",
    value_range=multibranch_choices,
)
search_space_updates.append(
    node_name="NetworkSelector",
    hyperparameter="shapedresnet:use_batch_normalization",
    value_range=[*args.use_batch_normalization],
)
search_space_updates.append(
    node_name="NetworkSelector",
    hyperparameter="shapedresnet:use_skip_connection",
    value_range=[*args.use_skip_connection],
)
search_space_updates.append(
    node_name="OptimizerSelector",
    hyperparameter="adamw:use_weight_decay",
    value_range=[*args.use_weight_decay],
)
search_space_updates.append(
    node_name="OptimizerSelector",
    hyperparameter="adamw:learning_rate",
    value_range=[args.learning_rate],
    log=False,
)
search_space_updates.append(
    node_name="InitializationSelector",
    hyperparameter="initializer:initialize_bias",
    value_range=['Yes'],
)
search_space_updates.append(
    node_name="LearningrateSchedulerSelector",
    hyperparameter="cosine_annealing:T_max",
    value_range=[15],
    log=False,
)
search_space_updates.append(
    node_name="LearningrateSchedulerSelector",
    hyperparameter="cosine_annealing:T_mult",
    value_range=[2],
    log=False,
)
search_space_updates.append(
    node_name="NetworkSelector",
    hyperparameter="shapedresnet:activation",
    value_range=['relu'],
)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(args.random_seed)
random.seed(args.random_seed)

result_directory = os.path.join(
    args.working_dir,
    f'{args.nr_units}',
    f'{args.task_id}',
)

if args.array_id == 1:
    os.makedirs(result_directory, exist_ok=True)

task = openml.tasks.get_task(task_id=args.task_id)
dataset = task.get_dataset()
X, y, categorical_indicator, _ = dataset.get_data(
    dataset_format='array',
    target=dataset.default_target_attribute,
)
run_id = args.run_id

different_seeds = set()
while len(different_seeds) < 10:
    seed_candidate = random.randint(1, 100)
    if seed_candidate not in different_seeds and seed_candidate != args.random_seed:
        different_seeds.add(seed_candidate)

train_curves = []
validation_curves = []
test_curves = []
test_accuracies = []

run_preset = '/home/fr/fr_fr/fr_ak1206/experiments/presets/regularization_fixed_arch' if args.run_type == 'final_run' else '/home/fr/fr_fr/fr_ak1206/experiments/presets/hpo_regularization_fixed_arch'

invalid_arguments_for_hpo_ss_updates = [
    'strategy',
    'normalization_strategy',
    'preprocessor',
    'target_size_strategy',
    'over_sampling_method',
    'under_sampling_method',
    'embedding',
    'network',
    'initialization_method',
    'optimizer',
    'lr_scheduler',
    'loss_module',
    'se_lastk',
]

use_lookahead = [*args.use_lookahead]
use_swa = [*args.use_swa]
use_se = [*args.use_se]
use_adversarial_training = [*args.use_adversarial_training]

number_configs_model = {
    'cocktail': 840,
    'weight_decay': 40,
    'dropout': 80,
}

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=args.random_seed,
    stratify=y,
)
if args.run_type == 'final_run':
    for seed in [args.random_seed]:
        seed_exp_dir = os.path.join(result_directory, f'{seed}')
        os.makedirs(seed_exp_dir, exist_ok=True)

        model_name = args.model_name
        hpo_dir = os.path.join(
            result_directory,
            'hpo_run',
            f'{args.random_seed}',
        )

        # checking if hpo has been performed before and we have
        # the best found hyperparameter configuration before.
        if os.path.exists(hpo_dir) and os.path.isdir(hpo_dir):
            
            best_config = return_best_config(
                result_directory,
                number_configs_model[model_name],
                args.random_seed,
            )
            print(f'Best configuration loaded for task with id: {args.task_id}')
            print(best_config)

            new_hpo_search_space = HyperparameterSearchSpaceUpdates()
            for hyperparameter in best_config:
                parts = hyperparameter.split(':', 1)
                hyperparameter_node_name = parts[0]
                hyperparameter_name = parts[1]
                hyperparameter_value = best_config[hyperparameter]

                # Not pretty but overwrite
                if hyperparameter_name == 'batch_loss_computation_technique':
                    example_augmentation_choices = [hyperparameter_value]
                    continue
                elif hyperparameter_name == 'use_lookahead':
                    use_lookahead = [hyperparameter_value]
                    continue
                elif hyperparameter_name == 'use_swa':
                    use_swa = [hyperparameter_value]
                    continue
                elif hyperparameter_name == 'use_se':
                    use_se = [hyperparameter_value]
                    continue
                elif hyperparameter_name == 'use_adversarial_training':
                    use_adversarial_training = [hyperparameter_value]
                    continue

                if hyperparameter_name in invalid_arguments_for_hpo_ss_updates:
                    continue

                if hyperparameter_name == 'shapedresnet:multi_branch_regularization':
                    new_hpo_search_space.append(
                        node_name="NetworkSelector",
                        hyperparameter="shapedresnet:multibranch_choices",
                        value_range=[hyperparameter_value],
                    )
                    continue

                new_hpo_search_space.append(
                    node_name=hyperparameter_node_name,
                    hyperparameter=hyperparameter_name,
                    value_range=[hyperparameter_value],
                )

            # switch search spaces
            search_space_updates = new_hpo_search_space

        else:
            print("No hpo run for the current network. Continuing without loading a hp configuration")

        autonet = AutoNetClassification(
            run_preset,
            random_seed=seed,
            run_id=f'{run_id}{seed}',
            task_id=args.array_id,
            categorical_features=categorical_indicator,
            min_workers=args.nr_workers,
            dataset_name=dataset.name,
            working_dir=seed_exp_dir,
            batch_loss_computation_techniques=example_augmentation_choices,
            use_lookahead=use_lookahead,
            use_swa=use_swa,
            use_se=use_se,
            use_adversarial_training=use_adversarial_training,
            hyperparameter_search_space_updates=search_space_updates,
            result_logger_dir=seed_exp_dir,
            torch_num_threads=args.num_threads,
            cuda=False,
            additional_logs=[test_result.__name__],
        )

        autonet.pipeline[autonet_nodes.LogFunctionsSelector.get_name()].add_log_function(
            name= test_result.__name__,
            log_function=test_result(autonet, X_test, y_test),
            loss_transform=False,
        )

        # Get the current configuration as dict
        current_configuration = autonet.get_current_autonet_config()
        print(current_configuration)
        # Get the ConfigSpace object with all hyperparameters, conditions, default values and default ranges
        hyperparameter_search_space = autonet.get_hyperparameter_search_space()
        print("Hyperparameter search space:")
        print(hyperparameter_search_space)
        # Print all possible configuration options
        autonet.print_help()

        results_fit = autonet.fit(
            X_train=X_train,
            Y_train=y_train,
            refit=False,
        )
        time.sleep(30)
        info = results_fit['info']
        train_curve = info[0]['train_balanced_accuracy']
        test_curve = info[0]['test_result']
        test_accuracy = test_curve[-1]

        train_curves.append(train_curve)
        test_curves.append(test_curve)
        test_accuracies.append(test_accuracy)

    train_mean_curve = np.mean(train_curves, 0)
    train_max_bound = np.max(train_curves, 0)
    train_min_bound = np.min(train_curves, 0)

    test_mean_curve = np.mean(test_curves, 0)
    test_max_bound = np.max(test_curves, 0)
    test_min_bound = np.min(test_curves, 0)

    curves = dict()
    curves['train_curves'] = train_curves
    curves['test_curves'] = test_curves

    with open(os.path.join(result_directory, 'curves.txt'), "w") as file:
        json.dump(curves, file)

    mean_accuracy = np.mean(test_accuracies)
    accuracy_std = np.std(test_accuracies)
    run_results = dict()
    run_results['mean_test_bal_acc'] = mean_accuracy
    run_results['std_test_bal_acc'] = accuracy_std

    with open(os.path.join(result_directory, 'run_results.txt'), "w") as file:
        json.dump(run_results, file)

elif args.run_type == 'hpo_run':

    seed_exp_dir = os.path.join(result_directory, args.run_type, f'{args.random_seed}')
    os.makedirs(seed_exp_dir, exist_ok=True)

    X_train, X_val, y_train, y_val = train_test_split(
        X_train,
        y_train,
        test_size=0.25,
        random_state=args.random_seed,
        stratify=y_train,
    )

    autonet = AutoNetClassification(
        run_preset,
        random_seed=args.random_seed,
        run_id=f'{run_id}{args.random_seed}',
        task_id=args.array_id,
        categorical_features=categorical_indicator,
        min_workers=args.nr_workers,
        dataset_name=dataset.name,
        working_dir=seed_exp_dir,
        batch_loss_computation_techniques=example_augmentation_choices,
        use_lookahead=[*args.use_lookahead],
        use_swa=[*args.use_swa],
        use_se=[*args.use_se],
        use_adversarial_training=[*args.use_adversarial_training],
        hyperparameter_search_space_updates=search_space_updates,
        result_logger_dir=seed_exp_dir,
        torch_num_threads=args.num_threads,
        cuda=False,
        additional_logs=[test_result.__name__],
    )

    autonet.pipeline[autonet_nodes.LogFunctionsSelector.get_name()].add_log_function(
        name= test_result.__name__,
        log_function=test_result(autonet, X_test, y_test),
        loss_transform=False,
    )

    # Get the current configuration as dict
    current_configuration = autonet.get_current_autonet_config()
    print(current_configuration)
    # Get the ConfigSpace object with all hyperparameters, conditions, default values and default ranges
    hyperparameter_search_space = autonet.get_hyperparameter_search_space()
    print("Hyperparameter search space:")
    print(hyperparameter_search_space)
    # Print all possible configuration options
    autonet.print_help()

    results_fit = autonet.fit(
        X_train=X_train,
        Y_train=y_train,
        X_valid=X_val,
        Y_valid=y_val,
        refit=False,
    )
