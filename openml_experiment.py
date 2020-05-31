import argparse
import json
import os

from autoPyTorch import (
    AutoNetClassification,
    HyperparameterSearchSpaceUpdates,
)

import openml


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


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
    '--use_swa',
    help='If stochastic weight averaging should be used.',
    type=str2bool,
    nargs='?',
    const=True,
    default=False,
)
parser.add_argument(
    '--use_se',
    help='If snapshot ensembling should be used.',
    type=str2bool,
    nargs='?',
    const=True,
    default=False,
)
parser.add_argument(
    '--use_lookahead',
    help='If the lookahead optimizing technique should be used.',
    type=str2bool,
    nargs='?',
    const=True,
    default=False,
)
parser.add_argument(
    '--use_weight_decay',
    help='If weight decay regularization should be used.',
    type=str2bool,
    nargs='?',
    const=True,
    default=False,
)
parser.add_argument(
    '--use_batch_normalization',
    help='If batch normalization regularization should be used.',
    type=str2bool,
    nargs='?',
    const=True,
    default=False,
)
parser.add_argument(
    '--use_skip_connection',
    help='If skip connections should be used. Turns the network into a residual network.',
    type=str2bool,
    nargs='?',
    const=True,
    default=False,
)
parser.add_argument(
    '--use_dropout',
    help='If dropout regularization should be used.',
    type=str2bool,
    nargs='?',
    const=True,
    default=False,
)
parser.add_argument(
    '--use_shake_drop',
    help='If shake drop regularization should be used.',
    type=str2bool,
    nargs='?',
    const=True,
    default=False,
)
parser.add_argument(
    '--use_shake_shake',
    help='If shake shake regularization should be used.',
    type=str2bool,
    nargs='?',
    const=True,
    default=False,
)
parser.add_argument(
    '--use_adversarial_training',
    help='If adversarial training should be used.',
    type=str2bool,
    nargs='?',
    const=True,
    default=False,
)
parser.add_argument(
    '--example_augmentation',
    help='If methods that augment examples should be used',
    type=str,
    choices=['mixup', 'cutout', 'cutmix', 'standard'],
    default='standard',
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
    default=3,
    type=int,
)
parser.add_argument(
    '--nr_units',
    help='Number of units per layer. To be used in the fixed architecture.',
    default=64,
    type=int,
)


args = parser.parse_args()
search_space_updates = HyperparameterSearchSpaceUpdates()

# Fixed architecture space
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
    log=False,
)
search_space_updates.append(
    node_name="NetworkSelector",
    hyperparameter="shapedresnet:num_groups",
    value_range=[4],
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
    value_range=[args.use_dropout],
    log=False,
)
search_space_updates.append(
    node_name="NetworkSelector",
    hyperparameter="shapedresnet:use_shake_shake",
    value_range=[args.use_shake_shake],
    log=False,
)
search_space_updates.append(
    node_name="NetworkSelector",
    hyperparameter="shapedresnet:use_shake_drop",
    value_range=[args.use_shake_drop],
    log=False,
)
search_space_updates.append(
    node_name="NetworkSelector",
    hyperparameter="shapedresnet:use_batch_normalization",
    value_range=[args.use_batch_normalization],
    log=False,
)
search_space_updates.append(
    node_name="NetworkSelector",
    hyperparameter="shapedresnet:use_skip_connection",
    value_range=[args.use_skip_connection],
    log=False,
)

search_space_updates.append(
    node_name="OptimizerSelector",
    hyperparameter="sgd:use_weight_decay",
    value_range=[args.use_weight_decay],
    log=False,
)
search_space_updates.append(
    node_name="OptimizerSelector",
    hyperparameter="adamw:use_weight_decay",
    value_range=[args.use_weight_decay],
    log=False,
)

result_directory = os.path.join(
    args.working_dir,
    f'{args.nr_units}',
    f'{args.task_id}',
)

os.makedirs(result_directory, exist_ok=True)


task = openml.tasks.get_task(task_id=args.task_id)
dataset = task.get_dataset()
X, y, categorical_indicator, _ = dataset.get_data(
    dataset_format='array',
    target=dataset.default_target_attribute,
)

ind_train, ind_test = task.get_train_test_split_indices()
X_train, Y_train = X[ind_train], y[ind_train]
X_test, Y_test = X[ind_test], y[ind_test]

autonet = AutoNetClassification(
   'no_regularization',
    random_seed=args.random_seed,
    run_id=args.run_id,
    task_id=args.array_id,
    categorical_features=categorical_indicator,
    min_workers=args.nr_workers,
    dataset_name=dataset.name,
    working_dir=result_directory,
    batch_loss_computation_techniques=[args.example_augmentation],
    use_lookahead=[args.use_lookahead],
    use_swa=[args.use_swa],
    use_se=[args.use_se],
    use_adversarial_training=[args.use_adversarial_training],
    hyperparameter_search_space_updates=search_space_updates,
    result_logger_dir=result_directory,
)

# Get the current configuration as dict
current_configuration = autonet.get_current_autonet_config()

# Get the ConfigSpace object with all hyperparameters, conditions, default values and default ranges
hyperparameter_search_space = autonet.get_hyperparameter_search_space()
print("Hyperparameter search space:")
print(hyperparameter_search_space)
# Print all possible configuration options
autonet.print_help()

results_fit = autonet.fit(
    X_train=X_train,
    Y_train=Y_train,
    refit=True,
)

# Save fit results as json
with open(os.path.join(result_directory, 'results_fit.json'), "w") as file:
    json.dump(results_fit, file)

# See how the random configuration performs (often it just predicts 0)
score = autonet.score(X_test=X_test, Y_test=Y_test)
pred = autonet.predict(X=X_test)

print("Model prediction:", pred[0:10])
print("Accuracy score", score)

# Save fit results as json
with open(os.path.join(result_directory, 'test_score.txt'), "w") as file:
    json.dump(score, file)
