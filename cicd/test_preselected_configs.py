import copy
import logging.handlers
import os
import random
import tempfile
import time

import numpy as np

import openml

import pytest

import sklearn.datasets

import torch

from autoPyTorch.automl_common.common.utils.backend import create
from autoPyTorch.data.tabular_validator import TabularInputValidator
from autoPyTorch.datasets.resampling_strategy import (
    CrossValTypes,
    HoldoutValTypes,
)
from autoPyTorch.datasets.tabular_dataset import TabularDataset
from autoPyTorch.optimizer.utils import read_return_initial_configurations
from autoPyTorch.pipeline.components.training.metrics.metrics import (
    accuracy,
    balanced_accuracy,
    roc_auc,
)
from autoPyTorch.pipeline.tabular_classification import TabularClassificationPipeline
from autoPyTorch.utils.pipeline import get_dataset_requirements


def get_backend_dirs_for_openml_task(openml_task_id):
    temporary_directory = os.path.join(tempfile.gettempdir(), f"tmp_{openml_task_id}_{time.time()}")
    output_directory = os.path.join(tempfile.gettempdir(), f"out_{openml_task_id}_{time.time()}")
    return temporary_directory, output_directory


def get_fit_dictionary(openml_task_id):
    # Make sure everything from here onwards is reproducible
    # Add CUDA for future testing also
    seed = 42
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)

    task = openml.tasks.get_task(openml_task_id)
    temporary_directory, output_directory = get_backend_dirs_for_openml_task(openml_task_id)
    backend = create(
        temporary_directory=temporary_directory,
        output_directory=output_directory,
        delete_tmp_folder_after_terminate=False,
        delete_output_folder_after_terminate=False,
        prefix='autoPyTorch'
    )
    X, y = sklearn.datasets.fetch_openml(data_id=task.dataset_id, return_X_y=True, as_frame=True)
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
        X, y, random_state=seed)
    validator = TabularInputValidator(
        is_classification='classification' in task.task_type.lower()).fit(X.copy(), y.copy())
    datamanager = TabularDataset(
        dataset_name=openml.datasets.get_dataset(task.dataset_id, download_data=False).name,
        X=X_train, Y=y_train,
        validator=validator,
        X_test=X_test, Y_test=y_test,
        resampling_strategy=CrossValTypes.stratified_k_fold_cross_validation
        if 'cross' in str(task.estimation_procedure) else HoldoutValTypes.holdout_validation
    )

    info = datamanager.get_required_dataset_info()

    dataset_properties = datamanager.get_dataset_properties(get_dataset_requirements(info))
    fit_dictionary = {
        'X_train': datamanager.train_tensors[0],
        'y_train': datamanager.train_tensors[1],
        'train_indices': datamanager.splits[0][0],
        'val_indices': datamanager.splits[0][1],
        'dataset_properties': dataset_properties,
        'num_run': openml_task_id,
        'device': 'cpu',
        'budget_type': 'epochs',
        'epochs': 200,
        'torch_num_threads': 1,
        'early_stopping': 100,
        'working_dir': '/tmp',
        'use_tensorboard_logger': False,
        'metrics_during_training': True,
        'split_id': 0,
        'backend': backend,
        'logger_port': logging.handlers.DEFAULT_TCP_LOGGING_PORT,
    }
    backend.save_datamanager(datamanager)
    return fit_dictionary


@pytest.mark.parametrize(
    'openml_task_id,configuration,scorer,lower_bound_score',
    (
        # Australian
        (146818, 0, balanced_accuracy, 0.85),
        (146818, 1, roc_auc, 0.90),
        (146818, 2, balanced_accuracy, 0.80),
        (146818, 3, balanced_accuracy, 0.85),
        # credit-g
        (31, 0, accuracy, 0.75),
        (31, 1, accuracy, 0.75),
        (31, 2, accuracy, 0.75),
        (31, 3, accuracy, 0.70),
        (31, 4, accuracy, 0.70),
        # segment
        (146822, 'default', accuracy, 0.90),
        # kr-vs-kp
        (3, 'default', accuracy, 0.90),
        # vehicle
        (53, 'default', accuracy, 0.75),
    ),
)
def test_can_properly_fit_a_config(openml_task_id, configuration, scorer, lower_bound_score):

    fit_dictionary = get_fit_dictionary(openml_task_id)
    fit_dictionary['additional_metrics'] = [scorer.name]
    fit_dictionary['optimize_metric'] = scorer.name

    pipeline = TabularClassificationPipeline(
        dataset_properties=fit_dictionary['dataset_properties'])
    cs = pipeline.get_hyperparameter_search_space()
    if configuration == 'default':
        config = cs.get_default_configuration()
    else:
        # Else configuration indicates what index of the greedy config
        config = read_return_initial_configurations(
            config_space=cs,
            portfolio_selection="greedy",
        )[configuration]
    pipeline.set_hyperparameters(config)
    pipeline.fit(copy.deepcopy(fit_dictionary))

    # First we make sure performance is deterministic
    # As we use the validation performance for early stopping, this is
    # not the true generalization performance, but our goal is to test
    # that we can learn the data and capture wrong configurations

    # Sadly, when using batch norm we have results that are dependent on the current
    # torch manual seed. Set seed zero here to make this test reproducible
    torch.manual_seed(0)
    val_indices = fit_dictionary['val_indices']
    train_data, target_data = fit_dictionary['backend'].load_datamanager().train_tensors
    predictions = pipeline.predict(train_data[val_indices])
    score = scorer(fit_dictionary['y_train'][val_indices], predictions)
    assert pytest.approx(score) >= lower_bound_score

    # Check that we reverted to the best score
    run_summary = pipeline.named_steps['trainer'].run_summary

    # Then check that the training progressed nicely
    # We fit a file to have the trajectory-tendency
    # Some epochs might be bad, but overall we should make progress
    train_scores = [run_summary.performance_tracker['train_metrics'][e][scorer.name]
                    for e in range(1, len(run_summary.performance_tracker['train_metrics']) + 1)]
    slope, intersect = np.polyfit(np.arange(len(train_scores)), train_scores, 1)
    if scorer._sign > 0:
        # We expect an increasing trajectory of training
        assert train_scores[0] < train_scores[-1]
        assert slope > 0
    else:
        # We expect a decreasing trajectory of training
        assert train_scores[0] > train_scores[-1]
        assert slope < 0

    # We do not expect the network to output zeros during training.
    # We add this check to prevent a dropout bug we had, where dropout probability
    # was a bool, not a float
    network = pipeline.named_steps['network'].network
    network.train()
    global_accumulator = {}

    def forward_hook(module, X_in, X_out):
        global_accumulator[f"{id(module)}_{module.__class__.__name__}"] = torch.mean(X_out)

    for i, (hierarchy, module) in enumerate(network.named_modules()):
        module.register_forward_hook(forward_hook)
    pipeline.predict(train_data[val_indices])
    for module_name, mean_tensor in global_accumulator.items():
        # The global accumulator has the output of each layer
        # of the network. If an output of any layer is zero, this
        # check will fail
        assert mean_tensor != 0, module_name
