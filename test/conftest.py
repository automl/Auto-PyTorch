import logging.handlers
import os
import re
import shutil
import time

import dask
import dask.distributed

import numpy as np

import openml

import pandas as pd

import pytest

from sklearn.datasets import fetch_openml, make_classification, make_regression

import torch

from autoPyTorch.automl_common.common.utils.backend import create
from autoPyTorch.data.tabular_validator import TabularInputValidator
from autoPyTorch.datasets.tabular_dataset import TabularDataset
from autoPyTorch.utils.hyperparameter_search_space_update import HyperparameterSearchSpaceUpdates
from autoPyTorch.utils.pipeline import get_dataset_requirements


N_SAMPLES = 300


@pytest.fixture(scope="session")
def callattr_ahead_of_alltests(request):
    """
    This procedure will run at the start of the pytest session.
    It will prefetch several task that are going to be used by
    the testing face, and it does so in a robust way, until the openml
    API provides the desired resources
    """
    tasks_used = [
        146818,  # Australian
        2295,    # cholesterol
        2075,    # abalone
        2071,    # adult
        3,       # kr-vs-kp
        9981,    # cnae-9
        146821,  # car
        146822,  # Segment
        2,       # anneal
        53,      # vehicle
        5136,    # tecator
        4871,    # sensory
        4857,    # boston
        3916,    # kc1
    ]

    # Populate the cache
    # This will make the test fail immediately rather than
    # Waiting for a openml fetch timeout
    openml.populate_cache(task_ids=tasks_used)
    # Also the bunch
    for task in tasks_used:
        fetch_openml(data_id=openml.tasks.get_task(task).dataset_id,
                     return_X_y=True)
    return


def slugify(text):
    return re.sub(r'[\[\]]+', '-', text.lower())


@pytest.fixture(scope="function")
def backend(request):
    test_dir = os.path.dirname(__file__)
    tmp = slugify(os.path.join(
        test_dir, '.tmp__%s__%s' % (request.module.__name__, request.node.name)))
    output = slugify(os.path.join(
        test_dir, '.output__%s__%s' % (request.module.__name__, request.node.name)))

    for dir in (tmp, output):
        for i in range(10):
            if os.path.exists(dir):
                try:
                    shutil.rmtree(dir)
                    break
                except OSError:
                    time.sleep(1)

    # Make sure the folders we wanna create do not already exist.
    backend = create(
        tmp,
        output,
        delete_tmp_folder_after_terminate=True,
        delete_output_folder_after_terminate=True,
        prefix='autoPyTorch'
    )

    def get_finalizer(tmp_dir, output_dir):
        def session_run_at_end():
            for dir in (tmp_dir, output_dir):
                for i in range(10):
                    if os.path.exists(dir):
                        try:
                            shutil.rmtree(dir)
                            break
                        except OSError:
                            time.sleep(1)

        return session_run_at_end

    request.addfinalizer(get_finalizer(tmp, output))

    return backend


@pytest.fixture(scope="function")
def tmp_dir(request):
    return _dir_fixture('tmp', request)


@pytest.fixture(scope="function")
def output_dir(request):
    return _dir_fixture('output', request)


def _dir_fixture(dir_type, request):
    test_dir = os.path.dirname(__file__)
    dir = os.path.join(
        test_dir, '.%s__%s__%s' % (dir_type, request.module.__name__, request.node.name)
    )

    for i in range(10):
        if os.path.exists(dir):
            try:
                shutil.rmtree(dir)
                break
            except OSError:
                pass

    def get_finalizer(dir):
        def session_run_at_end():
            for i in range(10):
                if os.path.exists(dir):
                    try:
                        shutil.rmtree(dir)
                        break
                    except OSError:
                        time.sleep(1)

        return session_run_at_end

    request.addfinalizer(get_finalizer(dir))

    return dir


@pytest.fixture(scope="function")
def dask_client(request):
    """
    This fixture is meant to be called one per pytest session.
    The goal of this function is to create a global client at the start
    of the testing phase. We can create clients at the start of the
    session (this case, as above scope is session), module, class or function
    level.
    The overhead of creating a dask client per class/module/session is something
    that travis cannot handle, so we rely on the following execution flow:
    1- At the start of the pytest session, session_run_at_beginning fixture is called
    to create a global client on port 4567.
    2- Any test that needs a client, would query the global scheduler that allows
    communication through port 4567.
    3- At the end of the test, we shutdown any remaining work being done by any worker
    in the client. This has a maximum 10 seconds timeout. The client object will afterwards
    be empty and when pytest closes, it can safely delete the object without hanging.
    More info on this file can be found on:
    https://docs.pytest.org/en/stable/writing_plugins.html#conftest-py-plugins
    """
    dask.config.set({'distributed.worker.daemon': False})

    client = dask.distributed.Client(n_workers=2, threads_per_worker=1, processes=False)

    def get_finalizer(address):
        def session_run_at_end():
            client = dask.distributed.get_client(address)
            client.shutdown()
            client.close()
            del client

        return session_run_at_end

    request.addfinalizer(get_finalizer(client.scheduler_info()['address']))

    return client


def get_tabular_data(task):
    if task == "classification_numerical_only":
        X, y = make_classification(
            n_samples=N_SAMPLES,
            n_features=4,
            n_informative=3,
            n_redundant=1,
            n_repeated=0,
            n_classes=2,
            n_clusters_per_class=2,
            shuffle=True,
            random_state=0
        )
        validator = TabularInputValidator(is_classification=True).fit(X.copy(), y.copy())

    elif task == "classification_categorical_only":
        X, y = fetch_openml(data_id=40981, return_X_y=True, as_frame=True)
        categorical_columns = [column for column in X.columns if X[column].dtype.name == 'category']
        X = X[categorical_columns]
        X = X.iloc[0:N_SAMPLES]
        y = y.iloc[0:N_SAMPLES]
        validator = TabularInputValidator(is_classification=True).fit(X.copy(), y.copy())

    elif task == "classification_numerical_and_categorical":
        X, y = fetch_openml(data_id=40981, return_X_y=True, as_frame=True)
        X = X.iloc[0:N_SAMPLES]
        y = y.iloc[0:N_SAMPLES]
        validator = TabularInputValidator(is_classification=True).fit(X.copy(), y.copy())

    elif task == "regression_numerical_only":
        X, y = make_regression(n_samples=3 * N_SAMPLES,
                               n_features=4,
                               n_informative=3,
                               n_targets=1,
                               shuffle=True,
                               random_state=0)
        y = (y - y.mean()) / y.std()
        validator = TabularInputValidator(is_classification=False).fit(X.copy(), y.copy())

    elif task == "regression_categorical_only":
        X, y = fetch_openml("boston", return_X_y=True, as_frame=True)
        categorical_columns = [column for column in X.columns if X[column].dtype.name == 'category']
        X = X[categorical_columns]

        # fill nan values for now since they are not handled properly yet
        for column in X.columns:
            if X[column].dtype.name == "category":
                X[column] = pd.Categorical(X[column],
                                           categories=list(X[column].cat.categories) + ["missing"]).fillna("missing")
            else:
                X[column] = X[column].fillna(0)

        X = X.iloc[0:N_SAMPLES]
        y = y.iloc[0:N_SAMPLES]
        y = (y - y.mean()) / y.std()
        validator = TabularInputValidator(is_classification=False).fit(X.copy(), y.copy())

    elif task == "regression_numerical_and_categorical":
        X, y = fetch_openml("boston", return_X_y=True, as_frame=True)

        # fill nan values for now since they are not handled properly yet
        for column in X.columns:
            if X[column].dtype.name == "category":
                X[column] = pd.Categorical(X[column],
                                           categories=list(X[column].cat.categories) + ["missing"]).fillna("missing")
            else:
                X[column] = X[column].fillna(0)

        X = X.iloc[0:N_SAMPLES]
        y = y.iloc[0:N_SAMPLES]
        y = (y - y.mean()) / y.std()
        validator = TabularInputValidator(is_classification=False).fit(X.copy(), y.copy())
    elif task == 'iris':
        X, y = fetch_openml("iris", return_X_y=True, as_frame=True)
        validator = TabularInputValidator(is_classification=True).fit(X.copy(), y.copy())
    else:
        raise ValueError("Unsupported task {}".format(task))

    return X, y, validator


def get_fit_dictionary(X, y, validator, backend):
    datamanager = TabularDataset(
        X=X, Y=y,
        validator=validator,
        X_test=X, Y_test=y,
    )

    info = datamanager.get_required_dataset_info()

    dataset_properties = datamanager.get_dataset_properties(get_dataset_requirements(info))
    fit_dictionary = {
        'X_train': datamanager.train_tensors[0],
        'y_train': datamanager.train_tensors[1],
        'train_indices': datamanager.splits[0][0],
        'val_indices': datamanager.splits[0][1],
        'dataset_properties': dataset_properties,
        'num_run': np.random.randint(50),
        'device': 'cpu',
        'budget_type': 'epochs',
        'epochs': 5,
        'torch_num_threads': 1,
        'early_stopping': 10,
        'working_dir': '/tmp',
        'use_tensorboard_logger': True,
        'metrics_during_training': True,
        'split_id': 0,
        'backend': backend,
        'logger_port': logging.handlers.DEFAULT_TCP_LOGGING_PORT,
    }
    backend.save_datamanager(datamanager)
    return fit_dictionary


@pytest.fixture
def fit_dictionary_tabular_dummy(request, backend):
    if request.param == "classification":
        X, y, validator = get_tabular_data("classification_numerical_only")
    elif request.param == "regression":
        X, y, validator = get_tabular_data("regression_numerical_only")
    else:
        raise ValueError("Unsupported indirect fixture {}".format(request.param))
    return get_fit_dictionary(X, y, validator, backend)


@pytest.fixture
def fit_dictionary_tabular(request, backend):
    X, y, validator = get_tabular_data(request.param)
    return get_fit_dictionary(X, y, validator, backend)


@pytest.fixture
def dataset(request):
    return request.getfixturevalue(request.param)


@pytest.fixture
def dataset_traditional_classifier_num_only():
    X, y = make_classification(
        n_samples=N_SAMPLES,
        n_features=4,
        n_informative=3,
        n_redundant=1,
        n_repeated=0,
        n_classes=2,
        n_clusters_per_class=2,
        shuffle=True,
        random_state=0
    )
    return X, y


@pytest.fixture
def dataset_traditional_classifier_categorical_only():
    X, y = fetch_openml(data_id=40981, return_X_y=True, as_frame=True)
    categorical_columns = [column for column in X.columns if X[column].dtype.name == 'category']
    X = X[categorical_columns]
    X, y = X[:N_SAMPLES].to_numpy(), y[:N_SAMPLES].to_numpy().astype(np.int)
    return X, y


@pytest.fixture
def dataset_traditional_classifier_num_categorical():
    X, y = fetch_openml(data_id=40981, return_X_y=True, as_frame=True)
    y = y.astype(np.int)
    X, y = X[:N_SAMPLES].to_numpy(), y[:N_SAMPLES].to_numpy().astype(np.int)
    return X, y


@pytest.fixture
def search_space_updates():
    updates = HyperparameterSearchSpaceUpdates()
    updates.append(node_name="imputer",
                   hyperparameter="numerical_strategy",
                   value_range=("mean", "most_frequent"),
                   default_value="mean")
    updates.append(node_name="data_loader",
                   hyperparameter="batch_size",
                   value_range=[16, 512],
                   default_value=32)
    updates.append(node_name="lr_scheduler",
                   hyperparameter="CosineAnnealingLR:T_max",
                   value_range=[50, 60],
                   default_value=55)
    updates.append(node_name='network_backbone',
                   hyperparameter='ResNetBackbone:dropout',
                   value_range=[0, 0.5],
                   default_value=0.2)
    return updates


@pytest.fixture
def error_search_space_updates():
    updates = HyperparameterSearchSpaceUpdates()
    updates.append(node_name="imputer",
                   hyperparameter="num_str",
                   value_range=("mean", "most_frequent"),
                   default_value="mean")
    updates.append(node_name="data_loader",
                   hyperparameter="batch_size",
                   value_range=[16, 512],
                   default_value=32)
    updates.append(node_name="lr_scheduler",
                   hyperparameter="CosineAnnealingLR:T_max",
                   value_range=[50, 60],
                   default_value=55)
    updates.append(node_name='network_backbone',
                   hyperparameter='ResNetBackbone:dropout',
                   value_range=[0, 0.5],
                   default_value=0.2)
    return updates


@pytest.fixture
def loss_cross_entropy_multiclass():
    dataset_properties = {'task_type': 'tabular_classification', 'output_type': 'multiclass'}
    predictions = torch.randn(4, 4, requires_grad=True)
    name = 'CrossEntropyLoss'
    targets = torch.empty(4, dtype=torch.long).random_(4)
    # to ensure we have all classes in the labels
    while True:
        labels = torch.empty(20, dtype=torch.long).random_(4)
        if len(torch.unique(labels)) == 4:
            break

    return dataset_properties, predictions, name, targets, labels


@pytest.fixture
def loss_cross_entropy_binary():
    dataset_properties = {'task_type': 'tabular_classification', 'output_type': 'binary'}
    predictions = torch.randn(4, 2, requires_grad=True)
    name = 'CrossEntropyLoss'
    targets = torch.empty(4, dtype=torch.long).random_(2)
    # to ensure we have all classes in the labels
    while True:
        labels = torch.empty(20, dtype=torch.long).random_(2)
        if len(torch.unique(labels)) == 2:
            break
    return dataset_properties, predictions, name, targets, labels


@pytest.fixture
def loss_bce():
    dataset_properties = {'task_type': 'tabular_classification', 'output_type': 'binary'}
    predictions = torch.empty(4).random_(2)
    name = 'BCEWithLogitsLoss'
    targets = torch.empty(4).random_(2)
    # to ensure we have all classes in the labels
    while True:
        labels = torch.empty(20, dtype=torch.long).random_(2)
        if len(torch.unique(labels)) == 2:
            break
    return dataset_properties, predictions, name, targets, labels


@pytest.fixture
def loss_mse():
    dataset_properties = {'task_type': 'tabular_regression', 'output_type': 'continuous'}
    predictions = torch.randn(4)
    name = 'MSELoss'
    targets = torch.randn(4)
    labels = None
    return dataset_properties, predictions, name, targets, labels


@pytest.fixture
def loss_details(request):
    return request.getfixturevalue(request.param)


@pytest.fixture
def n_samples():
    return N_SAMPLES
