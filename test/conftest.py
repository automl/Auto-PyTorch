import os
import re
import shutil
import time

import dask
import dask.distributed

import numpy as np

import pytest

from sklearn.datasets import fetch_openml, make_classification

from autoPyTorch.datasets.tabular_dataset import TabularDataset
from autoPyTorch.utils.backend import create
from autoPyTorch.utils.hyperparameter_search_space_update import HyperparameterSearchSpaceUpdates
from autoPyTorch.utils.pipeline import get_dataset_requirements


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


# Dataset fixture to test different scenarios on a scalable way
# Please refer to https://docs.pytest.org/en/stable/fixture.html for details
# on what fixtures are
@pytest.fixture
def fit_dictionary(request):
    return request.getfixturevalue(request.param)


@pytest.fixture
def fit_dictionary_numerical_only(backend):
    X, y = make_classification(
        n_samples=200,
        n_features=8,
        n_informative=4,
        n_redundant=4,
        n_repeated=0,
        n_classes=2,
        n_clusters_per_class=2,
        shuffle=True,
        random_state=0
    )
    X = X.astype('float64')
    datamanager = TabularDataset(
        X=X, Y=y,
        X_test=X, Y_test=y,
    )

    info = datamanager.get_required_dataset_info()

    dataset_properties = datamanager.get_dataset_properties(get_dataset_requirements(info))
    fit_dictionary = {
        'X_train': X,
        'y_train': y,
        'dataset_properties': dataset_properties,
        'num_run': np.random.randint(50),
        'device': 'cpu',
        'budget_type': 'epochs',
        'epochs': 1,
        'torch_num_threads': 1,
        'early_stopping': 20,
        'working_dir': '/tmp',
        'use_tensorboard_logger': True,
        'use_pynisher': False,
        'metrics_during_training': True,
        'split_id': 0,
        'backend': backend,
    }
    backend.save_datamanager(datamanager)
    return fit_dictionary


@pytest.fixture
def fit_dictionary_categorical_only(backend):
    X, y = fetch_openml(data_id=40981, return_X_y=True, as_frame=True)
    categorical_columns = [column for column in X.columns if X[column].dtype.name == 'category']
    X = X[categorical_columns]
    X = X.iloc[0:200]
    y = y.iloc[0:200]
    datamanager = TabularDataset(
        X=X, Y=y,
        X_test=X, Y_test=y,
    )
    info = datamanager.get_required_dataset_info()

    dataset_properties = datamanager.get_dataset_properties(get_dataset_requirements(info))
    fit_dictionary = {
        'X_train': X,
        'y_train': y,
        'dataset_properties': dataset_properties,
        'num_run': np.random.randint(50),
        'device': 'cpu',
        'budget_type': 'epochs',
        'epochs': 1,
        'torch_num_threads': 1,
        'early_stopping': 20,
        'working_dir': '/tmp',
        'use_tensorboard_logger': True,
        'use_pynisher': False,
        'metrics_during_training': True,
        'split_id': 0,
        'backend': backend,
    }
    datamanager = TabularDataset(
        X=X, Y=y,
        X_test=X, Y_test=y,
    )
    backend.save_datamanager(datamanager)
    return fit_dictionary


@pytest.fixture
def fit_dictionary_num_and_categorical(backend):
    X, y = fetch_openml(data_id=40981, return_X_y=True, as_frame=True)
    X = X.iloc[0:200]
    y = y.iloc[0:200]
    datamanager = TabularDataset(
        X=X, Y=y,
        X_test=X, Y_test=y,
    )
    info = datamanager.get_required_dataset_info()

    dataset_properties = datamanager.get_dataset_properties(get_dataset_requirements(info))

    fit_dictionary = {
        'X_train': X,
        'y_train': y,
        'dataset_properties': dataset_properties,
        'num_run': np.random.randint(50),
        'device': 'cpu',
        'budget_type': 'epochs',
        'epochs': 1,
        'torch_num_threads': 1,
        'early_stopping': 20,
        'working_dir': '/tmp',
        'use_tensorboard_logger': True,
        'use_pynisher': False,
        'metrics_during_training': True,
        'split_id': 0,
        'backend': backend,
    }
    backend.save_datamanager(datamanager)
    return fit_dictionary


@pytest.fixture
def dataset(request):
    return request.getfixturevalue(request.param)


@pytest.fixture
def dataset_traditional_classifier_num_only():
    X, y = make_classification(
        n_samples=200,
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
    X, y = X[:200].to_numpy(), y[:200].to_numpy().astype(np.int)
    return X, y


@pytest.fixture
def dataset_traditional_classifier_num_categorical():
    X, y = fetch_openml(data_id=40981, return_X_y=True, as_frame=True)
    y = y.astype(np.int)
    X, y = X[:200].to_numpy(), y[:200].to_numpy().astype(np.int)
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
