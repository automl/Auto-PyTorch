import datetime
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

from scipy import sparse

from sklearn.datasets import fetch_openml, make_classification, make_regression
from sklearn.utils import check_random_state

import torch

from autoPyTorch.automl_common.common.utils.backend import create
from autoPyTorch.data.tabular_validator import TabularInputValidator
from autoPyTorch.data.time_series_forecasting_validator import TimeSeriesForecastingInputValidator
from autoPyTorch.datasets.tabular_dataset import TabularDataset
from autoPyTorch.datasets.time_series_dataset import TimeSeriesForecastingDataset
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
        2295,    # cholesterol
        3916,    # kc1-binary
        293554,  # reuters
        294846   # rf1
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
def loss_mape():
    dataset_properties = {'task_type': 'time_series_forecasting', 'output_type': 'continuous'}
    predictions = torch.randn(4)
    name = 'MAPELoss'
    targets = torch.randn(4)
    labels = None
    return dataset_properties, predictions, name, targets, labels


@pytest.fixture
def loss_details(request):
    return request.getfixturevalue(request.param)


@pytest.fixture
def n_samples():
    return N_SAMPLES


# Fixtures for input validators. By default all elements have 100 datapoints
@pytest.fixture
def input_data_featuretest(request):
    if request.param == 'numpy_categoricalonly_nonan':
        return np.random.randint(10, size=(100, 10))
    elif request.param == 'numpy_numericalonly_nonan':
        return np.random.uniform(10, size=(100, 10))
    elif request.param == 'numpy_mixed_nonan':
        return np.column_stack([
            np.random.uniform(10, size=(100, 3)),
            np.random.randint(10, size=(100, 3)),
            np.random.uniform(10, size=(100, 3)),
            np.random.randint(10, size=(100, 1)),
        ])
    elif request.param == 'numpy_string_nonan':
        return np.array([
            ['a', 'b', 'c', 'a', 'b', 'c'],
            ['a', 'b', 'd', 'r', 'b', 'c'],
        ])
    elif request.param == 'numpy_categoricalonly_nan':
        array = np.random.randint(10, size=(100, 10)).astype('float')
        array[50, 0:5] = np.nan
        return array
    elif request.param == 'numpy_numericalonly_nan':
        array = np.full(fill_value=10.0, shape=(100, 10), dtype=np.float64)
        array[50, 0:5] = np.nan
        # Somehow array is changed to dtype object after np.nan
        return array.astype('float')
    elif request.param == 'numpy_mixed_nan':
        array = np.column_stack([
            np.random.uniform(10, size=(100, 3)),
            np.random.randint(10, size=(100, 3)),
            np.random.uniform(10, size=(100, 3)),
            np.random.randint(10, size=(100, 1)),
        ])
        array[50, 0:5] = np.nan
        return array
    elif request.param == 'numpy_string_nan':
        return np.array([
            ['a', 'b', 'c', 'a', 'b', 'c'],
            [np.nan, 'b', 'd', 'r', 'b', 'c'],
        ])
    elif request.param == 'pandas_categoricalonly_nonan':
        return pd.DataFrame([
            {'A': 1, 'B': 2},
            {'A': 3, 'B': 4},
        ], dtype='category')
    elif request.param == 'pandas_numericalonly_nonan':
        return pd.DataFrame([
            {'A': 1, 'B': 2},
            {'A': 3, 'B': 4},
        ], dtype='float')
    elif request.param == 'pandas_mixed_nonan':
        frame = pd.DataFrame([
            {'A': 1, 'B': 2},
            {'A': 3, 'B': 4},
        ], dtype='category')
        frame['B'] = pd.to_numeric(frame['B'])
        return frame
    elif request.param == 'pandas_categoricalonly_nan':
        return pd.DataFrame([
            {'A': 1, 'B': 2, 'C': np.nan},
            {'A': 3, 'C': np.nan},
        ], dtype='category')
    elif request.param == 'pandas_numericalonly_nan':
        return pd.DataFrame([
            {'A': 1, 'B': 2, 'C': np.nan},
            {'A': 3, 'C': np.nan},
        ], dtype='float')
    elif request.param == 'pandas_mixed_nan':
        frame = pd.DataFrame([
            {'A': 1, 'B': 2, 'C': 8},
            {'A': 3, 'B': 4},
        ], dtype='category')
        frame['B'] = pd.to_numeric(frame['B'])
        return frame
    elif request.param == 'pandas_string_nonan':
        return pd.DataFrame([
            {'A': 1, 'B': 2},
            {'A': 3, 'B': 4},
        ], dtype='string')
    elif request.param == 'list_categoricalonly_nonan':
        return [
            ['a', 'b', 'c', 'd'],
            ['e', 'f', 'c', 'd'],
        ]
    elif request.param == 'list_numericalonly_nonan':
        return [
            [1, 2, 3, 4],
            [5, 6, 7, 8]
        ]
    elif request.param == 'list_mixed_nonan':
        return [
            ['a', 2, 3, 4],
            ['b', 6, 7, 8]
        ]
    elif request.param == 'list_categoricalonly_nan':
        return [
            ['a', 'b', 'c', np.nan],
            ['e', 'f', 'c', 'd'],
        ]
    elif request.param == 'list_numericalonly_nan':
        return [
            [1, 2, 3, np.nan],
            [5, 6, 7, 8]
        ]
    elif request.param == 'list_mixed_nan':
        return [
            ['a', np.nan, 3, 4],
            ['b', 6, 7, 8]
        ]
    elif 'sparse' in request.param:
        # We expect the names to be of the type sparse_csc_nonan
        sparse_, type_, nan_ = request.param.split('_')
        if 'nonan' in nan_:
            data = np.ones(3)
        else:
            data = np.array([1, 2, np.nan])

        # Then the type of sparse
        row_ind = np.array([0, 1, 2])
        col_ind = np.array([1, 2, 1])
        if 'csc' in type_:
            return sparse.csc_matrix((data, (row_ind, col_ind)))
        elif 'csr' in type_:
            return sparse.csr_matrix((data, (row_ind, col_ind)))
        elif 'coo' in type_:
            return sparse.coo_matrix((data, (row_ind, col_ind)))
        elif 'bsr' in type_:
            return sparse.bsr_matrix((data, (row_ind, col_ind)))
        elif 'lil' in type_:
            return sparse.lil_matrix((data))
        elif 'dok' in type_:
            return sparse.dok_matrix(np.vstack((data, data, data)))
        elif 'dia' in type_:
            return sparse.dia_matrix(np.vstack((data, data, data)))
        else:
            ValueError("Unsupported indirect fixture {}".format(request.param))
    elif 'openml' in request.param:
        _, openml_id = request.param.split('_')
        X, y = fetch_openml(data_id=int(openml_id),
                            return_X_y=True, as_frame=True)
        return X
    else:
        ValueError("Unsupported indirect fixture {}".format(request.param))


# Forecasting tasks
def get_forecasting_data(request):
    uni_variant = False
    with_missing_values = False
    type_X = 'pd'
    with_series_id = False
    if request == 'uni_variant_wo_missing':
        uni_variant = True
    elif request == 'uni_variant_w_missing':
        uni_variant = True
        with_missing_values = True
    elif request == 'multi_variant_wo_missing':
        with_missing_values = False
    elif request == 'multi_variant_w_missing':
        with_missing_values = True

    generator = check_random_state(0)
    n_seq = 10
    base_length = 50
    targets = []

    start_times = []
    # the first character indicates the type of the feature:
    # n: numerical, c: categorical, s: static
    # for categorical features, the following character indicate how the feature is stored:
    # s: stored as string; n: stored as
    if type_X == 'pd':
        if 'only_cat' in request:
            feature_columns = ['cs2_10', 'cn4_5']
        elif 'only_num' in request:
            feature_columns = ['n1', 'n3', 'n5']
        else:
            feature_columns = ['n1', 'cs2_10', 'n3', 'cn4_5', 'n5']
    else:
        if 'only_cat' in request:
            feature_columns = ['cn2_5', 'cn4_5']
        elif 'only_num' in request:
            feature_columns = ['n1', 'n3', 'n5']
        else:
            feature_columns = ['n1', 'cn2_5', 'n3', 'cn4_5', 'n5']

    def generate_forecasting_features(feature_type, length):
        feature_type_content = list(feature_type)
        if feature_type_content[0] == 'n':
            # numerical features
            return generator.rand(length)
        elif feature_type_content[0] == 'c':
            num_class = int(feature_type.split("_")[-1])
            if feature_type_content[1] == 's':
                return generator.choice([f'value_{feature_id}' for feature_id in range(num_class)],
                                        size=length, replace=True)
            elif feature_type_content[1] == 'n':
                return generator.choice(list(range(num_class)), size=length, replace=True)
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError

    features = []
    for i in range(n_seq):
        new_seq = np.arange(i * 1000, base_length + i * 1010).astype(np.float)
        series_length = base_length + i * 10

        targets.append(np.arange(i * 1000, series_length + i * 1000))
        if not uni_variant:
            if type_X == 'np':
                feature = np.asarray([generate_forecasting_features(col, series_length) for col in feature_columns])
            elif type_X == 'pd':
                feature = {col: generate_forecasting_features(col, series_length) for col in feature_columns}
                if with_series_id:
                    feature["series_id"] = [i] * series_length
                feature = pd.DataFrame(
                    feature
                )

                for col in feature.columns:
                    if col.startswith("n"):
                        feature[col] = feature[col].astype('float')
                    elif col.startswith("cs"):
                        feature[col] = feature[col].astype('category')
                    elif col.startswith("cn"):
                        feature[col] = feature[col].astype('int')
            else:
                raise NotImplementedError
            features.append(feature)

        if with_missing_values:
            new_seq[5] = np.NAN
            new_seq[-5] = np.NAN

        start_time = datetime.datetime.strptime(f'190{i // 5}-01-01 00-00-00', '%Y-%m-%d %H-%M-%S')
        start_times.append(start_time)
    input_validator = TimeSeriesForecastingInputValidator(is_classification=False)
    features = features if len(features) > 0 else None
    return features, targets, input_validator.fit(features, targets, start_times=start_times)


def get_forecasting_datamangaer(X, y, validator, with_y_test=True, forecast_horizon=3, freq='1D'):
    if X is not None:
        X_test = []
        for x in X:
            if hasattr(x, 'iloc'):
                X_test.append(x.iloc[-forecast_horizon:].copy())
            else:
                X_test.append(x[-forecast_horizon:].copy())
        known_future_features = tuple(X[0].columns) if isinstance(X[0], pd.DataFrame) else \
            np.arange(X[0].shape[-1]).tolist()
    else:
        X_test = None
        known_future_features = None

    if with_y_test:
        y_test = []
        for y_seq in y:
            if hasattr(y_seq, 'iloc'):
                y_test.append(y_seq.iloc[-forecast_horizon:].copy() + 1)
            else:
                y_test.append(y_seq[-forecast_horizon:].copy() + 1)
    else:
        y_test = None
    datamanager = TimeSeriesForecastingDataset(
        X=X, Y=y,
        X_test=X_test,
        Y_test=y_test,
        validator=validator,
        freq=freq,
        n_prediction_steps=forecast_horizon,
        known_future_features=known_future_features
    )
    return datamanager


def get_forecasting_fit_dictionary(datamanager, backend, forecasting_budgets='epochs'):
    info = datamanager.get_required_dataset_info()

    dataset_properties = datamanager.get_dataset_properties(get_dataset_requirements(info))

    fit_dictionary = {
        'X_train': datamanager.train_tensors[0],
        'y_train': datamanager.train_tensors[1],
        'dataset_properties': dataset_properties,
        # Training configuration
        'num_run': 1,
        'working_dir': './tmp/example_ensemble_1',  # Hopefully generated by backend
        'device': 'cpu',
        'torch_num_threads': 1,
        'early_stopping': 1,
        'use_tensorboard_logger': False,
        'use_pynisher': False,
        'metrics_during_training': False,
        'seed': 1,
        'budget_type': 'epochs',
        'epochs': 1,
        'split_id': 0,
        'backend': backend,
        'logger_port': logging.handlers.DEFAULT_TCP_LOGGING_PORT,
    }
    if forecasting_budgets == 'epochs':
        fit_dictionary.update({'forecasting_budgets': 'epochs',
                               'epochs': 1})
    elif forecasting_budgets == 'resolution':
        fit_dictionary.update({'forecasting_budgets': 'resolution',
                               'sample_interval': 2})
    elif forecasting_budgets == 'num_sample_per_seq':
        fit_dictionary.update({'forecasting_budgets': 'num_sample_per_seq',
                               'fraction_samples_per_seq': 0.5})
    elif forecasting_budgets == 'num_seq':
        fit_dictionary.update({'forecasting_budgets': 'num_seq',
                               'fraction_seq': 0.5})
    else:
        raise NotImplementedError
    backend.save_datamanager(datamanager)
    return fit_dictionary


# Fixtures for forecasting input validators
@pytest.fixture
def input_data_forecastingfeaturetest(request):
    if request.param == 'numpy_nonan':
        return np.random.uniform(10, size=(100, 10)), None, None
    elif request.param == 'numpy_with_static':
        return np.zeros([2, 3], dtype=np.int), None, None
    elif request.param == 'numpy_with_seq_length':
        return np.zeros([5, 3], dtype=np.int), None, [2, 3]
    elif request.param == 'pandas_wo_seriesid':
        return pd.DataFrame([
            {'A': 1, 'B': 2},
            {'A': 3, 'B': 4},
        ], dtype='category'), None, [2]
    elif request.param == 'pandas_w_seriesid':
        return pd.DataFrame([
            {'A': 1, 'B': 0},
            {'A': 0, 'B': 1},
        ], dtype='category'), 'A', [2]
    elif request.param == 'pandas_only_seriesid':
        return pd.DataFrame([
            {'A': 1, 'B': 0},
            {'A': 0, 'B': 1},
        ], dtype='category'), ['A', 'B'], [2]
    elif request.param == 'pandas_without_seriesid':
        return pd.DataFrame([
            {'A': 1, 'B': 2},
            {'A': 3, 'B': 4},
        ], dtype='category'), None, [2]
    elif request.param == 'pandas_with_static_features':
        return pd.DataFrame([
            {'A': 1, 'B': 2},
            {'A': 1, 'B': 4},
        ], dtype='category'), None, [2]
    elif request.param == 'pandas_multi_seq':
        return pd.DataFrame([
            {'A': 1, 'B': 2},
            {'A': 1, 'B': 4},
            {'A': 3, 'B': 2},
            {'A': 2, 'B': 4},
        ], dtype='category'), None, [2, 2]
    elif request.param == 'pandas_multi_seq_w_idx':
        return pd.DataFrame([
            {'A': 1, 'B': 2},
            {'A': 1, 'B': 4},
            {'A': 3, 'B': 2},
            {'A': 2, 'B': 4},
        ], dtype='category', index=[0, 0, 1, 1]), None, None
    elif request.param == 'pandas_with_static_features_multi_series':
        return pd.DataFrame([
            {'A': 1, 'B': 2},
            {'A': 1, 'B': 2},
            {'A': 2, 'B': 3},
            {'A': 2, 'B': 3},
        ], dtype='category'), 'A', None
    else:
        ValueError("Unsupported indirect fixture {}".format(request.param))


@pytest.fixture(scope="class")
def get_forecasting_datamanager(request):
    X, y, validator = get_forecasting_data(request.param)
    datamanager = get_forecasting_datamangaer(X, y, validator)
    return datamanager


@pytest.fixture
def forecasting_toy_dataset(request):
    x, y, _ = get_forecasting_data(request.param)
    return x, y


@pytest.fixture(params=['epochs'])
def forecasting_budgets(request):
    return request.param


@pytest.fixture
def fit_dictionary_forecasting(request, forecasting_budgets, backend):
    X, y, validator = get_forecasting_data(request.param)
    datamanager = get_forecasting_datamangaer(X, y, validator)
    return get_forecasting_fit_dictionary(datamanager, backend, forecasting_budgets=forecasting_budgets)


# Fixtures for forecasting validators.
@pytest.fixture
def input_data_forecasting_featuretest(request):
    return [input_data_featuretest(request) for _ in range(3)]
