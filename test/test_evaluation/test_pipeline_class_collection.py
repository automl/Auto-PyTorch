import unittest.mock

from ConfigSpace import Configuration

import numpy as np

import pytest

import autoPyTorch.pipeline.tabular_regression
from autoPyTorch.constants import (
    IMAGE_CLASSIFICATION,
    REGRESSION_TASKS,
    TABULAR_CLASSIFICATION,
    TABULAR_REGRESSION,
    TIMESERIES_CLASSIFICATION
)
from autoPyTorch.evaluation.pipeline_class_collection import (
    DummyClassificationPipeline,
    DummyRegressionPipeline,
    MyTraditionalTabularClassificationPipeline,
    MyTraditionalTabularRegressionPipeline,
    get_default_pipeline_config,
    get_pipeline_class,
)


def test_get_default_pipeline_config():
    with pytest.raises(ValueError):
        get_default_pipeline_config(choice='fail')


@pytest.mark.parametrize('task_type', (
    TABULAR_CLASSIFICATION,
    TABULAR_REGRESSION
))
@pytest.mark.parametrize('config', (1, 'tradition'))
def test_get_pipeline_class(task_type, config):
    is_reg = task_type in REGRESSION_TASKS
    pipeline_cls = get_pipeline_class(config, task_type)
    if is_reg:
        assert 'Regression' in pipeline_cls.__mro__[0].__name__
    else:
        assert 'Classification' in pipeline_cls.__mro__[0].__name__


@pytest.mark.parametrize('config,ans', (
    (1, DummyRegressionPipeline),
    ('tradition', MyTraditionalTabularRegressionPipeline),
    (unittest.mock.Mock(spec=Configuration), autoPyTorch.pipeline.tabular_regression.TabularRegressionPipeline)
))
def test_get_pipeline_class_check_class(config, ans):
    task_type = TABULAR_REGRESSION
    pipeline_cls = get_pipeline_class(config, task_type)
    assert ans is pipeline_cls


def test_get_pipeline_class_errors():
    with pytest.raises(RuntimeError):
        get_pipeline_class(config=1.5, task_type=TABULAR_CLASSIFICATION)

    with pytest.raises(NotImplementedError):
        get_pipeline_class(config='config', task_type=IMAGE_CLASSIFICATION)

    config = unittest.mock.Mock(spec=Configuration)
    with pytest.raises(NotImplementedError):
        get_pipeline_class(config=config, task_type=TIMESERIES_CLASSIFICATION)

    # Check callable
    get_pipeline_class(config=config, task_type=IMAGE_CLASSIFICATION)
    get_pipeline_class(config=config, task_type=TABULAR_REGRESSION)


@pytest.mark.parametrize('pipeline_cls', (
    MyTraditionalTabularClassificationPipeline,
    MyTraditionalTabularRegressionPipeline
))
def test_traditional_pipelines(pipeline_cls):
    rng = np.random.RandomState()
    is_reg = (pipeline_cls == MyTraditionalTabularRegressionPipeline)
    pipeline = pipeline_cls(
        config='random_forest',
        dataset_properties={
            'numerical_columns': None,
            'categorical_columns': None
        },
        random_state=rng
    )
    # Check if it is callable
    pipeline.get_pipeline_representation()

    # fit and predict
    n_insts = 100
    X = {
        'X_train': np.random.random((n_insts, 10)),
        'y_train': np.random.random(n_insts),
        'train_indices': np.arange(n_insts // 2),
        'val_indices': np.arange(n_insts // 2, n_insts),
        'dataset_properties': {
            'task_type': 'tabular_regression' if is_reg else 'tabular_classification',
            'output_type': 'continuous' if is_reg else 'multiclass'
        }
    }
    if not is_reg:
        X['y_train'] = np.array(X['y_train'] * 3, dtype=np.int32)

    pipeline.fit(X, y=None)
    pipeline.predict(X['X_train'])

    if pipeline_cls == DummyClassificationPipeline:
        pipeline.predict_proba(X['X_train'])

    assert pipeline.get_default_pipeline_config() == get_default_pipeline_config(choice='default')
    for key in ['pipeline_configuration',
                'trainer_configuration',
                'configuration_origin']:
        assert key in pipeline.get_additional_run_info()


@pytest.mark.parametrize('pipeline_cls', (
    DummyRegressionPipeline,
    DummyClassificationPipeline
))
def test_dummy_pipelines(pipeline_cls):
    rng = np.random.RandomState()
    pipeline = pipeline_cls(
        config=1,
        random_state=rng
    )
    assert pipeline.get_additional_run_info() == {'configuration_origin': 'DUMMY'}
    assert pipeline.get_pipeline_representation() == {'Preprocessing': 'None', 'Estimator': 'Dummy'}
    assert pipeline.get_default_pipeline_config() == get_default_pipeline_config(choice='dummy')
    n_insts = 100
    X = {
        'X_train': np.random.random((n_insts, 10)),
        'y_train': np.random.random(n_insts),
        'train_indices': np.arange(n_insts // 2)
    }
    if pipeline_cls == DummyClassificationPipeline:
        X['y_train'] = np.array(X['y_train'] * 3, dtype=np.int32)

    pipeline.fit(X, y=None)
    pipeline.predict(X['X_train'])

    if pipeline_cls == DummyClassificationPipeline:
        pipeline.predict_proba(X['X_train'])
