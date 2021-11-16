import json
import logging
import os
import re
import unittest
from test.test_api.utils import make_dict_run_history_data
from unittest.mock import MagicMock

from ConfigSpace.configuration_space import ConfigurationSpace

import numpy as np

import pytest

from smac.runhistory.runhistory import RunHistory
from smac.tae.serial_runner import SerialRunner

from autoPyTorch.api.base_task import BaseTask, _pipeline_predict
from autoPyTorch.constants import TABULAR_CLASSIFICATION, TABULAR_REGRESSION
from autoPyTorch.metrics import accuracy, balanced_accuracy
from autoPyTorch.pipeline.tabular_classification import TabularClassificationPipeline


# ====
# Test
# ====
@pytest.mark.parametrize("fit_dictionary_tabular", ['classification_categorical_only'], indirect=True)
def test_nonsupported_arguments(fit_dictionary_tabular):
    with pytest.raises(ValueError, match=r".*Expected search space updates to be of instance.*"):
        api = BaseTask(search_space_updates='None')

    api = BaseTask()
    with pytest.raises(ValueError, match=r".*Invalid configuration arguments given.*"):
        api.set_pipeline_config(unsupported=True)
    with pytest.raises(ValueError, match=r".*No search space initialised and no dataset.*"):
        api.get_search_space()
    api.resampling_strategy = None
    with pytest.raises(ValueError, match=r".*Resampling strategy is needed to determine.*"):
        api._load_models()
    api.resampling_strategy = unittest.mock.MagicMock()
    with pytest.raises(ValueError, match=r".*Providing a metric to AutoPytorch is required.*"):
        api._load_models()
    api.ensemble_ = unittest.mock.MagicMock()
    with pytest.raises(ValueError, match=r".*No metric found. Either fit/search has not been.*"):
        api.score(np.ones(10), np.ones(10))
    api.task_type = None
    api._metric = MagicMock()
    with pytest.raises(ValueError, match=r".*AutoPytorch failed to infer a task type*"):
        api.score(np.ones(10), np.ones(10))
    api._metric = unittest.mock.MagicMock()
    with pytest.raises(ValueError, match=r".*No valid model found in run history.*"):
        api._load_models()
    dataset = fit_dictionary_tabular['backend'].load_datamanager()
    with pytest.raises(ValueError, match=r".*Incompatible dataset entered for current task.*"):
        api._search('accuracy', dataset)

    def returnfalse():
        return False

    api._load_models = returnfalse
    with pytest.raises(ValueError, match=r".*No ensemble found. Either fit has not yet.*"):
        api.predict(np.ones((10, 10)))
    with pytest.raises(ValueError, match=r".*No ensemble found. Either fit has not yet.*"):
        api.predict(np.ones((10, 10)))


def test_pipeline_predict_function():
    X = np.ones((10, 10))
    pipeline = MagicMock()
    pipeline.predict.return_value = np.full((10,), 3)
    pipeline.predict_proba.return_value = np.full((10, 2), 3)

    # First handle the classification case
    task = TABULAR_CLASSIFICATION
    with pytest.raises(ValueError, match='prediction probability not within'):
        _pipeline_predict(pipeline, X, 5, logging.getLogger, task)
    pipeline.predict_proba.return_value = np.zeros((10, 2))
    predictions = _pipeline_predict(pipeline, X, 5, logging.getLogger(), task)
    assert np.shape(predictions) == (10, 2)

    task = TABULAR_REGRESSION
    predictions = _pipeline_predict(pipeline, X, 5, logging.getLogger(), task)
    assert np.shape(predictions) == (10,)
    # Trigger warning msg with different shape for prediction
    pipeline.predict.return_value = np.full((12,), 3)
    predictions = _pipeline_predict(pipeline, X, 5, logging.getLogger(), task)


@pytest.mark.parametrize("fit_dictionary_tabular", ['classification_categorical_only'], indirect=True)
def test_show_models(fit_dictionary_tabular):
    api = BaseTask()
    api.ensemble_ = MagicMock()
    api.models_ = [TabularClassificationPipeline(dataset_properties=fit_dictionary_tabular['dataset_properties'])]
    api.ensemble_.get_models_with_weights.return_value = [(1.0, api.models_[0])]
    # Expect the default configuration
    expected = (r"0\s+|\s+SimpleImputer,OneHotEncoder,NoScaler,NoFeaturePreprocessing\s+"
                r"|\s+no embedding,ShapedMLPBackbone,FullyConnectedHead,nn.Sequential\s+|\s+1")
    assert re.search(expected, api.show_models()) is not None


def test_search_results_sprint_statistics():
    api = BaseTask()
    run_history_data = json.load(open(os.path.join(os.path.dirname(__file__),
                                                   '.tmp_api/runhistory.json'),
                                      mode='r'))['data']
    api.run_history = MagicMock()

    api.run_history.data = make_dict_run_history_data(run_history_data)
    api._metric = accuracy
    api.dataset_name = 'iris'
    api._scoring_functions = [accuracy, balanced_accuracy]
    api.search_space = MagicMock(spec=ConfigurationSpace)
    search_results = api.search_results

    # assert that contents of search_results are of expected types
    assert isinstance(search_results['mean_test_scores'], np.ndarray)
    assert search_results['mean_test_scores'].dtype is np.dtype(np.float)
    assert isinstance(search_results['mean_fit_times'], np.ndarray)
    assert search_results['mean_fit_times'].dtype is np.dtype(np.float)
    assert isinstance(search_results['metric_accuracy'], list)
    assert search_results['metric_accuracy'][0] > 0
    assert isinstance(search_results['metric_balanced_accuracy'], list)
    assert search_results['metric_balanced_accuracy'][0] > 0
    assert isinstance(search_results['params'], list)
    assert isinstance(search_results['rank_test_scores'], np.ndarray)
    assert search_results['rank_test_scores'].dtype is np.dtype(np.int)
    assert isinstance(search_results['status'], list)
    assert isinstance(search_results['status'][0], str)
    assert isinstance(search_results['budgets'], list)
    assert isinstance(search_results['budgets'][0], float)

    assert isinstance(api.sprint_statistics(), str)


def test_set_pipeline_config():
    # checks if we can correctly change the pipeline options
    estimator = BaseTask()
    pipeline_options = {"device": "cuda",
                        "budget_type": "epochs",
                        "epochs": 51,
                        "runtime": 360}
    estimator.set_pipeline_config(**pipeline_options)
    assert pipeline_options.items() <= estimator.get_pipeline_options().items()


@pytest.mark.parametrize("fit_dictionary_tabular", ['classification_categorical_only'], indirect=True)
@pytest.mark.parametrize(
    "min_budget,max_budget,budget_type,expected", [
        (5, 75, 'epochs', {'budget_type': 'epochs', 'epochs': 75}),
        (3, 50, 'runtime', {'budget_type': 'runtime', 'runtime': 50}),
    ])
def test_pipeline_get_budget(fit_dictionary_tabular, min_budget, max_budget, budget_type, expected):
    estimator = BaseTask(task_type='tabular_classification', ensemble_size=0)

    # Fixture pipeline config
    default_pipeline_config = {
        'device': 'cpu', 'budget_type': 'epochs', 'epochs': 50, 'runtime': 3600,
        'torch_num_threads': 1, 'early_stopping': 20, 'use_tensorboard_logger': False,
        'metrics_during_training': True, 'optimize_metric': 'accuracy'
    }
    default_pipeline_config.update(expected)

    # Create pre-requisites
    dataset = fit_dictionary_tabular['backend'].load_datamanager()
    pipeline_fit = unittest.mock.Mock()

    smac = unittest.mock.Mock()
    smac.solver.runhistory = RunHistory()
    smac.solver.intensifier.traj_logger.trajectory = []
    smac.solver.tae_runner = unittest.mock.Mock(spec=SerialRunner)
    smac.solver.tae_runner.budget_type = 'epochs'
    with unittest.mock.patch('autoPyTorch.optimizer.smbo.get_smac_object') as smac_mock:
        smac_mock.return_value = smac
        estimator._search(optimize_metric='accuracy', dataset=dataset, tae_func=pipeline_fit,
                          min_budget=min_budget, max_budget=max_budget, budget_type=budget_type,
                          enable_traditional_pipeline=False,
                          total_walltime_limit=20, func_eval_time_limit_secs=10,
                          load_models=False)
        assert list(smac_mock.call_args)[1]['ta_kwargs']['pipeline_config'] == default_pipeline_config
        assert list(smac_mock.call_args)[1]['max_budget'] == max_budget
        assert list(smac_mock.call_args)[1]['initial_budget'] == min_budget
