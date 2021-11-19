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

from smac.runhistory.runhistory import RunHistory, StatusType
from smac.tae.serial_runner import SerialRunner

from autoPyTorch.api.base_task import BaseTask, _pipeline_predict
from autoPyTorch.api.results_manager import STATUS2MSG
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


def _check_status(status):
    """ Based on runhistory_B.json """
    ans = [
        STATUS2MSG[StatusType.SUCCESS], STATUS2MSG[StatusType.SUCCESS],
        STATUS2MSG[StatusType.SUCCESS], STATUS2MSG[StatusType.SUCCESS],
        STATUS2MSG[StatusType.SUCCESS], STATUS2MSG[StatusType.SUCCESS],
        STATUS2MSG[StatusType.CRASHED], STATUS2MSG[StatusType.SUCCESS],
        STATUS2MSG[StatusType.SUCCESS], STATUS2MSG[StatusType.SUCCESS],
        STATUS2MSG[StatusType.SUCCESS], STATUS2MSG[StatusType.SUCCESS],
        STATUS2MSG[StatusType.SUCCESS], STATUS2MSG[StatusType.SUCCESS],
        STATUS2MSG[StatusType.TIMEOUT], STATUS2MSG[StatusType.TIMEOUT],
    ]
    assert isinstance(status, list)
    assert isinstance(status[0], str)
    assert status == ans


def _check_costs(costs):
    """ Based on runhistory_B.json """
    ans = [0.15204678362573099, 0.4444444444444444, 0.5555555555555556, 0.29824561403508776,
           0.4444444444444444, 0.4444444444444444, 1.0, 0.5555555555555556, 0.4444444444444444,
           0.15204678362573099, 0.15204678362573099, 0.4035087719298246, 0.4444444444444444,
           0.4444444444444444, 1.0, 1.0]
    assert np.allclose(1 - np.array(costs), ans)
    assert isinstance(costs, np.ndarray)
    assert costs.dtype is np.dtype(np.float)


def _check_fit_times(fit_times):
    """ Based on runhistory_B.json """
    ans = [3.154788017272949, 3.2763524055480957, 22.723600149154663, 4.990685224533081, 10.684926509857178,
           9.947429180145264, 11.687273979187012, 8.478890419006348, 5.485020637512207, 11.514830589294434,
           15.370736837387085, 23.846530199050903, 6.757539510726929, 15.061991930007935, 50.010520696640015,
           22.011935234069824]

    assert np.allclose(fit_times, ans)
    assert isinstance(fit_times, np.ndarray)
    assert fit_times.dtype is np.dtype(np.float)


def _check_budgets(budgets):
    """ Based on runhistory_B.json """
    ans = [5.555555555555555, 5.555555555555555, 5.555555555555555, 5.555555555555555,
           5.555555555555555, 5.555555555555555, 5.555555555555555, 5.555555555555555,
           5.555555555555555, 16.666666666666664, 50.0, 16.666666666666664, 16.666666666666664,
           16.666666666666664, 50.0, 50.0]
    assert np.allclose(budgets, ans)
    assert isinstance(budgets, list)
    assert isinstance(budgets[0], float)


def _check_additional_infos(status_types, additional_infos):
    for i, status in enumerate(status_types):
        info = additional_infos[i]
        if status in (STATUS2MSG[StatusType.SUCCESS], STATUS2MSG[StatusType.DONOTADVANCE]):
            metric_info = info.get('opt_loss', None)
            assert metric_info is not None
        elif info is not None:
            metric_info = info.get('opt_loss', None)
            assert metric_info is None


def _check_metric_dict(metric_dict, status_types):
    assert isinstance(metric_dict['accuracy'], list)
    assert metric_dict['accuracy'][0] > 0
    assert isinstance(metric_dict['balanced_accuracy'], list)
    assert metric_dict['balanced_accuracy'][0] > 0

    for key, vals in metric_dict.items():
        # ^ is a XOR operator
        # True and False / False and True must be fulfilled
        assert all([(s == STATUS2MSG[StatusType.SUCCESS]) ^ isnan
                    for s, isnan in zip(status_types, np.isnan(vals))])


def test_search_results_sprint_statistics():
    api = BaseTask()
    run_history_data = json.load(open(os.path.join(os.path.dirname(__file__),
                                                   '.tmp_api/runhistory_B.json'),
                                      mode='r'))['data']
    api._results_manager.run_history = MagicMock()
    api.run_history.empty = MagicMock(return_value=False)

    # The run_history has 16 runs + 1 run interruption ==> 16 runs
    api.run_history.data = make_dict_run_history_data(run_history_data)
    api._metric = accuracy
    api.dataset_name = 'iris'
    api._scoring_functions = [accuracy, balanced_accuracy]
    api.search_space = MagicMock(spec=ConfigurationSpace)
    search_results = api.get_search_results()

    _check_status(search_results.status_types)
    _check_costs(search_results.opt_scores)
    _check_fit_times(search_results.fit_times)
    _check_budgets(search_results.budgets)
    _check_metric_dict(search_results.metric_dict, search_results.status_types)
    _check_additional_infos(status_types=search_results.status_types,
                            additional_infos=search_results.additional_infos)

    # config_ids can duplicate because of various budget size
    config_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 1, 10, 11, 12, 10, 13]
    assert config_ids == search_results.config_ids

    # assert that contents of search_results are of expected types
    assert isinstance(search_results.rank_test_scores, np.ndarray)
    assert search_results.rank_test_scores.dtype is np.dtype(np.int)
    assert isinstance(search_results.configs, list)

    n_success, n_timeout, n_memoryout, n_crashed = 13, 2, 0, 1
    msg = ["autoPyTorch results:", f"\tDataset name: {api.dataset_name}",
           f"\tOptimisation Metric: {api._metric.name}",
           f"\tBest validation score: {max(search_results.opt_scores)}",
           "\tNumber of target algorithm runs: 16", f"\tNumber of successful target algorithm runs: {n_success}",
           f"\tNumber of crashed target algorithm runs: {n_crashed}",
           f"\tNumber of target algorithms that exceeded the time limit: {n_timeout}",
           f"\tNumber of target algorithms that exceeded the memory limit: {n_memoryout}"]

    assert isinstance(api.sprint_statistics(), str)
    assert all([m1 == m2 for m1, m2 in zip(api.sprint_statistics().split("\n"), msg)])


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
