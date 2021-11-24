import json
import os
from test.test_api.utils import make_dict_run_history_data
from unittest.mock import MagicMock

import ConfigSpace.hyperparameters as CSH
from ConfigSpace.configuration_space import Configuration, ConfigurationSpace

import numpy as np

import pytest

from smac.runhistory.runhistory import RunHistory, StatusType

from autoPyTorch.api.base_task import BaseTask
from autoPyTorch.api.results_manager import ResultsManager, STATUS2MSG, SearchResults, cost2metric
from autoPyTorch.metrics import accuracy, balanced_accuracy, log_loss


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


def _check_metric_dict(metric_dict, status_types, worst_val):
    assert isinstance(metric_dict['accuracy'], list)
    assert metric_dict['accuracy'][0] > 0
    assert isinstance(metric_dict['balanced_accuracy'], list)
    assert metric_dict['balanced_accuracy'][0] > 0

    for key, vals in metric_dict.items():
        # ^ is a XOR operator
        # True and False / False and True must be fulfilled
        assert all([(s == STATUS2MSG[StatusType.SUCCESS]) ^ np.isclose([val], [worst_val])
                    for s, val in zip(status_types, vals)])


def test_extract_results_from_run_history():
    # test the raise error for the `status_msg is None`
    run_history = RunHistory()
    cs = ConfigurationSpace()
    config = Configuration(cs, {})
    run_history.add(
        config=config,
        cost=0.0,
        time=1.0,
        status=StatusType.CAPPED,
    )
    with pytest.raises(ValueError) as excinfo:
        SearchResults(metric=accuracy, scoring_functions=[], run_history=run_history)

        assert excinfo._excinfo[0] == ValueError


def test_search_results_sprint_statistics():
    api = BaseTask()
    for method in ['get_search_results', 'sprint_statistics', 'get_incumbent_results']:
        with pytest.raises(RuntimeError) as excinfo:
            getattr(api, method)()

        assert excinfo._excinfo[0] == RuntimeError

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
    worst_val = api._metric._worst_possible_result
    search_results = api.get_search_results()

    _check_status(search_results.status_types)
    _check_costs(search_results.opt_scores)
    _check_fit_times(search_results.fit_times)
    _check_budgets(search_results.budgets)
    _check_metric_dict(search_results.metric_dict, search_results.status_types, worst_val)
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


@pytest.mark.parametrize('run_history', (None, RunHistory()))
def test_check_run_history(run_history):
    manager = ResultsManager()
    manager.run_history = run_history

    with pytest.raises(RuntimeError) as excinfo:
        manager._check_run_history()

    assert excinfo._excinfo[0] == RuntimeError


T, NT = 'traditional', 'non-traditional'
SCORES = [0.1 * (i + 1) for i in range(10)]


@pytest.mark.parametrize('include_traditional', (True, False))
@pytest.mark.parametrize('metric', (accuracy, log_loss))
@pytest.mark.parametrize('origins', ([T] * 5 + [NT] * 5, [T, NT] * 5, [NT] * 5 + [T] * 5))
@pytest.mark.parametrize('scores', (SCORES, SCORES[::-1]))
def test_get_incumbent_results(include_traditional, metric, origins, scores):
    manager = ResultsManager()
    cs = ConfigurationSpace()
    cs.add_hyperparameter(CSH.UniformFloatHyperparameter('a', lower=0, upper=1))

    configs = [0.1 * (i + 1) for i in range(len(scores))]
    if metric.name == "log_loss":
        # This is to detect mis-computation in reversion
        metric._optimum = 0.1

    best_cost, best_idx = np.inf, -1
    for idx, (a, origin, score) in enumerate(zip(configs, origins, scores)):
        config = Configuration(cs, {'a': a})

        # conversion defined in:
        # autoPyTorch/pipeline/components/training/metrics/utils.py::calculate_loss
        cost = metric._optimum - metric._sign * score
        manager.run_history.add(
            config=config,
            cost=cost,
            time=1.0,
            status=StatusType.SUCCESS,
            additional_info={'opt_loss': {metric.name: score},
                             'configuration_origin': origin}
        )
        if cost > best_cost:
            continue

        if include_traditional:
            best_cost, best_idx = cost, idx
        elif origin != T:
            best_cost, best_idx = cost, idx

    incumbent_config, incumbent_results = manager.get_incumbent_results(
        metric=metric,
        include_traditional=include_traditional
    )

    assert isinstance(incumbent_config, Configuration)
    assert isinstance(incumbent_results, dict)
    best_score, best_a = scores[best_idx], configs[best_idx]
    assert np.allclose(
        [best_score, best_score, best_a],
        [cost2metric(best_cost, metric),
         incumbent_results['opt_loss'][metric.name],
         incumbent_config['a']]
    )

    if not include_traditional:
        assert incumbent_results['configuration_origin'] != T
