import json
import os
from datetime import datetime
from test.test_api.utils import make_dict_run_history_data
from unittest.mock import MagicMock

import ConfigSpace.hyperparameters as CSH
from ConfigSpace.configuration_space import Configuration, ConfigurationSpace

import numpy as np

import pytest

from smac.runhistory.runhistory import RunHistory, RunKey, RunValue, StatusType

from autoPyTorch.api.base_task import BaseTask
from autoPyTorch.metrics import accuracy, balanced_accuracy, log_loss
from autoPyTorch.utils.results_manager import (
    EnsembleResults,
    MetricResults,
    ResultsManager,
    SearchResults,
    cost2metric,
    get_start_time
)


T, NT = 'traditional', 'non-traditional'
SCORES = [0.1 * (i + 1) for i in range(10)]
END_TIMES = [8, 4, 3, 6, 0, 7, 1, 9, 2, 5]


def _check_status(status):
    """ Based on runhistory.json """
    ans = [
        StatusType.SUCCESS, StatusType.SUCCESS,
        StatusType.SUCCESS, StatusType.SUCCESS,
        StatusType.SUCCESS, StatusType.SUCCESS,
        StatusType.CRASHED, StatusType.SUCCESS,
        StatusType.SUCCESS, StatusType.SUCCESS,
        StatusType.SUCCESS, StatusType.SUCCESS,
        StatusType.SUCCESS, StatusType.SUCCESS,
        StatusType.TIMEOUT, StatusType.TIMEOUT,
    ]
    assert isinstance(status, list)
    assert isinstance(status[0], StatusType)
    assert status == ans


def _check_costs(costs):
    """ Based on runhistory.json """
    ans = [0.15204678362573099, 0.4444444444444444, 0.5555555555555556, 0.29824561403508776,
           0.4444444444444444, 0.4444444444444444, 1.0, 0.5555555555555556, 0.4444444444444444,
           0.15204678362573099, 0.15204678362573099, 0.4035087719298246, 0.4444444444444444,
           0.4444444444444444, 1.0, 1.0]
    assert np.allclose(1 - np.array(costs), ans)
    assert isinstance(costs, np.ndarray)
    assert costs.dtype is np.dtype(np.float)


def _check_end_times(end_times):
    """ Based on runhistory.json """
    ans = [1637342642.7887495, 1637342647.2651122, 1637342675.2555833, 1637342681.334954,
           1637342693.2717755, 1637342704.341065, 1637342726.1866672, 1637342743.3274522,
           1637342749.9442234, 1637342762.5487585, 1637342779.192385, 1637342804.3368232,
           1637342820.8067145, 1637342846.0210106, 1637342897.1205413, 1637342928.7456856]

    assert np.allclose(end_times, ans)
    assert isinstance(end_times, np.ndarray)
    assert end_times.dtype is np.dtype(np.float)


def _check_fit_times(fit_times):
    """ Based on runhistory.json """
    ans = [3.154788017272949, 3.2763524055480957, 22.723600149154663, 4.990685224533081, 10.684926509857178,
           9.947429180145264, 11.687273979187012, 8.478890419006348, 5.485020637512207, 11.514830589294434,
           15.370736837387085, 23.846530199050903, 6.757539510726929, 15.061991930007935, 50.010520696640015,
           22.011935234069824]

    assert np.allclose(fit_times, ans)
    assert isinstance(fit_times, np.ndarray)
    assert fit_times.dtype is np.dtype(np.float)


def _check_budgets(budgets):
    """ Based on runhistory.json """
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
        if status in (StatusType.SUCCESS, StatusType.DONOTADVANCE):
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
        assert all([(s == StatusType.SUCCESS) ^ np.isclose([val], [worst_val])
                    for s, val in zip(status_types, vals)])


def _check_metric_results(scores, metric, run_history, ensemble_performance_history):
    if metric.name == 'accuracy':  # Check the case when ensemble does not have the metric name
        dummy_history = [{'Timestamp': datetime(2000, 1, 1), 'train_log_loss': 1, 'test_log_loss': 1}]
        mr = MetricResults(metric, run_history, dummy_history)
        # ensemble_results should be None because ensemble evaluated log_loss
        assert mr.ensemble_results.empty()
        data = mr.get_ensemble_merged_data()
        # since ensemble_results is None, merged_data must be identical to the run_history data
        assert all(np.allclose(data[key], mr.data[key]) for key in data.keys())

    mr = MetricResults(metric, run_history, ensemble_performance_history)
    perfs = np.array([cost2metric(s, metric) for s in scores])
    modified_scores = scores[::2] + [0]
    modified_scores.insert(2, 0)
    ens_perfs = np.array([s for s in modified_scores])
    assert np.allclose(mr.data[f'single::train::{metric.name}'], perfs)
    assert np.allclose(mr.data[f'single::opt::{metric.name}'], perfs)
    assert np.allclose(mr.data[f'single::test::{metric.name}'], perfs)
    assert np.allclose(mr.data[f'ensemble::train::{metric.name}'], ens_perfs)
    assert np.allclose(mr.data[f'ensemble::test::{metric.name}'], ens_perfs)

    # the end times of synthetic ensemble is [0.25, 0.45, 0.45, 0.65, 0.85, 0.85]
    # the end times of synthetic run history is 0.1 * np.arange(1, 9) or 0.1 * np.arange(2, 10)
    ensemble_ends_later = mr.search_results.end_times[-1] < mr.ensemble_results.end_times[-1]
    indices = [2, 4, 4, 6, 8, 8] if ensemble_ends_later else [1, 3, 3, 5, 7, 7]

    merged_data = mr.get_ensemble_merged_data()
    worst_val = metric._worst_possible_result
    minimize = metric._sign == -1
    ans = np.full_like(mr.cum_times, worst_val)
    for idx, s in zip(indices, mr.ensemble_results.train_scores):
        ans[idx] = min(ans[idx], s) if minimize else max(ans[idx], s)

    assert np.allclose(ans, merged_data[f'ensemble::train::{metric.name}'])
    assert np.allclose(ans, merged_data[f'ensemble::test::{metric.name}'])


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
    with pytest.raises(ValueError):
        SearchResults(metric=accuracy, scoring_functions=[], run_history=run_history)


def test_raise_error_in_update_and_sort_by_time():
    cs = ConfigurationSpace()
    cs.add_hyperparameter(CSH.UniformFloatHyperparameter('a', lower=0, upper=1))
    config = Configuration(cs, {'a': 0.1})

    sr = SearchResults(metric=accuracy, scoring_functions=[], run_history=RunHistory())
    er = EnsembleResults(metric=accuracy, ensemble_performance_history=[])

    with pytest.raises(RuntimeError):
        sr._update(
            config=config,
            run_key=RunKey(config_id=0, instance_id=0, seed=0),
            run_value=RunValue(
                cost=0, time=1, status=StatusType.SUCCESS,
                starttime=0, endtime=1, additional_info={}
            )
        )

    with pytest.raises(RuntimeError):
        sr._sort_by_endtime()

    with pytest.raises(RuntimeError):
        er._update(data={})

    with pytest.raises(RuntimeError):
        er._sort_by_endtime()


@pytest.mark.parametrize('starttimes', (list(range(10)), list(range(10))[::-1]))
@pytest.mark.parametrize('status_types', (
    [StatusType.SUCCESS] * 9 + [StatusType.STOP],
    [StatusType.RUNNING] + [StatusType.SUCCESS] * 9
))
def test_get_start_time(starttimes, status_types):
    run_history = RunHistory()
    cs = ConfigurationSpace()
    cs.add_hyperparameter(CSH.UniformFloatHyperparameter('a', lower=0, upper=1))
    endtime = 1e9
    kwargs = dict(cost=1.0, endtime=endtime)
    for starttime, status_type in zip(starttimes, status_types):
        config = Configuration(cs, {'a': 0.1 * starttime})
        run_history.add(
            config=config,
            starttime=starttime,
            time=endtime - starttime,
            status=status_type,
            **kwargs
        )
    starttime = get_start_time(run_history)

    # this rule is strictly defined on the inputs defined from pytest
    ans = min(t for s, t in zip(status_types, starttimes) if s == StatusType.SUCCESS)
    assert starttime == ans


def test_raise_error_in_get_start_time():
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

    with pytest.raises(ValueError):
        get_start_time(run_history)


def test_search_results_sort_by_endtime():
    run_history = RunHistory()
    n_configs = len(SCORES)
    cs = ConfigurationSpace()
    cs.add_hyperparameter(CSH.UniformFloatHyperparameter('a', lower=0, upper=1))
    order = np.argsort(END_TIMES)
    ans = np.array(SCORES)[order].tolist()
    status_types = [StatusType.SUCCESS, StatusType.DONOTADVANCE] * (n_configs // 2)

    for i, (fixed_val, et, status) in enumerate(zip(SCORES, END_TIMES, status_types)):
        config = Configuration(cs, {'a': fixed_val})
        run_history.add(
            config=config, cost=fixed_val,
            status=status, budget=fixed_val,
            time=et - fixed_val, starttime=fixed_val, endtime=et,
            additional_info={
                'a': fixed_val,
                'configuration_origin': [T, NT][i % 2],
                'train_loss': {accuracy.name: fixed_val - 0.1},
                'opt_loss': {accuracy.name: fixed_val},
                'test_loss': {accuracy.name: fixed_val + 0.1}
            }
        )

    sr = SearchResults(accuracy, scoring_functions=[], run_history=run_history, order_by_endtime=True)
    assert sr.budgets == ans
    assert np.allclose(accuracy._optimum - accuracy._sign * sr.opt_scores, ans)
    assert np.allclose(accuracy._optimum - accuracy._sign * sr.train_scores, np.array(ans) - accuracy._sign * 0.1)
    assert np.allclose(accuracy._optimum - accuracy._sign * sr.test_scores, np.array(ans) + accuracy._sign * 0.1)
    assert np.allclose(1 - sr.opt_scores, ans)
    assert sr._end_times == list(range(n_configs))
    assert all(c.get('a') == val for val, c in zip(ans, sr.configs))
    assert all(info['a'] == val for val, info in zip(ans, sr.additional_infos))
    assert np.all(np.array([s for s in status_types])[order] == np.array(sr.status_types))
    assert sr.is_traditionals == np.array([True, False] * 5)[order].tolist()
    assert np.allclose(sr.fit_times, np.subtract(np.arange(n_configs), ans))


def test_ensemble_results():
    order = np.argsort(END_TIMES)
    end_times = [datetime.timestamp(datetime(2000, et + 1, 1)) for et in END_TIMES]
    ensemble_performance_history = [
        {'Timestamp': datetime(2000, et + 1, 1), 'train_accuracy': s1, 'test_accuracy': s2}
        for et, s1, s2 in zip(END_TIMES, SCORES, SCORES[::-1])
    ]

    er = EnsembleResults(log_loss, ensemble_performance_history)
    assert er.empty()

    er = EnsembleResults(accuracy, ensemble_performance_history)
    assert er._train_scores == SCORES
    assert np.allclose(er.train_scores, SCORES)
    assert er._test_scores == SCORES[::-1]
    assert np.allclose(er.test_scores, SCORES[::-1])
    assert np.allclose(er.end_times, end_times)

    er = EnsembleResults(accuracy, ensemble_performance_history, order_by_endtime=True)
    assert np.allclose(er.train_scores, np.array(SCORES)[order])
    assert np.allclose(er.test_scores, np.array(SCORES[::-1])[order])
    assert np.allclose(er.end_times, np.array(end_times)[order])


@pytest.mark.parametrize('metric', (accuracy, log_loss))
@pytest.mark.parametrize('scores', (SCORES[:8], SCORES[:8][::-1]))
@pytest.mark.parametrize('ensemble_ends_later', (True, False))
def test_metric_results(metric, scores, ensemble_ends_later):
    # since datetime --> timestamp variates between machines and float64 might not
    # be able to handle time precisely enough, we might need to change t0 in the future.
    # Basically, it happens because this test is checking by the precision of milli second
    t0, ms_unit = (1970, 1, 1, 9, 0, 0), 100000
    ensemble_performance_history = [
        {'Timestamp': datetime(*t0, ms_unit * 2 * (i + 1) + ms_unit // 2),
         f'train_{metric.name}': s,
         f'test_{metric.name}': s}
        for i, s in enumerate(scores[::2])
    ]
    # Add a record with the exact same stamp as the last one
    ensemble_performance_history.append(
        {'Timestamp': datetime(*t0, ms_unit * 8 + ms_unit // 2),
         f'train_{metric.name}': 0,
         f'test_{metric.name}': 0}
    )
    # Add a record with the exact same stamp as a middle one
    ensemble_performance_history.append(
        {'Timestamp': datetime(*t0, ms_unit * 4 + ms_unit // 2),
         f'train_{metric.name}': 0,
         f'test_{metric.name}': 0}
    )

    run_history = RunHistory()
    cs = ConfigurationSpace()
    cs.add_hyperparameter(CSH.UniformFloatHyperparameter('a', lower=0, upper=1))

    for i, fixed_val in enumerate(scores):
        config = Configuration(cs, {'a': fixed_val})
        st = datetime.timestamp(datetime(*t0, ms_unit * (i + 1 - ensemble_ends_later)))
        et = datetime.timestamp(datetime(*t0, ms_unit * (i + 2 - ensemble_ends_later)))
        run_history.add(
            config=config, cost=1, budget=0,
            time=0.1, starttime=st, endtime=et,
            status=StatusType.SUCCESS,
            additional_info={
                'configuration_origin': T,
                'train_loss': {f'{metric.name}': fixed_val},
                'opt_loss': {f'{metric.name}': fixed_val},
                'test_loss': {f'{metric.name}': fixed_val}
            }
        )
    _check_metric_results(scores, metric, run_history, ensemble_performance_history)


def test_search_results_sprint_statistics():
    BaseTask.__abstractmethods__ = set()
    api = BaseTask()
    for method in ['get_search_results', 'sprint_statistics', 'get_incumbent_results']:
        with pytest.raises(RuntimeError):
            getattr(api, method)()

    run_history_data = json.load(open(os.path.join(os.path.dirname(__file__),
                                                   'runhistory.json'),
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
    _check_end_times(search_results.end_times)
    _check_fit_times(search_results.fit_times)
    _check_budgets(search_results.budgets)
    _check_metric_dict(search_results.opt_metric_dict, search_results.status_types, worst_val)
    _check_additional_infos(status_types=search_results.status_types,
                            additional_infos=search_results.additional_infos)

    # config_ids can duplicate because of various budget size
    config_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 1, 10, 11, 12, 10, 13]
    assert config_ids == search_results.config_ids

    # assert that contents of search_results are of expected types
    assert isinstance(search_results.rank_opt_scores, np.ndarray)
    assert search_results.rank_opt_scores.dtype is np.dtype(np.int)
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


def test_search_results_sprint_statistics_no_test():
    BaseTask.__abstractmethods__ = set()
    api = BaseTask()
    for method in ['get_search_results', 'sprint_statistics', 'get_incumbent_results']:
        with pytest.raises(RuntimeError):
            getattr(api, method)()

    run_history_data = json.load(open(os.path.join(os.path.dirname(__file__),
                                                   'runhistory_no_test.json'),
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
    _check_end_times(search_results.end_times)
    _check_fit_times(search_results.fit_times)
    _check_budgets(search_results.budgets)
    _check_metric_dict(search_results.opt_metric_dict, search_results.status_types, worst_val)
    _check_additional_infos(status_types=search_results.status_types,
                            additional_infos=search_results.additional_infos)

    # config_ids can duplicate because of various budget size
    config_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 1, 10, 11, 12, 10, 13]
    assert config_ids == search_results.config_ids

    # assert that contents of search_results are of expected types
    assert isinstance(search_results.rank_opt_scores, np.ndarray)
    assert search_results.rank_opt_scores.dtype is np.dtype(np.int)
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

    with pytest.raises(RuntimeError):
        manager._check_run_history()


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
            additional_info={'train_loss': {metric.name: cost},
                             'opt_loss': {metric.name: cost},
                             'test_loss': {metric.name: cost},
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
         cost2metric(incumbent_results['opt_loss'][metric.name], metric),
         incumbent_config['a']]
    )

    if not include_traditional:
        assert incumbent_results['configuration_origin'] != T
