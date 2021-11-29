import json
import os
from datetime import datetime
from test.test_api.utils import make_dict_run_history_data
from unittest.mock import MagicMock

from ConfigSpace import ConfigurationSpace

import matplotlib.pyplot as plt

import numpy as np

import pytest

from autoPyTorch.api.base_task import BaseTask
from autoPyTorch.api.results_visualizer import PlotSettingParams, ResultsVisualizer, _get_perf_and_time
from autoPyTorch.metrics import accuracy, balanced_accuracy


@pytest.mark.parametrize('params', (
    PlotSettingParams(xscale='none', yscale='none'),
    PlotSettingParams(xscale='none', yscale='log'),
    PlotSettingParams(xscale='none', yscale='none'),
    PlotSettingParams(xscale='none', yscale='log')
))
def test_raise_value_error_in_set_plot_args(params):
    _, ax = plt.subplots(nrows=1, ncols=1)
    viz = ResultsVisualizer()

    with pytest.raises(ValueError) as excinfo:
        viz._set_plot_args(ax, params)

    assert excinfo._excinfo[0] == ValueError
    plt.close()


@pytest.mark.parametrize('params', (
    PlotSettingParams(xlim=(-100, 100), ylim=(-200, 200)),
    PlotSettingParams(xlabel='x label', ylabel='y label'),
    PlotSettingParams(xscale='log', yscale='log'),
    PlotSettingParams(legend=False, title='Title')
))
def test_set_plot_args(params):
    _, ax = plt.subplots(nrows=1, ncols=1)
    viz = ResultsVisualizer()
    viz._set_plot_args(ax, params)

    if params.xlim is not None:
        assert ax.get_xlim() == params.xlim
    if params.ylim is not None:
        assert ax.get_ylim() == params.ylim

    assert ax.xaxis.get_label()._text == ('' if params.xlabel is None else params.xlabel)
    assert ax.yaxis.get_label()._text == ('' if params.ylabel is None else params.ylabel)
    assert ax.get_title() == ('' if params.title is None else params.title)
    assert params.xscale == ax.get_xscale()
    assert params.yscale == ax.get_yscale()

    if params.legend:
        assert ax.get_legend() is not None
    else:
        assert ax.get_legend() is None

    plt.close()


@pytest.mark.parametrize('metric_name', ('unknown', 'accuracy'))
def test_raise_error_in_plot_perf_over_time_in_base_task(metric_name):
    api = BaseTask()

    if metric_name == 'unknown':
        with pytest.raises(ValueError) as excinfo:
            api.plot_perf_over_time(metric_name)
        assert excinfo._excinfo[0] == ValueError
    else:
        with pytest.raises(RuntimeError) as excinfo:
            api.plot_perf_over_time(metric_name)
        assert excinfo._excinfo[0] == RuntimeError


@pytest.mark.parametrize('metric_name', ('balanced_accuracy', 'accuracy'))
def test_plot_perf_over_time(metric_name):
    dummy_history = [{'Timestamp': datetime(2022, 1, 1), 'train_accuracy': 1, 'test_accuracy': 1}]
    api = BaseTask()
    run_history_data = json.load(open(os.path.join(os.path.dirname(__file__),
                                                   '.tmp_api/runhistory_B.json'),
                                      mode='r'))['data']
    api._results_manager.run_history = MagicMock()
    api.run_history.empty = MagicMock(return_value=False)

    # The run_history has 16 runs + 1 run interruption ==> 16 runs
    api.run_history.data = make_dict_run_history_data(run_history_data)
    api._results_manager.ensemble_performance_history = dummy_history
    api._metric = accuracy
    api.dataset_name = 'iris'
    api._scoring_functions = [accuracy, balanced_accuracy]
    api.search_space = MagicMock(spec=ConfigurationSpace)

    api.plot_perf_over_time(metric_name=metric_name)
    _, ax = plt.subplots(nrows=1, ncols=1)
    api.plot_perf_over_time(metric_name=metric_name, ax=ax)

    ans = set([
        name
        for name in [f'single train {metric_name}',
                     f'single test {metric_name}',
                     f'single opt {metric_name}',
                     f'ensemble train {metric_name}',
                     f'ensemble test {metric_name}']
        if name.startswith('single') or metric_name == api._metric.name
    ])
    legend_set = set([txt._text for txt in ax.get_legend().texts])
    assert ans == legend_set
    plt.close()


@pytest.mark.parametrize('params', (
    PlotSettingParams(xscale='none', yscale='none'),
    PlotSettingParams(xscale='none', yscale='log'),
    PlotSettingParams(xscale='log', yscale='none'),
    PlotSettingParams(yscale='log')
))
def test_raise_error_get_perf_and_time(params):
    results = np.linspace(-1, 1, 10)
    cum_times = np.linspace(0, 1, 10)

    with pytest.raises(ValueError) as excinfo:
        _get_perf_and_time(
            cum_results=results,
            cum_times=cum_times,
            plot_setting_params=params,
            worst_val=np.inf
        )

    assert excinfo._excinfo[0] == ValueError


@pytest.mark.parametrize('params', (
    PlotSettingParams(n_points=20, xscale='linear', yscale='linear'),
    PlotSettingParams(n_points=20, xscale='log', yscale='log')
))
def test_get_perf_and_time(params):
    y_min, y_max = 1e-5, 1
    results = np.linspace(y_min, y_max, 10)
    cum_times = np.linspace(y_min, y_max, 10)

    check_points, perf_by_time_step = _get_perf_and_time(
        cum_results=results,
        cum_times=cum_times,
        plot_setting_params=params,
        worst_val=np.inf
    )

    times_ans = np.linspace(
        y_min if params.xscale == 'linear' else np.log(y_min),
        y_max if params.xscale == 'linear' else np.log(y_max),
        params.n_points
    )
    times_ans = times_ans if params.xscale == 'linear' else np.exp(times_ans)
    assert np.allclose(check_points, times_ans)

    if params.xscale == 'linear':
        """
        each time step to record the result
        [1.00000000e-05, 5.26410526e-02, 1.05272105e-01, 1.57903158e-01,
         2.10534211e-01, 2.63165263e-01, 3.15796316e-01, 3.68427368e-01,
         4.21058421e-01, 4.73689474e-01, 5.26320526e-01, 5.78951579e-01,
         6.31582632e-01, 6.84213684e-01, 7.36844737e-01, 7.89475789e-01,
         8.42106842e-01, 8.94737895e-01, 9.47368947e-01, 1.00000000e+00]

        The time steps when each result was recorded
        [
            1.0000e-05,  # cover index 0 ~ 2
            1.1112e-01,  # cover index 3, 4
            2.2223e-01,  # cover index 5, 6
            3.3334e-01,  # cover index 7, 8
            4.4445e-01,  # cover index 9, 10
            5.5556e-01,  # cover index 11, 12
            6.6667e-01,  # cover index 13, 14
            7.7778e-01,  # cover index 15, 16
            8.8889e-01,  # cover index 17, 18
            1.0000e+00   # cover index 19
        ]
        Since the sequence is monotonically increasing,
        if multiple elements cover the same index, take the best.
        """
        results_ans = [r for r in results]
        results_ans = [results[0]] + results_ans + results_ans[:-1]
        results_ans = np.sort(results_ans)
    else:
        """
        each time step to record the result
        [1.00000000e-05, 1.83298071e-05, 3.35981829e-05, 6.15848211e-05,
         1.12883789e-04, 2.06913808e-04, 3.79269019e-04, 6.95192796e-04,
         1.27427499e-03, 2.33572147e-03, 4.28133240e-03, 7.84759970e-03,
         1.43844989e-02, 2.63665090e-02, 4.83293024e-02, 8.85866790e-02,
         1.62377674e-01, 2.97635144e-01, 5.45559478e-01, 1.00000000e+00]

        The time steps when each result was recorded
        [
            1.0000e-05,  # cover index 0 ~ 15
            1.1112e-01,  # cover index 16
            2.2223e-01,  # cover index 17
            3.3334e-01,  # cover index 18
            4.4445e-01,  # cover index 18
            5.5556e-01,  # cover index 19
            6.6667e-01,  # cover index 19
            7.7778e-01,  # cover index 19
            8.8889e-01,  # cover index 19
            1.0000e+00   # cover index 19
        ]
        Since the sequence is monotonically increasing,
        if multiple elements cover the same index, take the best.
        """
        results_ans = [
            *([results[0]] * 16),
            results[1],
            results[2],
            results[4],
            results[-1]
        ]

    assert np.allclose(perf_by_time_step, results_ans)
