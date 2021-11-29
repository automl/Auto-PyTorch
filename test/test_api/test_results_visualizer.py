import json
import os
from datetime import datetime
from test.test_api.utils import make_dict_run_history_data
from unittest.mock import MagicMock

from ConfigSpace import ConfigurationSpace

import matplotlib.pyplot as plt

import pytest

from autoPyTorch.api.base_task import BaseTask
from autoPyTorch.api.results_visualizer import PlotSettingParams, ResultsVisualizer
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
