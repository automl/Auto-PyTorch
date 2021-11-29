import matplotlib.pyplot as plt

import pytest

from autoPyTorch.api.results_visualizer import PlotSettingParams, ResultsVisualizer


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
