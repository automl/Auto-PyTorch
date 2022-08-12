from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, NamedTuple, Optional, Tuple

import matplotlib.pyplot as plt

import numpy as np

from autoPyTorch.constants import OPTIONAL_INFERENCE_CHOICES
from autoPyTorch.utils.results_manager import MetricResults


plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 18


@dataclass(frozen=True)
class ColorLabelSettings:
    """
    The settings for each plot.
    If None is provided, those plots are omitted.

    Attributes:
        single_train (Optional[Tuple[Optional[str], Optional[str]]]):
            The setting for the plot of the optimal single train result.
        single_opt (Optional[Tuple[Optional[str], Optional[str]]]):
            The setting for the plot of the optimal single result used in optimization.
        single_test (Optional[Tuple[Optional[str], Optional[str]]]):
            The setting for the plot of the optimal single test result.
        ensemble_train (Optional[Tuple[Optional[str], Optional[str]]]):
            The setting for the plot of the optimal ensemble train result.
        ensemble_test (Optional[Tuple[Optional[str], Optional[str]]]):
            The setting for the plot of the optimal ensemble test result.
    """
    single_train: Optional[Tuple[Optional[str], Optional[str]]] = ('red', None)
    single_opt: Optional[Tuple[Optional[str], Optional[str]]] = ('blue', None)
    single_test: Optional[Tuple[Optional[str], Optional[str]]] = ('green', None)
    ensemble_train: Optional[Tuple[Optional[str], Optional[str]]] = ('brown', None)
    ensemble_test: Optional[Tuple[Optional[str], Optional[str]]] = ('purple', None)

    def extract_dicts(
        self,
        results: MetricResults
    ) -> Tuple[Dict[str, Optional[str]], Dict[str, Optional[str]]]:
        """
        Args:
            results (MetricResults):
                The results of the optimization in the base task API.
                It determines what keys to include.

        Returns:
            colors, labels (Tuple[Dict[str, Optional[str]], Dict[str, Optional[str]]]):
                The dicts for colors and labels.
                The keys are determined by results and each label and color
                are determined by each instantiation.
                Note that the keys include the metric name.
        """

        colors, labels = {}, {}

        for key, color_label in vars(self).items():
            if color_label is None:
                continue

            prefix = '::'.join(key.split('_'))
            try:
                new_key = [key for key in results.data.keys() if key.startswith(prefix)][0]
                colors[new_key], labels[new_key] = color_label
            except IndexError:  # ensemble does not always have results
                pass

        return colors, labels


class PlotSettingParams(NamedTuple):
    """
    Parameters for the plot environment.

    Attributes:
        n_points (int):
            The number of points to plot.
        xlabel (Optional[str]):
            The label in the x axis.
        ylabel (Optional[str]):
            The label in the y axis.
        xscale (str):
            The scale of x axis.
        yscale (str):
            The scale of y axis.
        title (Optional[str]):
            The title of the subfigure.
        xlim (Tuple[float, float]):
            The range of x axis.
        ylim (Tuple[float, float]):
            The range of y axis.
        grid (bool):
            Whether to have grid lines.
            If users would like to define lines in detail,
            they need to deactivate it.
        legend (bool):
            Whether to have legend in the figure.
        legend_kwargs (Dict[str, Any]):
            The kwargs for ax.legend.
            Ref: https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.legend.html
        title (Optional[str]):
            The title of the figure.
        title_kwargs (Dict[str, Any]):
            The kwargs for ax.set_title except title label.
            Ref: https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.axes.Axes.set_title.html
        show (bool):
            Whether to show the plot.
            If figname is not None, the save will be prioritized.
        figname (Optional[str]):
            Name of a figure to save. If None, no figure will be saved.
        savefig_kwargs (Dict[str, Any]):
            The kwargs for plt.savefig except filename.
            Ref: https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.savefig.html
        args, kwargs (Any):
            Arguments for the ax.plot.
    """
    n_points: int = 20
    xscale: str = 'linear'
    yscale: str = 'linear'
    xlabel: Optional[str] = None
    ylabel: Optional[str] = None
    title: Optional[str] = None
    title_kwargs: Dict[str, Any] = {}
    xlim: Optional[Tuple[float, float]] = None
    ylim: Optional[Tuple[float, float]] = None
    grid: bool = True
    legend: bool = True
    legend_kwargs: Dict[str, Any] = {}
    show: bool = False
    figname: Optional[str] = None
    figsize: Optional[Tuple[int, int]] = None
    savefig_kwargs: Dict[str, Any] = {}


class ScaleChoices(Enum):
    linear = 'linear'
    log = 'log'


def _get_perf_and_time(
    cum_results: np.ndarray,
    cum_times: np.ndarray,
    plot_setting_params: PlotSettingParams,
    worst_val: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get the performance and time step to plot.

    Args:
        cum_results (np.ndarray):
            The cumulated performance per evaluation.
        cum_times (np.ndarray):
            The cumulated runtime at the end of each evaluation.
        plot_setting_params (PlotSettingParams):
            Parameters for the plot.
        worst_val (float):
            The worst possible value given a metric.

    Returns:
        check_points (np.ndarray):
            The time in second where the plot will happen.
        perf_by_time_step (np.ndarray):
            The best performance at the corresponding time in second
            where the plot will happen.
    """

    scale_choices = [s.name for s in ScaleChoices]
    if plot_setting_params.xscale not in scale_choices or plot_setting_params.yscale not in scale_choices:
        raise ValueError(f'xscale and yscale must be in {scale_choices}, '
                         f'but got xscale={plot_setting_params.xscale}, yscale={plot_setting_params.yscale}')

    n_evals, runtime_lb, runtime_ub = cum_results.size, cum_times[0], cum_times[-1]

    if plot_setting_params.xscale == 'log':
        # Take the even time interval in the log scale and revert
        check_points = np.exp(np.linspace(np.log(runtime_lb), np.log(runtime_ub), plot_setting_params.n_points))
    else:
        check_points = np.linspace(runtime_lb, runtime_ub, plot_setting_params.n_points)

    check_points += 1e-8  # Prevent float error

    # The worst possible value is always at the head
    perf_by_time_step = np.full_like(check_points, worst_val)
    cur = 0

    for i, check_point in enumerate(check_points):
        while cur < n_evals and cum_times[cur] <= check_point:
            # Guarantee that cum_times[cur] > check_point
            # ==> cum_times[cur - 1] <= check_point
            cur += 1
        if cur:  # filter cur - 1 == -1
            # results[cur - 1] was obtained before or at the checkpoint
            # ==> The best performance up to this checkpoint
            perf_by_time_step[i] = cum_results[cur - 1]

    if plot_setting_params.yscale == 'log' and np.any(perf_by_time_step < 0):
        raise ValueError('log scale is not available when performance metric can be negative.')

    return check_points, perf_by_time_step


class ResultsVisualizer:
    @staticmethod
    def _set_plot_args(
        ax: plt.Axes,
        plot_setting_params: PlotSettingParams
    ) -> None:
        if plot_setting_params.xlim is not None:
            ax.set_xlim(*plot_setting_params.xlim)
        if plot_setting_params.ylim is not None:
            ax.set_ylim(*plot_setting_params.ylim)

        if plot_setting_params.xlabel is not None:
            ax.set_xlabel(plot_setting_params.xlabel)
        if plot_setting_params.ylabel is not None:
            ax.set_ylabel(plot_setting_params.ylabel)

        ax.set_xscale(plot_setting_params.xscale)
        ax.set_yscale(plot_setting_params.yscale)

        if plot_setting_params.grid:
            if plot_setting_params.xscale == 'log' or plot_setting_params.yscale == 'log':
                ax.grid(True, which='minor', color='gray', linestyle=':')

            ax.grid(True, which='major', color='black')

        if plot_setting_params.legend:
            ax.legend(**plot_setting_params.legend_kwargs)

        if plot_setting_params.title is not None:
            ax.set_title(plot_setting_params.title, **plot_setting_params.title_kwargs)

        if plot_setting_params.figname is not None:
            plt.savefig(plot_setting_params.figname, **plot_setting_params.savefig_kwargs)
        elif plot_setting_params.show:
            plt.show()

    @staticmethod
    def _plot_individual_perf_over_time(
        ax: plt.Axes,
        cum_times: np.ndarray,
        cum_results: np.ndarray,
        worst_val: float,
        plot_setting_params: PlotSettingParams,
        label: Optional[str] = None,
        color: Optional[str] = None,
        *args: Any,
        **kwargs: Any
    ) -> None:
        """
        Plot the incumbent performance of the AutoPytorch over time.
        This method is created to make plot_perf_over_time more readable
        and it is not supposed to be used only in this class, but not from outside.

        Args:
            ax (plt.Axes):
                axis to plot (subplots of matplotlib).
            cum_times (np.ndarray):
                The cumulated time until each end of config evaluation.
            results (np.ndarray):
                The cumulated performance per evaluation.
            worst_val (float):
                The worst possible value given a metric.
            plot_setting_params (PlotSettingParams):
                Parameters for the plot.
            label (Optional[str]):
                The name of the plot.
            color (Optional[str]):
                Color of the plot.
            args, kwargs (Any):
                Arguments for the ax.plot.
        """
        check_points, perf_by_time_step = _get_perf_and_time(
            cum_results=cum_results,
            cum_times=cum_times,
            plot_setting_params=plot_setting_params,
            worst_val=worst_val
        )

        ax.plot(check_points, perf_by_time_step, color=color, label=label, *args, **kwargs)

    def plot_perf_over_time(
        self,
        results: MetricResults,
        plot_setting_params: PlotSettingParams,
        colors: Dict[str, Optional[str]],
        labels: Dict[str, Optional[str]],
        ax: Optional[plt.Axes] = None,
        *args: Any,
        **kwargs: Any
    ) -> None:
        """
        Plot the incumbent performance of the AutoPytorch over time.

        Args:
            results (MetricResults):
                The module that handles results from various sources.
            plot_setting_params (PlotSettingParams):
                Parameters for the plot.
            labels (Dict[str, Optional[str]]):
                The name of the plot.
            colors (Dict[str, Optional[str]]):
                Color of the plot.
            ax (Optional[plt.Axes]):
                axis to plot (subplots of matplotlib).
                If None, it will be created automatically.
            args, kwargs (Any):
                Arguments for the ax.plot.
        """
        if ax is None:
            _, ax = plt.subplots(nrows=1, ncols=1)

        data = results.get_ensemble_merged_data()
        cum_times = results.cum_times
        minimize = (results.metric._sign == -1)

        for key in data.keys():
            inference_name = key.split('::')[1]
            _label, _color, _perfs = labels[key], colors[key], data[key]
            all_null_perfs = all([perf is None for perf in _perfs])

            if all_null_perfs:
                if inference_name not in OPTIONAL_INFERENCE_CHOICES:
                    raise ValueError(f"Expected loss for {inference_name} set to not be None")
                else:
                    continue
            # Take the best results over time
            _cum_perfs = np.minimum.accumulate(_perfs) if minimize else np.maximum.accumulate(_perfs)

            self._plot_individual_perf_over_time(  # type: ignore
                ax=ax, cum_results=_cum_perfs, cum_times=cum_times,
                plot_setting_params=plot_setting_params,
                worst_val=results.metric._worst_possible_result,
                label=_label if _label is not None else ' '.join(key.split('::')),
                color=_color,
                *args, **kwargs
            )

        self._set_plot_args(ax=ax, plot_setting_params=plot_setting_params)
