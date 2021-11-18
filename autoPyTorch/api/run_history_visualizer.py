from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt

from autoPyTorch import metrics

from smac.runhistory.runhistory import RunHistory
from smac.tae import StatusType


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


@dataclass(frozen=True)
class PlotSettingParams:
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
        legend (bool):
            Whether to have legend in the figure.
        legend_loc (str):
            The location of the legend.
        show (bool):
            Whether to show the plot.
        args, kwargs (Any):
            Arguments for the ax.plot.
    """
    n_points: int = 20
    xscale: str = 'linear'
    yscale: str = 'linear'
    xlabel: Optional[str] = None
    ylabel: Optional[str] = None
    title: Optional[str] = None
    xlim: Optional[Tuple[float, float]] = None
    ylim: Optional[Tuple[float, float]] = None
    legend: bool = True
    legend_loc: str = 'best'
    show: bool = False
    figsize: Optional[Tuple[int, int]] = None


class ScaleChoices(Enum):
    linear = 'linear'
    log = 'log'


def _get_perf_and_time(
    results: np.ndarray,
    cum_times: np.ndarray,
    plot_setting_params: PlotSettingParams
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get the performance and time step to plot.

    Args:
        results (np.ndarray):
            The cumulated performance per evaluation.
        cum_times (np.ndarray):
            The cumulated runtime at each end of evaluations.
        plot_setting_params (PlotSettingParams):
            Parameters for the plot.

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

    n_evals, runtime_lb, runtime_ub = results.size, cum_times[0], cum_times[-1]

    if plot_setting_params.xscale == 'log':
        # Take the even time interval in the log scale and revert
        check_points = np.exp(np.linspace(np.log(runtime_lb), np.log(runtime_ub), plot_setting_params.n_points))
    else:
        check_points = np.linspace(runtime_lb, runtime_ub, plot_setting_params.n_points)

    # The worst possible value is always at the head
    perf_by_time_step = np.full_like(check_points, results[0])
    cur = 0

    for i, check_point in enumerate(check_points):
        while cur < n_evals and cum_times[cur] <= check_point:
            # Guarantee that cum_times[cur] > check_point
            # ==> cum_times[cur - 1] <= check_point
            cur += 1
        if cur:  # filter cur - 1 == -1
            # results[cur - 1] was obtained before or at the checkpoint
            # ==> The best performance up to this checkpoint
            perf_by_time_step[i] = results[cur - 1]

    if plot_setting_params.yscale == 'log' and np.any(perf_by_time_step < 0):
        raise ValueError('log scale is not available when performance metric can be negative.')

    return check_points, perf_by_time_step


def _split_perf_metric_name(perf_metric_name: str) -> Tuple[str, str, str]:
    """
    Check if the performance metric name is valid and
    split the performance name if it is valid.

    Args:
        perf_metric_name (str):
            The format of name must comply:
                `(ensemble or single)::(train, test or opt)::(metric_name)`
                e.g. `ensemble::train::accuracy`

    Returns:
        ensemble_name, inference_name, metric_name (Tuple[str, str, str]):
            The splitted names.
    """

    err_msg = 'perf_metric_name must be `(ensemble or single)::(train, test or opt)::(metric_name)`, ' \
              'e.g. `ensemble::train::accuracy`, but got '

    try:
        ensemble_name, inference_name, metric_name = perf_metric_name.split('::')
    except ValueError:
        raise ValueError(f'{err_msg}{perf_metric_name}')

    if (
        ensemble_name not in ['ensemble', 'single']
        or inference_name not in ['train', 'opt', 'test']
        or (metric_name not in metrics.CLASSIFICATION_METRICS
            and metric_name not in metrics.REGRESSION_METRICS)
    ):
        raise ValueError(f'{err_msg}{perf_metric_name}')

    return ensemble_name, inference_name, metric_name


def _check_valid_metric(inference_name: str, inference_choices: List[str], metric_name: str) -> None:
    """
    Check whether the inputs are valid.

    Args:
        inference_name (str):
            Stands which inference target metric are.
        inference_choices (str):
            The choices of possible inference names.
        metric_name (str):
            The name of metric to plot.
    """

    if inference_name not in inference_choices:
        raise ValueError(
            f'inference_name must be in {inference_choices}, but '
            f'{inference_name}'
        )
    if not hasattr(metrics, metric_name):
        raise ValueError(
            f'metric_name must be in {list(metrics.CLASSIFICATION_METRICS.keys())} '
            f'or {list(metrics.REGRESSION_METRICS.keys())}, but got {metric_name}'
        )


class RunHistoryVisualizer:
    def __init__(self, *args: Any, **kwargs: Any):
        """
        Module that realizes the visualization of the results
        obtained in the optimization.

        Attributes:
            starttime (float):
                The start of the run.
            cum_times (np.ndarray):
                The runtime for each end of the evaluation of
                each configuration
        """
        self._starttime: float = 0.0
        self._cum_times: np.ndarray = np.array([])
        self._order_by_runtime: np.ndarray = np.array([])

    @property
    def starttime(self) -> float:
        return self._starttime

    @property
    def cum_times(self) -> np.ndarray:
        return self._cum_times.copy()

    @property
    def order_by_runtime(self) -> np.ndarray:
        return self._order_by_runtime.copy()

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
        if plot_setting_params.xscale == 'log' or plot_setting_params.yscale == 'log':
            ax.grid(True, which='minor', color='gray', linestyle=':')

        ax.grid(True, which='major', color='black')

        if plot_setting_params.legend:
            ax.legend(loc=plot_setting_params.legend_loc)

        if plot_setting_params.title is not None:
            ax.set_title(plot_setting_params.title)
        if plot_setting_params.show:
            plt.show()

    def _get_results_from_perf_metric_name(
        self,
        perf_metric_name: str,
        run_history: RunHistory,
        ensemble_performance_history: List[Dict[str, Any]]
    ) -> np.ndarray:
        """
        Args:
            perf_metric_name (str):
                The format of name must comply:
                `(ensemble or single)::(train, test or opt)::(metric_name)`
                    e.g. `ensemble::train::accuracy`
            run_history (RunHistory):
                The history of the optimization from SMAC
            ensemble_performance_history (List[Dict[str, Any]]):
                The history of the ensemble optimization from SMAC.
                Its keys are `train_xxx`, `test_xxx` or `Timestamp`.

        Returns:
            results (np.ndarray):
                The extracted data from either run_history or ensemble_performance_history.
        """

        ensemble_name, inference_name, metric_name = _split_perf_metric_name(perf_metric_name)

        if ensemble_name == 'ensemble':
            results = self._extract_info_from_ensemble_history(
                ensemble_performance_history,
                metric_name=metric_name,
                inference_name=inference_name
            )
        else:
            results = self._extract_info_from_run_history(
                run_history,
                metric_name=metric_name,
                inference_name=inference_name
            )

        return results

    def _extract_info_from_run_history(
        self,
        run_history: RunHistory,
        metric_name: str,
        inference_name: str
    ) -> np.ndarray:
        """
        Extract the needed information from the running history.

        Args:
            run_history (RunHistory):
                The history of the optimization from SMAC
            inference_name (str):
                Which inference to retrieve.
                Either `train`, `opt`, i.e. validation, `test`.
            metric_name (str):
                metric_name of the target.
                The list of metric_name is available in autoPyTorch.metrics.

        Returns:
            results (np.ndarray):
                The cumulated performance corresponding to the runtime.
        """
        _check_valid_metric(
            inference_name=inference_name,
            inference_choices=['train', 'test', 'opt'],
            metric_name=metric_name
        )

        metric_cls = getattr(metrics, metric_name)
        minimization = metric_cls._sign == -1
        worst_val = metric_cls._worst_possible_result
        results: List[float] = []
        runtime: List[float] = []
        self._starttime = min([d.starttime for d in run_history.data.values() if d.starttime > 0])

        key = f'{inference_name}_loss'
        for run_value in run_history.data.values():
            if run_value.endtime == 0:  # Skip crushed runs
                continue

            runtime.append(run_value.endtime - self.starttime)
            results.append(worst_val)

            if run_value.status == StatusType.SUCCESS:
                results[-1] = run_value.additional_info[key][metric_name]

        if self.cum_times.size == 0:
            # Runtime is not necessarily ascending order originally due to the parallel computation
            self._cum_times = np.array(runtime)
            self._order_by_runtime = np.argsort(self._cum_times)
            self._cum_times = self._cum_times[self._order_by_runtime]

        _results = np.array(results)[self._order_by_runtime]

        # Calculate the cumulated results
        return np.minimum.accumulate(_results) if minimization else np.maximum.accumulate(_results)

    def _extract_info_from_ensemble_history(
        self,
        ensemble_performance_history: List[Dict[str, Any]],
        metric_name: str,
        inference_name: str
    ) -> np.ndarray:
        """
        Extract the needed information from the ensemble performance history.
        More strictly, this method will return an array with the same shape
        as self.cum_times and each element corresponds to the best performance
        up to this time.

        Args:
            ensemble_performance_history (List[Dict[str, Any]]):
                The history of the ensemble optimization from SMAC.
                Its keys are `train_xxx`, `test_xxx` or `Timestamp`.
            metric_name (str):
                metric_name of the target.
                The list of metric_name is available in autoPyTorch.metrics.
            inference_name (str):
                Which inference to retrieve.
                Either `train` or `test`.

        Returns:
            results (np.ndarray):
                The best performance at the corresponding time in second
                where the plot will happen.
        """
        _check_valid_metric(
            inference_name=inference_name,
            inference_choices=['train', 'test'],
            metric_name=metric_name
        )

        metric_cls = getattr(metrics, metric_name)
        minimization = metric_cls._sign == -1
        worst_val = metric_cls._worst_possible_result

        cur, timestep_size, results = 0, self.cum_times.size, np.full_like(self.cum_times, worst_val)
        key = [k for k in ensemble_performance_history[0].keys()
               if k.startswith(inference_name)][0]

        # Sort in the order of the timestamp
        order = np.argsort(np.argsort([datetime.timestamp(data['Timestamp']) for data in ensemble_performance_history]))
        for idx in order:
            data = ensemble_performance_history[idx]
            avail_time = datetime.timestamp(data['Timestamp']) - self.starttime
            while cur < timestep_size and self.cum_times[cur] < avail_time:
                # Guarantee that cum_times[cur] >= avail_time
                cur += 1

            # results[cur] is the closest available checkpoint after or at the avail_time
            # ==> Assign this data to that checkpoint
            # Do not assign this data to the checkpoint before avail_time
            results[min(cur, results.size - 1)] = data[key]

        # Calculate the cumulated results
        return np.minimum.accumulate(results) if minimization else np.maximum.accumulate(results)

    def _plot_individual_perf_over_time(
        self,
        ax: plt.Axes,
        results: np.ndarray,
        plot_setting_params: PlotSettingParams,
        label: Optional[str] = None,
        color: Optional[str] = None,
        *args: Any,
        **kwargs: Any
    ) -> None:
        """
        Plot the performance of the AutoPytorch over time.

        Args:
            ax (plt.Axes):
                axis to plot (subplots of matplotlib).
            results (np.ndarray):
                The performance per evaluation.
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
            results=results,
            cum_times=self.cum_times,
            plot_setting_params=plot_setting_params
        )

        ax.plot(check_points, perf_by_time_step, color=color, label=label, *args, **kwargs)

    def plot_perf_over_time(
        self,
        metric_name: str,
        plot_setting_params: PlotSettingParams,
        colors: Dict[str, Optional[str]],
        labels: Dict[str, Optional[str]],
        run_history: RunHistory,
        ensemble_performance_history: Optional[List[Dict[str, Any]]] = None,
        ax: Optional[plt.Axes] = None,
        *args: Any,
        **kwargs: Any
    ) -> None:
        """
        Plot the performance of the AutoPytorch over time.

        Args:
            metric_name (str):
                The name of metric to visualize.
                The names are available in
                    * autoPyTorch.metrics.CLASSIFICATION_METRICS
                    * autoPyTorch.metrics.REGRESSION_METRICS
            labels (Dict[str, Optional[str]]):
                The name of the plot.
            colors (Dict[str, Optional[str]]):
                Color of the plot.
            run_history (RunHistory):
                The history of the optimization from SMAC
            ensemble_performance_history (Optional[List[Dict[str, Any]]]):
                The history of the ensemble optimization from SMAC.
                Its keys are `train_xxx`, `test_xxx` or `Timestamp`.
            ax (Optional[plt.Axes]):
                axis to plot (subplots of matplotlib).
                If None, it will be created automatically.
            plot_setting_params (PlotSettingParams):
                Parameters for the plot.
            args, kwargs (Any):
                Arguments for the ax.plot.
        """
        if ax is None:
            _, ax = plt.subplots(nrows=1, ncols=1)

        # Initialize at every plot
        self._cum_times, self._order_by_runtime = np.array([]), np.array([])
        # Calculate cum_times and order_by_runtime
        self._extract_info_from_run_history(run_history, metric_name=metric_name, inference_name='train')

        for key in colors.keys():
            _label, _color = labels[key], colors[key]
            perf_metric_name = f'{key}::{metric_name}'
            ensemble_name, inference_name, _metric_name = _split_perf_metric_name(perf_metric_name)
            if metric_name != _metric_name:
                raise ValueError(
                    f'metric_name for a plot must be same, i.e. {metric_name}, '
                    f'but got {_metric_name} from {perf_metric_name}'
                )

            results = self._get_results_from_perf_metric_name(
                perf_metric_name=perf_metric_name, run_history=run_history,
                ensemble_performance_history=ensemble_performance_history
            )

            self._plot_individual_perf_over_time(
                ax=ax, results=results, plot_setting_params=plot_setting_params,
                label=_label if _label is not None else f'{ensemble_name} {inference_name} {metric_name}',
                color=_color, *args, **kwargs
            )

        self._set_plot_args(ax=ax, plot_setting_params=plot_setting_params)
