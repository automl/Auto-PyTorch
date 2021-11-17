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


class ScaleChoices(Enum):
    linear = 'linear'
    log = 'log'


def _get_perf_and_time(
    results: np.ndarray,
    cum_times: np.ndarray,
    n_points: int,
    xscale: str = 'log',
    yscale: str = 'log'
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get the performance and time step to plot.

    Args:
        results (np.ndarray):
            The cumulated performance per evaluation.
        cum_times (np.ndarray):
            The cumulated runtime at each end of evaluations.
        n_points (int):
            The number of points to plot.
        xscale (str):
            The scale of x axis.
        yscale (str):
            The scale of y axis.

    Returns:
        check_points (np.ndarray):
            The time in second where the plot will happen.
        perf_by_time_step (np.ndarray):
            The best performance at the corresponding time in second
            where the plot will happen.
    """

    scale_choices = [s.name for s in ScaleChoices]
    if xscale not in scale_choices or yscale not in scale_choices:
        raise ValueError(f'xscale and yscale must be in {scale_choices}, '
                         f'but got xscale={xscale}, yscale={yscale}')

    n_evals, runtime_lb, runtime_ub = results.size, cum_times[0], cum_times[-1]

    if xscale == 'log':
        # Take the even time interval in the log scale and revert
        check_points = np.exp(np.linspace(np.log(runtime_lb), np.log(runtime_ub), n_points))
    else:
        check_points = np.linspace(np.log(runtime_lb), np.log(runtime_ub), n_points)

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

    if yscale == 'log' and np.any(perf_by_time_step < 0):
        raise ValueError('log scale is not available when performance metric can be negative.')

    return check_points, perf_by_time_step


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
        self.starttime: float = 0.0
        self.cum_times: np.ndarray = np.array([])

    def extract_info_from_run_history(
        self,
        run_history: RunHistory,
        metric_name: str,
        which_inference: str = 'opt'
    ) -> np.ndarray:
        """
        Extract the needed information from the running history.

        Args:
            run_history (RunHistory):
                The history of the optimization from SMAC
            which_inference (str):
                Which inference to retrieve.
                Either `train`, `opt`, i.e. validation, `test`.
            metric_name (str):
                metric_name of the target.
                The list of metric_name is available in autoPyTorch.metrics.

        Returns:
            results (np.ndarray):
                The cumulated performance corresponding to the runtime.
        """
        inference_choices = ['train', 'test', 'opt']
        if which_inference not in inference_choices:
            raise ValueError(
                f'which_inference must be in {inference_choices}, but '
                f'{which_inference}'
            )
        if not hasattr(metrics, metric_name):
            raise ValueError(
                f'metric_name must be in {metrics.CLASSIFICATION_METRICS} '
                f'or {metrics.REGRESSION_METRICS}, but got {metric_name}'
            )

        metric_cls = getattr(metrics, metric_name)
        minimization = metric_cls._sign == -1
        worst_val = metric_cls._worst_possible_result
        results: List[float] = []
        runtime: List[float] = []
        self.starttime = list(run_history.data.values())[0].starttime

        key = 'test_loss' if which_inference == 'test' else f'{which_inference}_loss'
        for run_value in run_history.data.values():
            runtime.append(run_value.endtime - self.starttime)
            results.append(worst_val)

            if run_value.status == StatusType.SUCCESS:
                info = run_value.additional_info[key]
                # TODO: Check what is this metric in general
                results[-1] = info if which_inference == 'test' else info[metric_name]

        self.cum_times = np.array(runtime)

        # Calculate the cumulated results
        return np.minimum.accumulate(results) if minimization else np.maximum.accumulate(results)

    def extract_info_from_ensemble_history(
        self,
        ensemble_performance_history: List[Dict[str, Any]],
        metric_name: str,
        which_inference: str = 'test'
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
            which_inference (str):
                Which inference to retrieve.
                Either `train` or `test`.

        Returns:
            results (np.ndarray):
                The best performance at the corresponding time in second
                where the plot will happen.
        """
        inference_choices = ['train', 'test']
        if which_inference not in inference_choices:
            raise ValueError(
                f'which_inference must be in {inference_choices}, but '
                f'{which_inference}'
            )
        if not hasattr(metrics, metric_name):
            raise ValueError(
                f'metric_name must be in {metrics.CLASSIFICATION_METRICS} '
                f'or {metrics.REGRESSION_METRICS}, but got {metric_name}'
            )

        metric_cls = getattr(metrics, metric_name)
        minimization = metric_cls._sign == -1
        worst_val = metric_cls._worst_possible_result

        cur, timestep_size, results = 0, self.cum_times.size, np.full_like(self.cum_times, worst_val)
        key = [k for k in ensemble_performance_history[0].keys()
               if k.startswith(which_inference)][0]

        for data in ensemble_performance_history:
            avail_time = datetime.timestamp(data['Timestamp']) - self.starttime
            while cur < timestep_size and self.cum_times[cur] < avail_time:
                # Guarantee that cum_times[cur] >= avail_time
                cur += 1

            # results[cur] is the closest available checkpoint after or at the avail_time
            # ==> Assign this data to that checkpoint
            # Do not assign this data to the checkpoint before avail_time
            results[cur] = data[key]

        # Calculate the cumulated results
        return np.minimum.accumulate(results) if minimization else np.maximum.accumulate(results)

    def plot_perf_over_time(
        self,
        ax: plt.Axes,
        results: np.ndarray,
        n_points: int = 20,
        label: Optional[str] = None,
        color: str = 'red',
        xlabel: str = 'runtime',
        ylabel: str = 'Validation loss',
        xscale: str = 'log',
        yscale: str = 'log',
        title: Optional[str] = None,
        xlim: Optional[Tuple[float, float]] = None,
        ylim: Optional[Tuple[float, float]] = None,
        show: bool = False,
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
            n_points (int):
                The number of points to plot.
            label (str):
                The name of the plot.
            color (str):
                Color of the plot.
            xlabel (str):
                The label in the x axis.
            ylabel (str):
                The label in the y axis.
            xscale (str):
                The scale of x axis.
            yscale (str):
                The scale of y axis.
            xscale (Tuple[float, float]):
                The range of x axis.
            yscale (Tuple[float, float]):
                The range of y axis.
            title (str):
                The title of the subfigure.
            show (bool):
                Whether to show the plot.
            args, kwargs (Any):
                Arguments for the ax.plot.
        """
        check_points, perf_by_time_step = _get_perf_and_time(
            results=results,
            cum_times=self.cum_times,
            n_points=n_points,
            xscale=xscale,
            yscale=yscale
        )

        ax.plot(check_points, perf_by_time_step, color=color, label=label, *args, **kwargs)

        if xlim is not None:
            ax.set_xlim(*xlim)
        if ylim is not None:
            ax.set_ylim(*ylim)

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_xscale(xscale)
        ax.set_yscale(yscale)

        if xscale == 'log' or yscale == 'log':
            ax.grid(True, which='minor', color='gray', linestyle=':')

        ax.grid(True, which='major', color='black')

        if label is not None:
            ax.legend()
        if title is not None:
            ax.set_title(title)
        if show:
            plt.show()
