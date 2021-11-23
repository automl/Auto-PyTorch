import io
from typing import Any, Dict, List, Optional, Tuple, Union

from ConfigSpace.configuration_space import Configuration

import numpy as np

import scipy

from smac.runhistory.runhistory import RunHistory, RunValue
from smac.tae import StatusType
from smac.utils.io.traj_logging import TrajEntry

from autoPyTorch.pipeline.components.training.metrics.base import autoPyTorchMetric


# TODO remove StatusType.RUNNING at some point in the future when the new SMAC 0.13.2
#  is the new minimum required version!
STATUS2MSG = {
    StatusType.SUCCESS: 'Success',
    StatusType.DONOTADVANCE: 'Success (but did not advance to higher budget)',
    StatusType.TIMEOUT: 'Timeout',
    StatusType.CRASHED: 'Crash',
    StatusType.ABORT: 'Abort',
    StatusType.MEMOUT: 'Memory out'
}


def cost2metric(cost: float, metric: autoPyTorchMetric) -> float:
    """
    Revert cost metric evaluated in SMAC to the original metric.

    The conversion is defined in:
        autoPyTorch/pipeline/components/training/metrics/utils.py::calculate_loss
        cost = metric._optimum - metric._sign * original_metric_value
        ==> original_metric_value = metric._sign * (metric._optimum - cost)
    """
    return metric._sign * (metric._optimum - cost)


def _extract_metrics_info(
    run_value: RunValue,
    scoring_functions: List[autoPyTorchMetric]
) -> Dict[str, float]:
    """
    Extract the metric information given a run_value
    and a list of metrics of interest.

    Args:
        run_value (RunValue):
            The information for each config evaluation.
        scoring_functions (List[autoPyTorchMetric]):
            The list of metrics to retrieve the info.
    """

    if run_value.status not in (StatusType.SUCCESS, StatusType.DONOTADVANCE):
        # Additional info for metrics is not available in this case.
        return {metric.name: np.nan for metric in scoring_functions}

    cost_info = run_value.additional_info['opt_loss']
    avail_metrics = cost_info.keys()

    return {
        metric.name: cost2metric(cost=cost_info[metric.name], metric=metric)
        if metric.name in avail_metrics else np.nan
        for metric in scoring_functions
    }


class SearchResults:
    def __init__(
        self,
        metric: autoPyTorchMetric,
        scoring_functions: List[autoPyTorchMetric],
        run_history: RunHistory
    ):
        self.metric_dict: Dict[str, List[float]] = {
            metric.name: []
            for metric in scoring_functions
        }
        self._opt_scores: List[float] = []
        self._fit_times: List[float] = []
        self.configs: List[Configuration] = []
        self.status_types: List[str] = []
        self.budgets: List[float] = []
        self.config_ids: List[int] = []
        self.is_traditionals: List[bool] = []
        self.additional_infos: List[Optional[Dict[str, Any]]] = []
        self.rank_test_scores: np.ndarray = np.array([])
        self._scoring_functions = scoring_functions
        self._metric = metric

        self._extract_results_from_run_history(run_history)

    @property
    def opt_scores(self) -> np.ndarray:
        return np.asarray(self._opt_scores)

    @property
    def fit_times(self) -> np.ndarray:
        return np.asarray(self._fit_times)

    def update(
        self,
        config: Configuration,
        status: str,
        budget: float,
        fit_time: float,
        config_id: int,
        is_traditional: bool,
        additional_info: Dict[str, Any],
        score: float,
        metric_info: Dict[str, float]
    ) -> None:

        self.status_types.append(status)
        self.configs.append(config)
        self.budgets.append(budget)
        self.config_ids.append(config_id)
        self.is_traditionals.append(is_traditional)
        self.additional_infos.append(additional_info)
        self._fit_times.append(fit_time)
        self._opt_scores.append(score)

        for metric_name, val in metric_info.items():
            self.metric_dict[metric_name].append(val)

    def clear(self) -> None:
        self._opt_scores = []
        self._fit_times = []
        self.configs = []
        self.status_types = []
        self.budgets = []
        self.config_ids = []
        self.additional_infos = []
        self.is_traditionals = []
        self.rank_test_scores = np.array([])

    def _extract_results_from_run_history(self, run_history: RunHistory) -> None:
        """
        Extract the information to match this class format.

        Args:
            run_history (RunHistory):
                The history of config evals from SMAC.
        """

        self.clear()  # Delete cache before the extraction

        for run_key, run_value in run_history.data.items():
            config_id = run_key.config_id
            config = run_history.ids_config[config_id]

            status_msg = STATUS2MSG.get(run_value.status, None)
            if run_value.status in (StatusType.STOP, StatusType.RUNNING):
                continue
            elif status_msg is None:
                raise ValueError(f'Unexpected run status: {run_value.status}')

            is_traditional = False  # If run is not successful, unsure ==> not True ==> False
            if run_value.additional_info is not None:
                is_traditional = run_value.additional_info['configuration_origin'] == 'traditional'

            self.update(
                status=status_msg,
                config=config,
                budget=run_key.budget,
                fit_time=run_value.time,
                score=cost2metric(cost=run_value.cost, metric=self._metric),
                metric_info=_extract_metrics_info(run_value=run_value, scoring_functions=self._scoring_functions),
                is_traditional=is_traditional,
                additional_info=run_value.additional_info,
                config_id=config_id
            )

        self.rank_test_scores = scipy.stats.rankdata(
            -1 * self._metric._sign * self.opt_scores,  # rank order
            method='min'
        )


class ResultsManager:
    def __init__(self, *args: Any, **kwargs: Any):
        """
        Attributes:
            run_history (RunHistory):
                A `SMAC Runshistory <https://automl.github.io/SMAC3/master/apidoc/smac.runhistory.runhistory.html>`_
                object that holds information about the runs of the target algorithm made during search
            ensemble_performance_history (List[Dict[str, Any]]):
                The list of ensemble performance in the optimization.
                The list includes the `timestamp`, `result on train set`, and `result on test set`
            trajectory (List[TrajEntry]):
                A list of all incumbent configurations during search
        """
        self.run_history: RunHistory = RunHistory()
        self.ensemble_performance_history: List[Dict[str, Any]] = []
        self.trajectory: List[TrajEntry] = []

    def _check_run_history(self) -> None:
        if self.run_history is None:
            raise RuntimeError("No Run History found, search has not been called.")

        if self.run_history.empty():
            raise RuntimeError("Run History is empty. Something went wrong, "
                               "SMAC was not able to fit any model?")

    def get_incumbent_results(
        self,
        metric: autoPyTorchMetric,
        include_traditional: bool = False
    ) -> Tuple[Configuration, Dict[str, Union[int, str, float]]]:
        """
        Get Incumbent config and the corresponding results

        Args:
            metric (autoPyTorchMetric):
                A metric that is evaluated when searching with fit AutoPytorch.
            include_traditional (bool):
                Whether to include results from tradtional pipelines

        Returns:
            Configuration (CS.ConfigurationSpace):
                The incumbent configuration
            Dict[str, Union[int, str, float]]:
                Additional information about the run of the incumbent configuration.
        """
        self._check_run_history()

        results = SearchResults(metric=metric, scoring_functions=[], run_history=self.run_history)

        if not include_traditional:
            non_traditional = ~np.array(results.is_traditionals)
            scores = results.opt_scores[non_traditional]
            indices = np.arange(len(results.configs))[non_traditional]
        else:
            scores = results.opt_scores
            indices = np.arange(len(results.configs))

        incumbent_idx = indices[np.nanargmax(metric._sign * scores)]
        incumbent_config = results.configs[incumbent_idx]
        incumbent_results = results.additional_infos[incumbent_idx]

        assert incumbent_results is not None  # mypy check
        return incumbent_config, incumbent_results

    def get_search_results(
        self,
        scoring_functions: List[autoPyTorchMetric],
        metric: autoPyTorchMetric
    ) -> SearchResults:
        """
        This attribute is populated with data from `self.run_history`
        and contains information about the configurations, and their
        corresponding metric results, status of run, parameters and
        the budget

        Args:
            scoring_functions (List[autoPyTorchMetric]):
                Metrics to show in the results.
            metric (autoPyTorchMetric):
                A metric that is evaluated when searching with fit AutoPytorch.

        Returns:
            SearchResults:
                An instance that contains the results from search
        """
        self._check_run_history()
        return SearchResults(metric=metric, scoring_functions=scoring_functions, run_history=self.run_history)

    def sprint_statistics(
        self,
        dataset_name: str,
        scoring_functions: List[autoPyTorchMetric],
        metric: autoPyTorchMetric
    ) -> str:
        """
        Prints statistics about the SMAC search.

        These statistics include:

        1. Optimisation Metric
        2. Best Optimisation score achieved by individual pipelines
        3. Total number of target algorithm runs
        4. Total number of successful target algorithm runs
        5. Total number of crashed target algorithm runs
        6. Total number of target algorithm runs that exceeded the time limit
        7. Total number of successful target algorithm runs that exceeded the memory limit

        Args:
            dataset_name (str):
                The dataset name that was used in the run.
            scoring_functions (List[autoPyTorchMetric]):
                Metrics to show in the results.
            metric (autoPyTorchMetric):
                A metric that is evaluated when searching with fit AutoPytorch.

        Returns:
            (str):
                Formatted string with statistics
        """
        search_results = self.get_search_results(scoring_functions, metric)
        success_msgs = (STATUS2MSG[StatusType.SUCCESS], STATUS2MSG[StatusType.DONOTADVANCE])
        sio = io.StringIO()
        sio.write("autoPyTorch results:\n")
        sio.write(f"\tDataset name: {dataset_name}\n")
        sio.write(f"\tOptimisation Metric: {metric}\n")

        num_runs = len(search_results.status_types)
        num_success = sum([s in success_msgs for s in search_results.status_types])
        num_crash = sum([s == STATUS2MSG[StatusType.CRASHED] for s in search_results.status_types])
        num_timeout = sum([s == STATUS2MSG[StatusType.TIMEOUT] for s in search_results.status_types])
        num_memout = sum([s == STATUS2MSG[StatusType.MEMOUT] for s in search_results.status_types])

        if num_success > 0:
            best_score = metric._sign * np.nanmax(metric._sign * search_results.opt_scores)
            sio.write(f"\tBest validation score: {best_score}\n")

        sio.write(f"\tNumber of target algorithm runs: {num_runs}\n")
        sio.write(f"\tNumber of successful target algorithm runs: {num_success}\n")
        sio.write(f"\tNumber of crashed target algorithm runs: {num_crash}\n")
        sio.write(f"\tNumber of target algorithms that exceeded the time "
                  f"limit: {num_timeout}\n")
        sio.write(f"\tNumber of target algorithms that exceeded the memory "
                  f"limit: {num_memout}\n")

        return sio.getvalue()
