import io
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import scipy

from ConfigSpace.configuration_space import Configuration

from smac.runhistory.runhistory import RunHistory, RunValue
from smac.tae import StatusType

from autoPyTorch.pipeline.components.training.metrics.base import autoPyTorchMetric


# TODO remove StatusType.RUNNING at some point in the future when the new SMAC 0.13.2
#  is the new minimum required version!
status2msg = {
    StatusType.SUCCESS: 'Success',
    StatusType.DONOTADVANCE: 'Success (but did not advance to higher budget)',
    StatusType.TIMEOUT: 'Timeout',
    StatusType.CRASHED: 'Crash',
    StatusType.ABORT: 'Abort',
    StatusType.MEMOUT: 'Memory out'
}


class SearchResults:
    def __init__(self, scoring_functions: List[autoPyTorchMetric]):
        self.metric_dict: Dict[str, List[float]] = {
            metric.name: []
            for metric in scoring_functions
        }
        self._mean_opt_scores: List[float] = []
        self._mean_fit_time: List[float] = []
        self.params: List[Dict] = []
        self.status: List[str] = []
        self.budgets: List[float] = []
        self.rank_test_scores: np.ndarray = np.array([])

    @property
    def mean_opt_scores(self) -> np.ndarray:
        return np.asarray(self._mean_opt_scores)

    @property
    def mean_fit_time(self) -> np.ndarray:
        return np.asarray(self._mean_opt_scores)

    def update(
        self,
        param: Dict,
        status: str,
        budget: float,
        fit_time: float,
        score: float,
        metric_info: Dict[str, float]
    ) -> None:

        self.status.append(status)
        self.params.append(param)
        self.budgets.append(budget)
        self._mean_fit_time.append(fit_time)
        self._mean_opt_scores.append(score)

        for metric_name, val in metric_info.items():
            self.metric_dict[metric_name].append(val)


class ResultsManager:
    def __init__(self):
        """
        Attributes:
            run_history (RunHistory):
                A `SMAC Runshistory <https://automl.github.io/SMAC3/master/apidoc/smac.runhistory.runhistory.html>`_
                object that holds information about the runs of the target algorithm made during search
            ensemble_performance_history (List[Dict[str, Any]]):
                The list of ensemble performance in the optimization.
                The list includes the `timestamp`, `result on train set`, and `result on test set`
            trajectory (Optional[List]):
                A list of all incumbent configurations during search
        """
        self.run_history: RunHistory = RunHistory()
        self.ensemble_performance_history: List[Dict[str, Any]] = []
        self.trajectory: Optional[List] = None

    def _is_valid_run_history(self) -> bool:
        if self.run_history is None:
            raise RuntimeError("No Run History found, search has not been called.")

        if self.run_history.empty():
            raise RuntimeError("Run History is empty. Something went wrong, "
                               "SMAC was not able to fit any model?")

    def get_incumbent_results(
        self,
        include_traditional: bool = False
    ) -> Tuple[Configuration, Dict[str, Union[int, str, float]]]:
        """
        Get Incumbent config and the corresponding results

        Args:
            include_traditional (bool):
                Whether to include results from tradtional pipelines

        Returns:
            Configuration (CS.ConfigurationSpace):
                The incumbent configuration
            Dict[str, Union[int, str, float]]:
                Additional information about the run of the incumbent configuration.

        """
        self._is_valid_run_history()

        run_history_data = self.run_history.data

        if not include_traditional:
            # traditional classifiers have trainer_configuration in their additional info
            run_history_data = dict(
                filter(lambda elem: elem[1].status == StatusType.SUCCESS and elem[1].
                       additional_info is not None and elem[1].
                       additional_info['configuration_origin'] != 'traditional',
                       run_history_data.items()))

        run_history_data = dict(
            filter(lambda elem: 'SUCCESS' in str(elem[1].status), run_history_data.items()))
        sorted_runvalue_by_cost = sorted(run_history_data.items(), key=lambda item: item[1].cost)
        incumbent_run_key, incumbent_run_value = sorted_runvalue_by_cost[0]
        incumbent_config = self.run_history.ids_config[incumbent_run_key.config_id]
        incumbent_results = incumbent_run_value.additional_info
        return incumbent_config, incumbent_results

    @staticmethod
    def cost2metric(cost: float, metric: autoPyTorchMetric) -> float:
        return metric._optimum - (metric._sign * cost)

    @classmethod
    def _extract_metrics_info(
        cls,
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

        success_status = (StatusType.SUCCESS, StatusType.DONOTADVANCE)
        cost_info = run_value.additional_info['opt_loss']
        avail_metrics = cost_info.keys()

        return {
            metric.name: cls.cost2metric(cost=cost_info[metric.name], metric=metric)
            if run_value.status in success_status and metric.name in avail_metrics
            else np.nan
            for metric in scoring_functions
        }

    def _get_search_results(
        self,
        scoring_functions: Optional[List[autoPyTorchMetric]],
        metric: Optional[autoPyTorchMetric]
    ) -> SearchResults:
        """
        This attribute is populated with data from `self.run_history`
        and contains information about the configurations, and their
        corresponding metric results, status of run, parameters and
        the budget

        Args:
            scoring_functions (Optional[List[autoPyTorchMetric]]):
                Metrics to show in the results.

            metric (Optional[autoPyTorchMetric]):
                A metric that is used to fit AutoPytorch.

        Returns:
            SearchResults:
                An instance that contains the results from search
        """
        self._is_valid_run_history()

        if scoring_functions is None or metric is None:
            raise RuntimeError("`search_results` is only available after a search has finished.")

        results = SearchResults(scoring_functions)

        for run_key, run_value in self.run_history.data.items():
            config_id = run_key.config_id
            config = self.run_history.ids_config[config_id]

            status_msg = status2msg.get(run_value.status, None)
            if run_value.status in (StatusType.STOP, StatusType.RUNNING):
                continue
            elif status_msg is None:
                raise ValueError(f'Unexpected run status: {run_value.status}')

            results.update(
                status=status_msg,
                param=config.get_dictionary(),
                budget=run_key.budget,
                fit_time=run_value.time,  # To ravin: I added this line since mean_fit_time was untouched.
                score=self.cost2metric(cost=run_value.cost, metric=metric),
                metric_info=self._extract_metrics_info(run_value=run_value, scoring_functions=scoring_functions)
            )

        results.rank_test_scores = scipy.stats.rankdata(
            -1 * self._metric._sign * results.mean_opt_scores,  # rank order
            method='min'
        )

        return results

    def sprint_statistics(
        self,
        scoring_functions: Optional[List[autoPyTorchMetric]],
        metric: Optional[autoPyTorchMetric]
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
            scoring_functions (Optional[List[autoPyTorchMetric]]):
                Metrics to show in the results.

            metric (Optional[autoPyTorchMetric]):
                A metric that is used to fit AutoPytorch.

        Returns:
            (str):
                Formatted string with statistics
        """
        search_results = self._get_search_results(scoring_functions, metric)
        success_msgs = (status2msg[StatusType.SUCCESS], status2msg[StatusType.DONOTADVANCE])
        sio = io.StringIO()
        sio.write("autoPyTorch results:\n")
        sio.write(f"\tDataset name: {self.dataset_name}\n")
        sio.write(f"\tOptimisation Metric: {metric}\n")

        num_runs = len(search_results.status)
        num_success = sum([s in success_msgs for s in search_results.status])
        num_crash = sum([s == status2msg[StatusType.CRASHED] for s in search_results.status])
        num_timeout = sum([s == status2msg[StatusType.TIMEOUT] for s in search_results.status])
        num_memout = sum([s == status2msg[StatusType.MEMOUT] for s in search_results.status])

        assert metric is not None  # mypy

        if num_success > 0:
            best_score = metric._sign * np.max(metric._sign * search_results.mean_opt_scores)
            sio.write(f"\tBest validation score: {best_score}\n")

        sio.write(f"\tNumber of target algorithm runs: {num_runs}\n")
        sio.write(f"\tNumber of successful target algorithm runs: {num_success}\n")
        sio.write(f"\tNumber of crashed target algorithm runs: {num_crash}\n")
        sio.write(f"\tNumber of target algorithms that exceeded the time "
                  f"limit: {num_timeout}\n")
        sio.write(f"\tNumber of target algorithms that exceeded the memory "
                  f"limit: {num_memout}\n")

        return sio.getvalue()
