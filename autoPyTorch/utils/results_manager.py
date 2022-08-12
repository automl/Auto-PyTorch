import io
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

from ConfigSpace.configuration_space import Configuration

import numpy as np

import scipy

from smac.runhistory.runhistory import RunHistory, RunKey, RunValue
from smac.tae import StatusType
from smac.utils.io.traj_logging import TrajEntry

from autoPyTorch.constants import OPTIONAL_INFERENCE_CHOICES
from autoPyTorch.pipeline.components.training.metrics.base import autoPyTorchMetric


# TODO remove StatusType.RUNNING at some point in the future when the new SMAC 0.13.2
#  is the new minimum required version!
STATUS_TYPES = [
    StatusType.SUCCESS,
    # Success (but did not advance to higher budget such as cutoff by hyperband)
    StatusType.DONOTADVANCE,
    StatusType.TIMEOUT,
    StatusType.CRASHED,
    StatusType.ABORT,
    StatusType.MEMOUT
]


def cost2metric(cost: float, metric: autoPyTorchMetric) -> float:
    """
    Revert cost metric evaluated in SMAC to the original metric.

    The conversion is defined in:
        autoPyTorch/pipeline/components/training/metrics/utils.py::calculate_loss
        cost = metric._optimum - metric._sign * original_metric_value
        ==> original_metric_value = metric._sign * (metric._optimum - cost)
    """
    return metric._sign * (metric._optimum - cost)


def get_start_time(run_history: RunHistory) -> float:
    """
    Get start time of optimization.

    Args:
        run_history (RunHistory):
            The history of config evals from SMAC.

    Returns:
        starttime (float):
            The start time of the first training.
    """

    start_times = []
    for run_value in run_history.data.values():
        if run_value.status in (StatusType.STOP, StatusType.RUNNING):
            continue
        elif run_value.status not in STATUS_TYPES:
            raise ValueError(f'Unexpected run status: {run_value.status}')

        start_times.append(run_value.starttime)

    return float(np.min(start_times))  # mypy redefinition


def _extract_metrics_info(
    run_value: RunValue,
    scoring_functions: List[autoPyTorchMetric],
    inference_name: str
) -> Dict[str, Optional[float]]:
    """
    Extract the metric information given a run_value
    and a list of metrics of interest.

    Args:
        run_value (RunValue):
            The information for each config evaluation.
        scoring_functions (List[autoPyTorchMetric]):
            The list of metrics to retrieve the info.
        inference_name (str):
            The name of the inference. Either `train`, `opt` or `test`.

    Returns:
        metric_info (Dict[str, float]):
            The metric values of interest.
            Since the metrics in additional_info are `cost`,
            we transform them into the original form.
    """

    if run_value.status not in (StatusType.SUCCESS, StatusType.DONOTADVANCE):
        # Additional info for metrics is not available in this case.
        return {metric.name: metric._worst_possible_result for metric in scoring_functions}

    inference_choices = ['train', 'opt', 'test']
    if inference_name not in inference_choices:
        raise ValueError(f'inference_name must be in {inference_choices}, but got {inference_choices}')

    cost_info = run_value.additional_info.get(f'{inference_name}_loss', None)
    if cost_info is None:
        if inference_name not in OPTIONAL_INFERENCE_CHOICES:
            raise ValueError(f"Expected loss for {inference_name} set to not be None, but got {cost_info}")
        else:
            # Additional info for metrics is not available in this case.
            return {metric.name: None for metric in scoring_functions}

    avail_metrics = cost_info.keys()

    return {
        metric.name: cost2metric(cost=cost_info[metric.name], metric=metric)
        if metric.name in avail_metrics else metric._worst_possible_result
        for metric in scoring_functions
    }


class EnsembleResults:
    def __init__(
        self,
        metric: autoPyTorchMetric,
        ensemble_performance_history: List[Dict[str, Any]],
        order_by_endtime: bool = False
    ):
        """
        The wrapper class for ensemble_performance_history.
        This class extracts the information from ensemble_performance_history
        and allows other class to easily handle the history.

        Attributes:
            train_scores (List[float]):
                The ensemble scores on the training dataset.
            test_scores (List[float]):
                The ensemble scores on the test dataset.
            end_times (List[float]):
                The end time of the end of each ensemble evaluation.
                Each element is a float timestamp.
            empty (bool):
                Whether the ensemble history about `self.metric` is empty or not.
            metric (autoPyTorchMetric):
                The information about the metric to contain.
                In the case when such a metric does not exist in the record,
                This class raises KeyError.
        """
        self._test_scores: List[float] = []
        self._train_scores: List[float] = []
        self._end_times: List[float] = []
        self._metric = metric
        self._empty = True  # Initial state is empty.
        self._instantiated = False

        self._extract_results_from_ensemble_performance_history(ensemble_performance_history)
        if order_by_endtime:
            self._sort_by_endtime()

        self._instantiated = True

    @property
    def train_scores(self) -> np.ndarray:
        return np.asarray(self._train_scores)

    @property
    def test_scores(self) -> np.ndarray:
        return np.asarray(self._test_scores)

    @property
    def end_times(self) -> np.ndarray:
        return np.asarray(self._end_times)

    @property
    def metric_name(self) -> str:
        return self._metric.name

    def empty(self) -> bool:
        """ This is not property to follow coding conventions. """
        return self._empty

    def _update(self, data: Dict[str, Any]) -> None:
        if self._instantiated:
            raise RuntimeError(
                'EnsembleResults should not be overwritten once instantiated. '
                'Instantiate new object rather than using update.'
            )

        self._train_scores.append(data[f'train_{self.metric_name}'])
        self._test_scores.append(data.get(f'test_{self.metric_name}', None))
        self._end_times.append(datetime.timestamp(data['Timestamp']))

    def _sort_by_endtime(self) -> None:
        """
        Since the default order is by start time
        and parallel computation might change the order of ending,
        this method provides the feature to sort by end time.
        Note that this method is destructive.
        """
        if self._instantiated:
            raise RuntimeError(
                'EnsembleResults should not be overwritten once instantiated. '
                'Instantiate new object with order_by_endtime=True.'
            )

        order = np.argsort(self._end_times)

        self._train_scores = self.train_scores[order].tolist()
        self._test_scores = self.test_scores[order].tolist()
        self._end_times = self.end_times[order].tolist()

    def _extract_results_from_ensemble_performance_history(
        self,
        ensemble_performance_history: List[Dict[str, Any]]
    ) -> None:
        """
        Extract information to from `ensemble_performance_history`
        to match the format of this class format.

        Args:
            ensemble_performance_history (List[Dict[str, Any]]):
                The history of the ensemble performance from EnsembleBuilder.
                Its key must be either `train_xxx`, `test_xxx` or `Timestamp`.
        """

        if (
            len(ensemble_performance_history) == 0
            or f'train_{self.metric_name}' not in ensemble_performance_history[0].keys()
        ):
            self._empty = True
            return

        self._empty = False  # We can extract ==> not empty
        for data in ensemble_performance_history:
            self._update(data)


class SearchResults:
    def __init__(
        self,
        metric: autoPyTorchMetric,
        scoring_functions: List[autoPyTorchMetric],
        run_history: RunHistory,
        order_by_endtime: bool = False
    ):
        """
        The wrapper class for run_history.
        This class extracts the information from run_history
        and allows other class to easily handle the history.
        Note that the data is sorted by starttime by default and
        metric_dict has the original form of metric value, i.e. not necessarily cost.

        Attributes:
            train_metric_dict (Dict[str, List[float]]):
                The extracted train metric information at each evaluation.
                Each list keeps the metric information specified by scoring_functions and metric.
            opt_metric_dict (Dict[str, List[float]]):
                The extracted opt metric information at each evaluation.
                Each list keeps the metric information specified by scoring_functions and metric.
            test_metric_dict (Dict[str, List[float]]):
                The extracted test metric information at each evaluation.
                Each list keeps the metric information specified by scoring_functions and metric.
            fit_times (List[float]):
                The time needed to fit each model.
            end_times (List[float]):
                The end time of the end of each evaluation.
                Each element is a float timestamp.
            configs (List[Configuration]):
                The configurations at each evaluation.
            status_types (List[StatusType]):
                The list of status types of each evaluation (e.g. success, crush).
            budgets (List[float]):
                The budgets used for each evaluation.
                Here, budget refers to the definition in Hyperband or Successive halving.
            config_ids (List[int]):
                The ID of each configuration. Since we use cutoff such as in Hyperband,
                we need to store it to know whether each configuration is a suvivor.
            is_traditionals (List[bool]):
                Whether each configuration is from traditional machine learning methods.
            additional_infos (List[Dict[str, float]]):
                It usually serves as the source of each metric at each evaluation.
                In other words, train or test performance is extracted from this info.
            rank_opt_scores (np.ndarray):
                The rank of each evaluation among all the evaluations.
            metric (autoPyTorchMetric):
                The metric of the main interest.
            scoring_functions (List[autoPyTorchMetric]):
                The list of metrics to contain in the additional_infos.
        """
        if metric not in scoring_functions:
            scoring_functions.append(metric)

        self.train_metric_dict: Dict[str, List[float]] = {metric.name: [] for metric in scoring_functions}
        self.opt_metric_dict: Dict[str, List[float]] = {metric.name: [] for metric in scoring_functions}
        self.test_metric_dict: Dict[str, List[float]] = {metric.name: [] for metric in scoring_functions}

        self._fit_times: List[float] = []
        self._end_times: List[float] = []
        self.configs: List[Configuration] = []
        self.status_types: List[StatusType] = []
        self.budgets: List[float] = []
        self.config_ids: List[int] = []
        self.is_traditionals: List[bool] = []
        self.additional_infos: List[Dict[str, float]] = []
        self.rank_opt_scores: np.ndarray = np.array([])
        self._scoring_functions = scoring_functions
        self._metric = metric
        self._instantiated = False

        self._extract_results_from_run_history(run_history)
        if order_by_endtime:
            self._sort_by_endtime()

        self._instantiated = True

    @property
    def train_scores(self) -> np.ndarray:
        """ training metric values at each evaluation """
        return np.asarray(self.train_metric_dict[self.metric_name])

    @property
    def opt_scores(self) -> np.ndarray:
        """ validation metric values at each evaluation """
        return np.asarray(self.opt_metric_dict[self.metric_name])

    @property
    def test_scores(self) -> np.ndarray:
        """ test metric values at each evaluation """
        return np.asarray(self.test_metric_dict[self.metric_name])

    @property
    def fit_times(self) -> np.ndarray:
        return np.asarray(self._fit_times)

    @property
    def end_times(self) -> np.ndarray:
        return np.asarray(self._end_times)

    @property
    def metric_name(self) -> str:
        return self._metric.name

    def _update(
        self,
        config: Configuration,
        run_key: RunKey,
        run_value: RunValue
    ) -> None:

        if self._instantiated:
            raise RuntimeError(
                'SearchResults should not be overwritten once instantiated. '
                'Instantiate new object rather than using update.'
            )
        elif run_value.status in (StatusType.STOP, StatusType.RUNNING):
            return
        elif run_value.status not in STATUS_TYPES:
            raise ValueError(f'Unexpected run status: {run_value.status}')

        is_traditional = False  # If run is not successful, unsure ==> not True ==> False
        if run_value.additional_info is not None:
            is_traditional = run_value.additional_info['configuration_origin'] == 'traditional'

        self.status_types.append(run_value.status)
        self.configs.append(config)
        self.budgets.append(run_key.budget)
        self.config_ids.append(run_key.config_id)
        self.is_traditionals.append(is_traditional)
        self.additional_infos.append(run_value.additional_info)
        self._fit_times.append(run_value.time)
        self._end_times.append(run_value.endtime)

        for inference_name in ['train', 'opt', 'test']:
            metric_info = _extract_metrics_info(
                run_value=run_value,
                scoring_functions=self._scoring_functions,
                inference_name=inference_name
            )
            for metric_name, val in metric_info.items():
                getattr(self, f'{inference_name}_metric_dict')[metric_name].append(val)

    def _sort_by_endtime(self) -> None:
        """
        Since the default order is by start time
        and parallel computation might change the order of ending,
        this method provides the feature to sort by end time.
        Note that this method is destructive.
        """
        if self._instantiated:
            raise RuntimeError(
                'SearchResults should not be overwritten once instantiated. '
                'Instantiate new object with order_by_endtime=True.'
            )

        order = np.argsort(self._end_times)

        self.train_metric_dict = {name: [arr[idx] for idx in order] for name, arr in self.train_metric_dict.items()}
        self.opt_metric_dict = {name: [arr[idx] for idx in order] for name, arr in self.opt_metric_dict.items()}
        self.test_metric_dict = {name: [arr[idx] for idx in order] for name, arr in self.test_metric_dict.items()}

        self._fit_times = [self._fit_times[idx] for idx in order]
        self._end_times = [self._end_times[idx] for idx in order]
        self.status_types = [self.status_types[idx] for idx in order]
        self.budgets = [self.budgets[idx] for idx in order]
        self.config_ids = [self.config_ids[idx] for idx in order]
        self.is_traditionals = [self.is_traditionals[idx] for idx in order]
        self.additional_infos = [self.additional_infos[idx] for idx in order]

        # Don't use numpy slicing to avoid version dependency (cast config to object might cause issues)
        self.configs = [self.configs[idx] for idx in order]

        # Only rank_opt_scores is np.ndarray
        self.rank_opt_scores = self.rank_opt_scores[order]

    def _extract_results_from_run_history(self, run_history: RunHistory) -> None:
        """
        Extract the information to match this class format.

        Args:
            run_history (RunHistory):
                The history of config evals from SMAC.
        """

        for run_key, run_value in run_history.data.items():
            config = run_history.ids_config[run_key.config_id]
            self._update(config=config, run_key=run_key, run_value=run_value)

        self._check_null_in_optional_inference_choices()

        self.rank_opt_scores = scipy.stats.rankdata(
            -1 * self._metric._sign * self.opt_scores,  # rank order
            method='min'
        )

    def _check_null_in_optional_inference_choices(
        self
    ) -> None:
        """
        Checks if the data is missing or if all the runs failed for each optional inference choice and
        sets the scores for that inference choice to all None.
        """
        for inference_choice in OPTIONAL_INFERENCE_CHOICES:
            metrics_dict = getattr(self, f'{inference_choice}_metric_dict')
            new_metric_dict = {}

            for metric in self._scoring_functions:
                scores = metrics_dict[metric.name]
                if all([score is None or score == metric._worst_possible_result for score in scores]):
                    scores = [None] * len(self.status_types)
                new_metric_dict[metric.name] = scores
            setattr(self, f'{inference_choice}_metric_dict', new_metric_dict)


class MetricResults:
    def __init__(
        self,
        metric: autoPyTorchMetric,
        run_history: RunHistory,
        ensemble_performance_history: List[Dict[str, Any]]
    ):
        """
        The wrapper class for ensemble_performance_history.
        This class extracts the information from ensemble_performance_history
        and allows other class to easily handle the history.
        Note that all the data is sorted by endtime!

        Attributes:
            start_time (float):
                The timestamp at the very beginning of the optimization.
            cum_times (np.ndarray):
                The runtime needed to reach the end of each evaluation.
                The time unit is second.
            metric (autoPyTorchMetric):
                The information about the metric to contain.
            search_results (SearchResults):
                The instance to fetch the metric values of `self.metric`
                from run_history.
            ensemble_results (EnsembleResults):
                The instance to fetch the metric values of `self.metric`
                from ensemble_performance_history.
                If there is no information available, self.empty() returns True.
            data (Dict[str, np.ndarray]):
                Keys are `{single, ensemble}::{train, opt, test}::{metric.name}`.
                Each array contains the evaluated values for the corresponding category.
        """
        self.start_time = get_start_time(run_history)
        self.metric = metric
        self.search_results = SearchResults(
            metric=metric,
            run_history=run_history,
            scoring_functions=[],
            order_by_endtime=True
        )
        self.ensemble_results = EnsembleResults(
            metric=metric,
            ensemble_performance_history=ensemble_performance_history,
            order_by_endtime=True
        )

        if (
            not self.ensemble_results.empty()
            and self.search_results.end_times[-1] < self.ensemble_results.end_times[-1]
        ):
            # Augment runtime table with the final available end time
            self.cum_times = np.hstack(
                [self.search_results.end_times - self.start_time,
                 [self.ensemble_results.end_times[-1] - self.start_time]]
            )
        else:
            self.cum_times = self.search_results.end_times - self.start_time

        self.data: Dict[str, np.ndarray] = {}
        self._extract_results()

    def _extract_results(self) -> None:
        """ Extract metric values of `self.metric` and store them in `self.data`. """
        metric_name = self.metric.name
        for inference_name in ['train', 'test', 'opt']:
            # TODO: Extract information from self.search_results
            data = getattr(self.search_results, f'{inference_name}_metric_dict')[metric_name]
            if all([d is None for d in data]):
                if inference_name not in OPTIONAL_INFERENCE_CHOICES:
                    raise ValueError(f"Expected {metric_name} score for {inference_name} set"
                                     f" to not be None, but got {data}")
                else:
                    continue
            self.data[f'single::{inference_name}::{metric_name}'] = np.array(data)

            if self.ensemble_results.empty() or inference_name == 'opt':
                continue

            data = getattr(self.ensemble_results, f'{inference_name}_scores')
            if all([d is None for d in data]):
                if inference_name not in OPTIONAL_INFERENCE_CHOICES:
                    raise ValueError(f"Expected {metric_name} score for {inference_name} set"
                                     f" to not be None, but got {data}")
                else:
                    continue
            self.data[f'ensemble::{inference_name}::{metric_name}'] = np.array(data)

    def get_ensemble_merged_data(self) -> Dict[str, np.ndarray]:
        """
        Merge the ensemble performance data to the closest time step
        available in the run_history.
        One performance metric will be allocated to one time step.
        Other time steps will be filled by the worst possible value.

        Returns:
            data (Dict[str, np.ndarray]):
                Merged data as mentioned above
        """

        data = {k: v.copy() for k, v in self.data.items()}  # deep copy

        if self.ensemble_results.empty():  # no ensemble data available
            return data

        train_scores, test_scores = self.ensemble_results.train_scores, self.ensemble_results.test_scores
        end_times = self.ensemble_results.end_times
        cur, timestep_size, sign = 0, self.cum_times.size, self.metric._sign
        key_train, key_test = f'ensemble::train::{self.metric.name}', f'ensemble::test::{self.metric.name}'

        all_test_perfs_null = all([perf is None for perf in test_scores])

        train_perfs = np.full_like(self.cum_times, self.metric._worst_possible_result)
        test_perfs = np.full_like(self.cum_times, self.metric._worst_possible_result)

        for timestamp, train_score, test_score in zip(end_times, train_scores, test_scores):
            avail_time = timestamp - self.start_time
            while cur < timestep_size and self.cum_times[cur] < avail_time:
                # Guarantee that cum_times[cur] >= avail_time
                cur += 1

            # results[cur] is the closest available checkpoint after or at the avail_time
            # ==> Assign this data to that checkpoint
            time_index = min(cur, timestep_size - 1)
            # If there already exists a previous allocated value, update by a better value
            train_perfs[time_index] = sign * max(sign * train_perfs[time_index], sign * train_score)
            # test_perfs can be none when X_test is not passed
            if not all_test_perfs_null:
                test_perfs[time_index] = sign * max(sign * test_perfs[time_index], sign * test_score)

        update_dict = {key_train: train_perfs}
        if not all_test_perfs_null:
            update_dict[key_test] = test_perfs

        data.update(update_dict)

        return data


class ResultsManager:
    def __init__(self, *args: Any, **kwargs: Any):
        """
        This module is used to gather result information for BaseTask.
        In other words, this module is supposed to be wrapped by BaseTask.

        Attributes:
            run_history (RunHistory):
                A `SMAC Runshistory <https://automl.github.io/SMAC3/master/apidoc/smac.runhistory.runhistory.html>`_
                object that holds information about the runs of the target algorithm made during search
            ensemble_performance_history (List[Dict[str, Any]]):
                The history of the ensemble performance from EnsembleBuilder.
                Its keys are `train_xxx`, `test_xxx` or `Timestamp`.
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

        incumbent_idx = indices[np.argmax(metric._sign * scores)]
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
        success_status = (StatusType.SUCCESS, StatusType.DONOTADVANCE)
        sio = io.StringIO()
        sio.write("autoPyTorch results:\n")
        sio.write(f"\tDataset name: {dataset_name}\n")
        sio.write(f"\tOptimisation Metric: {metric}\n")

        num_runs = len(search_results.status_types)
        num_success = sum([s in success_status for s in search_results.status_types])
        num_crash = sum([s == StatusType.CRASHED for s in search_results.status_types])
        num_timeout = sum([s == StatusType.TIMEOUT for s in search_results.status_types])
        num_memout = sum([s == StatusType.MEMOUT for s in search_results.status_types])

        if num_success > 0:
            best_score = metric._sign * np.max(metric._sign * search_results.opt_scores)
            sio.write(f"\tBest validation score: {best_score}\n")

        sio.write(f"\tNumber of target algorithm runs: {num_runs}\n")
        sio.write(f"\tNumber of successful target algorithm runs: {num_success}\n")
        sio.write(f"\tNumber of crashed target algorithm runs: {num_crash}\n")
        sio.write(f"\tNumber of target algorithms that exceeded the time "
                  f"limit: {num_timeout}\n")
        sio.write(f"\tNumber of target algorithms that exceeded the memory "
                  f"limit: {num_memout}\n")

        return sio.getvalue()
