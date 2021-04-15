import copy
import json
import logging.handlers
import multiprocessing
import os
import platform
import sys
import tempfile
import time
import typing
import unittest.mock
import uuid
import warnings
from abc import abstractmethod
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union, cast

from ConfigSpace.configuration_space import Configuration, ConfigurationSpace

import dask

import joblib

import numpy as np

import pandas as pd

from smac.runhistory.runhistory import DataOrigin, RunHistory
from smac.stats.stats import Stats
from smac.tae import StatusType

from autoPyTorch.constants import (
    REGRESSION_TASKS,
    STRING_TO_OUTPUT_TYPES,
    STRING_TO_TASK_TYPES,
)
from autoPyTorch.datasets.base_dataset import BaseDataset
from autoPyTorch.datasets.resampling_strategy import CrossValTypes, HoldoutValTypes
from autoPyTorch.ensemble.ensemble_builder import EnsembleBuilderManager
from autoPyTorch.ensemble.ensemble_selection import EnsembleSelection
from autoPyTorch.ensemble.singlebest_ensemble import SingleBest
from autoPyTorch.evaluation.abstract_evaluator import fit_and_suppress_warnings
from autoPyTorch.evaluation.tae import AdditionalRunInfoType, ExecuteTAFuncWithQueue, ResultType, get_cost_of_crash
from autoPyTorch.optimizer.smbo import AutoMLSMBO
from autoPyTorch.pipeline.base_pipeline import BasePipeline
from autoPyTorch.pipeline.components.setup.traditional_ml.classifier_models import get_available_classifiers
from autoPyTorch.pipeline.components.training.metrics.base import autoPyTorchMetric
from autoPyTorch.pipeline.components.training.metrics.utils import calculate_score, get_metrics
from autoPyTorch.utils.backend import Backend, create
from autoPyTorch.utils.common import replace_string_bool_to_bool
from autoPyTorch.utils.hyperparameter_search_space_update import HyperparameterSearchSpaceUpdates
from autoPyTorch.utils.logging_ import (
    PicklableClientLogger,
    get_named_client_logger,
    setup_logger,
    start_log_server,
)
from autoPyTorch.utils.pipeline import get_configuration_space, get_dataset_requirements
from autoPyTorch.utils.stopwatch import StopWatch


# Typing for ExecuteTAFuncWithQueue.run
ExecuteTAFuncWithQueueRunType = Callable[[Dict[str, Any]], ResultType]


class DaskFutureTaskType():
    def __init__(self, ta: ExecuteTAFuncWithQueueRunType,
                 *args: List[Any], **kwargs: Dict[str, Any]):
        self.ta = ta
        self.args = args
        self.kwargs = kwargs
        raise TypeError("Cannot instantiate `DaskFutureTaskType` instances.")

    def result(self) -> ResultType:
        # Implement `return self.ta(*self.args, **self.kwargs)`
        raise NotImplementedError


def _pipeline_predict(pipeline: BasePipeline,
                      X: Union[np.ndarray, pd.DataFrame],
                      batch_size: int,
                      logger: PicklableClientLogger,
                      task: int) -> np.ndarray:
    @typing.no_type_check
    def send_warnings_to_log(
            message, category, filename, lineno, file=None, line=None):
        logger.debug('%s:%s: %s:%s' % (filename, lineno, category.__name__, message))
        return

    X_ = X.copy()
    with warnings.catch_warnings():
        warnings.showwarning = send_warnings_to_log
        if task in REGRESSION_TASKS:
            # Voting regressor does not support batch size
            prediction = pipeline.predict(X_)
        else:
            # Voting classifier predict proba does not support batch size
            prediction = pipeline.predict_proba(X_)
            # Check that all probability values lie between 0 and 1.
            if not ((prediction >= 0).all() and (prediction <= 1).all()):
                np.set_printoptions(threshold=sys.maxsize)
                raise ValueError("For {}, prediction probability not within [0, 1]: {}/{}!".format(
                    pipeline,
                    prediction,
                    np.sum(prediction, axis=1)
                ))

    if len(prediction.shape) < 1 or len(X_.shape) < 1 or \
            X_.shape[0] < 1 or prediction.shape[0] != X_.shape[0]:
        logger.warning(
            "Prediction shape for model %s is %s while X_.shape is %s",
            pipeline, str(prediction.shape), str(X_.shape)
        )
    return prediction


class BaseTask:
    """
    Base class for the tasks that serve as API to the pipelines.
    Args:
        seed (int), (default=1): seed to be used for reproducibility.
        n_jobs (int), (default=1): number of consecutive processes to spawn.
        logging_config (Optional[Dict]): specifies configuration
            for logging, if None, it is loaded from the logging.yaml
        ensemble_size (int), (default=50): Number of models added to the ensemble built by
            Ensemble selection from libraries of models.
            Models are drawn with replacement.
        ensemble_nbest (int), (default=50): only consider the ensemble_nbest
            models to build the ensemble
        max_models_on_disc (int), (default=50): maximum number of models saved to disc.
            Also, controls the size of the ensemble as any additional models will be deleted.
            Must be greater than or equal to 1.
        temporary_directory (str): folder to store configuration output and log file
        output_directory (str): folder to store predictions for optional test set
        delete_tmp_folder_after_terminate (bool): determines whether to delete the temporary directory,
            when finished
        include_components (Optional[Dict]): If None, all possible components are used.
            Otherwise specifies set of components to use.
        exclude_components (Optional[Dict]): If None, all possible components are used.
            Otherwise specifies set of components not to use. Incompatible with include
            components
    """

    def __init__(
        self,
        seed: int = 1,
        n_jobs: int = 1,
        logging_config: Optional[Dict] = None,
        ensemble_size: int = 50,
        ensemble_nbest: int = 50,
        max_models_on_disc: int = 50,
        temporary_directory: Optional[str] = None,
        output_directory: Optional[str] = None,
        delete_tmp_folder_after_terminate: bool = True,
        delete_output_folder_after_terminate: bool = True,
        include_components: Optional[Dict] = None,
        exclude_components: Optional[Dict] = None,
        backend: Optional[Backend] = None,
        resampling_strategy: Union[CrossValTypes, HoldoutValTypes] = HoldoutValTypes.holdout_validation,
        resampling_strategy_args: Optional[Dict[str, Any]] = None,
        search_space_updates: Optional[HyperparameterSearchSpaceUpdates] = None,
        task_type: Optional[str] = None
    ) -> None:
        self.seed = seed
        self.n_jobs = n_jobs
        self.ensemble_size = ensemble_size
        self.ensemble_nbest = ensemble_nbest
        self.max_models_on_disc = max_models_on_disc
        self.logging_config: Optional[Dict] = logging_config
        self.include_components: Optional[Dict] = include_components
        self.exclude_components: Optional[Dict] = exclude_components
        self._temporary_directory = temporary_directory
        self._output_directory = output_directory
        if backend is not None:
            self._backend = backend
        else:
            self._backend = create(
                temporary_directory=self._temporary_directory,
                output_directory=self._output_directory,
                delete_tmp_folder_after_terminate=delete_tmp_folder_after_terminate,
                delete_output_folder_after_terminate=delete_output_folder_after_terminate,
            )
        self.task_type = task_type or ""
        self._stopwatch = StopWatch()

        self.pipeline_options = replace_string_bool_to_bool(json.load(open(
            os.path.join(os.path.dirname(__file__), '../configs/default_pipeline_options.json'))))

        self.search_space: Optional[ConfigurationSpace] = None
        self._metric: Optional[autoPyTorchMetric] = None
        self._logger: Optional[PicklableClientLogger] = None
        self.run_history: RunHistory = RunHistory()
        self.trajectory: Optional[List] = None
        self.dataset_name: str = ""
        self.cv_models_: Dict = {}
        self.experiment_task_name: str = 'runSearch'

        # By default try to use the TCP logging port or get a new port
        self._logger_port = logging.handlers.DEFAULT_TCP_LOGGING_PORT

        # Store the resampling strategy from the dataset, to load models as needed
        self.resampling_strategy = resampling_strategy
        self.resampling_strategy_args = resampling_strategy_args

        self.stop_logging_server = None  # type: Optional[multiprocessing.synchronize.Event]

        self._dask_client = None

        self.search_space_updates = search_space_updates
        if search_space_updates is not None:
            if not isinstance(self.search_space_updates,
                              HyperparameterSearchSpaceUpdates):
                raise ValueError("Expected search space updates to be of instance"
                                 " HyperparameterSearchSpaceUpdates got {}".format(type(self.search_space_updates)))

    @abstractmethod
    def _get_required_dataset_properties(self, dataset: BaseDataset) -> Dict[str, Any]:
        """
        given a pipeline type, this function returns the
        dataset properties required by the dataset object
        """
        raise NotImplementedError

    @abstractmethod
    def build_pipeline(self, dataset_properties: Dict[str, Any]) -> BasePipeline:
        """
        Build pipeline according to current task
        and for the passed dataset properties
        Args:
            dataset_properties (Dict[str,Any]):

        Returns:

        """
        raise NotImplementedError

    def set_pipeline_config(
            self,
            **pipeline_config_kwargs: Any) -> None:
        """
        Check whether arguments are valid and
        then sets them to the current pipeline
        configuration.
        Args:
            **pipeline_config_kwargs: Valid config options include "num_run",
            "device", "budget_type", "epochs", "runtime", "torch_num_threads",
            "early_stopping", "use_tensorboard_logger",
            "metrics_during_training"

        Returns:
            None
        """
        unknown_keys = []
        for option, value in pipeline_config_kwargs.items():
            if option in self.pipeline_options.keys():
                pass
            else:
                unknown_keys.append(option)

        if len(unknown_keys) > 0:
            raise ValueError("Invalid configuration arguments given {},"
                             " expected arguments to be in {}".
                             format(unknown_keys, self.pipeline_options.keys()))

        self.pipeline_options.update(pipeline_config_kwargs)

    def get_pipeline_options(self) -> dict:
        """
        Returns the current pipeline configuration.
        """
        return self.pipeline_options

    # def set_search_space(self, search_space: ConfigurationSpace) -> None:
    #     """
    #     Update the search space.
    #     """
    #     raise NotImplementedError
    #
    def get_search_space(self, dataset: BaseDataset = None) -> ConfigurationSpace:
        """
        Returns the current search space as ConfigurationSpace object.
        """
        if self.search_space is not None:
            return self.search_space
        elif dataset is not None:
            dataset_requirements = get_dataset_requirements(
                info=self._get_required_dataset_properties(dataset))
            return get_configuration_space(info=dataset.get_dataset_properties(dataset_requirements),
                                           include=self.include_components,
                                           exclude=self.exclude_components,
                                           search_space_updates=self.search_space_updates)
        raise Exception("No search space initialised and no dataset passed. "
                        "Can't create default search space without the dataset")

    def _get_logger(self, name: str) -> PicklableClientLogger:
        """
        Instantiates the logger used throughout the experiment
        Args:
            name (str): name of the log file,
            usually the dataset name

        Returns:
            PicklableClientLogger
        """
        logger_name = 'AutoPyTorch:%s:%d' % (name, self.seed)

        # Setup the configuration for the logger
        # This is gonna be honored by the server
        # Which is created below
        setup_logger(
            filename='%s.log' % str(logger_name),
            logging_config=self.logging_config,
            output_dir=self._backend.temporary_directory,
        )

        # As AutoPyTorch works with distributed process,
        # we implement a logger server that can receive tcp
        # pickled messages. They are unpickled and processed locally
        # under the above logging configuration setting
        # We need to specify the logger_name so that received records
        # are treated under the logger_name ROOT logger setting
        context = multiprocessing.get_context('spawn')
        self.stop_logging_server = context.Event()
        port = context.Value('l')  # be safe by using a long
        port.value = -1

        # "BaseContext" has no attribute "Process" motivates to ignore the attr check
        self.logging_server = context.Process(  # type: ignore [attr-defined]
            target=start_log_server,
            kwargs=dict(
                host='localhost',
                logname=logger_name,
                event=self.stop_logging_server,
                port=port,
                filename='%s.log' % str(logger_name),
                logging_config=self.logging_config,
                output_dir=self._backend.temporary_directory,
            ),
        )

        self.logging_server.start()

        while True:
            with port.get_lock():
                if port.value == -1:
                    time.sleep(0.01)
                else:
                    break

        self._logger_port = int(port.value)

        return get_named_client_logger(
            name=logger_name,
            host='localhost',
            port=self._logger_port,
        )

    def _clean_logger(self) -> None:
        """
        cleans the logging server created
        Returns:

        """
        if not hasattr(self, 'stop_logging_server') or self.stop_logging_server is None:
            return

        # Clean up the logger
        if self.logging_server.is_alive():
            self.stop_logging_server.set()

            # We try to join the process, after we sent
            # the terminate event. Then we try a join to
            # nicely join the event. In case something
            # bad happens with nicely trying to kill the
            # process, we execute a terminate to kill the
            # process.
            self.logging_server.join(timeout=5)
            self.logging_server.terminate()
            del self.stop_logging_server

    def _create_dask_client(self) -> None:
        """
        creates the dask client that is used to parallelize
        the training of pipelines
        Returns:
            None
        """
        self._is_dask_client_internally_created = True
        dask.config.set({'distributed.worker.daemon': False})
        self._dask_client = dask.distributed.Client(
            dask.distributed.LocalCluster(
                n_workers=self.n_jobs,
                processes=True,
                threads_per_worker=1,
                # We use the temporal directory to save the
                # dask workers, because deleting workers
                # more time than deleting backend directories
                # This prevent an error saying that the worker
                # file was deleted, so the client could not close
                # the worker properly
                local_directory=tempfile.gettempdir(),
                # Memory is handled by the pynisher, not by the dask worker/nanny
                memory_limit=0,
            ),
            # Heartbeat every 10s
            heartbeat_interval=10000,
        )

    def _close_dask_client(self) -> None:
        """
        Closes the created dask client
        Returns:
            None
        """
        if (
                hasattr(self, '_is_dask_client_internally_created')
                and self._is_dask_client_internally_created
                and self._dask_client
        ):
            self._dask_client.shutdown()
            self._dask_client.close()
            del self._dask_client
            self._dask_client = None
            self._is_dask_client_internally_created = False
            del self._is_dask_client_internally_created

    def _load_models(self) -> bool:

        """
        Loads the models saved in the temporary directory
        during the smac run and the final ensemble created

        Returns:
            None
        """
        if self.resampling_strategy is None:
            raise ValueError("Resampling strategy is needed to determine what models to load")
        self.ensemble_ = self._backend.load_ensemble(self.seed)

        # If no ensemble is loaded, try to get the best performing model
        if not self.ensemble_:
            self.ensemble_ = self._load_best_individual_model()

        if self.ensemble_:
            identifiers = self.ensemble_.get_selected_model_identifiers()
            self.models_ = self._backend.load_models_by_identifiers(identifiers)
            if isinstance(self.resampling_strategy, CrossValTypes):
                self.cv_models_ = self._backend.load_cv_models_by_identifiers(identifiers)

            if isinstance(self.resampling_strategy, CrossValTypes):
                if len(self.cv_models_) == 0:
                    raise ValueError('No models fitted!')

        elif 'pipeline' not in self._disable_file_output:
            model_names = self._backend.list_all_models(self.seed)

            if len(model_names) == 0:
                raise ValueError('No models fitted!')

            self.models_ = {}

        else:
            self.models_ = {}

        return True

    def _load_best_individual_model(self) -> SingleBest:
        """
        In case of failure during ensemble building,
        this method returns the single best model found
        by AutoML.
        This is a robust mechanism to be able to predict,
        even though no ensemble was found by ensemble builder.
        """

        if self._metric is None:
            raise ValueError("Providing a metric to AutoPytorch is required to fit a model. "
                             "A default metric could not be inferred. Please check the log "
                             "for error messages."
                             )

        # SingleBest contains the best model found by AutoML
        ensemble = SingleBest(
            metric=self._metric,
            seed=self.seed,
            run_history=self.run_history,
            backend=self._backend,
        )
        msg = "No valid ensemble was created. Please check the log" \
              f"file for errors. Default to the best individual estimator:{ensemble.identifiers_}"

        if self._logger is None:
            warnings.warn(msg)
        else:
            self._logger.exception(msg)

        return ensemble

    def _get_target_algorithm(self, wallclock_limit: int) -> ExecuteTAFuncWithQueue:
        scenario_mock = unittest.mock.Mock()
        scenario_mock.wallclock_limit = wallclock_limit
        stats = Stats(scenario_mock)
        stats.start_timing()

        assert self._metric is not None
        ta = ExecuteTAFuncWithQueue(
            backend=self._backend,
            seed=self.seed,
            metric=self._metric,
            logger_port=self._logger_port,
            cost_for_crash=get_cost_of_crash(self._metric),
            abort_on_first_run_crash=False,
            initial_num_run=self._backend.get_next_num_run(),
            stats=stats,
            memory_limit=self._memory_limit,
            disable_file_output=True if len(self._disable_file_output) > 0 else False,
            all_supported_metrics=self._all_supported_metrics
        )
        return ta

    def _logging_failed_prediction(self, additional_info: Any, header: str) -> None:
        assert self._logger is not None

        if additional_info.get('exitcode') == -6:
            err_msg = "The error suggests that the provided memory limits were too tight. Please " \
                      "increase the 'ml_memory_limit' and try again. If this does not solve your " \
                      "problem, please open an issue and paste the additional output. " \
                      f"Additional output: {str(additional_info)}.",
            output = f"{header}. {err_msg}"
            self._logger.error(output)
            raise ValueError(output)

        else:
            output = f"{header} and additional output: {str(additional_info)}."
            self._logger.error(output)
            raise ValueError(output)

    def _parallel_worker_allocation(self, num_future_jobs: int, run_history: RunHistory,
                                    dask_futures: List[Tuple[str, DaskFutureTaskType]]
                                    ) -> None:
        """
        The functin to allocate and implement jobs to unused workers.
        The history is recorded in run_history.

        Args:
            num_future_jobs (int): The number of jobs to run
            dask_futures (List[Tuple[str, ExecuteTAFuncWithQueue.run]]):
                The list of pairs of the name of the classifier to run and
                the function to train the classifier
            run_history (RunHistory):
                The running history of the experiment

        Note:
            - `dask_futures.pop(0)` gives a classifier and a next job to run
            - `future.result()` calls a submitted job in and return the results
            - We have to wait for the return of `future.result()`
              once the number of running jobs reaches `num_workers` in self.dask_client

        """
        assert self._logger is not None

        while num_future_jobs >= 1:
            num_future_jobs -= 1
            classifier, future = dask_futures.pop(0)
            # call the training by future.result()
            status, cost, runtime, additional_info = future.result()

            if status == StatusType.SUCCESS:
                self._logger.info(
                    f"Fitting {classifier} took {runtime}s, performance:{cost}/{additional_info}")
                configuration = additional_info['pipeline_configuration']
                origin = additional_info['configuration_origin']
                run_history.add(config=configuration, cost=cost,
                                time=runtime, status=status, seed=self.seed,
                                origin=origin)
            else:
                header = f"Traditional prediction for {classifier} failed with run state {str(status)}"
                self._logging_failed_prediction(additional_info, header)

    def _traditional_predictions(self, time_left: int) -> None:
        """
        Fits traditional machine learning algorithms to the provided dataset, while
        complying with time resource allocation.

        This method currently only supports classification.

        Args:
            time_left: (int)
                Hard limit on how many machine learning algorithms can be fit. Depending on how
                fast a traditional machine learning algorithm trains, it will allow multiple
                models to be fitted.
            func_eval_time_limit_secs: (int)
                Maximum training time each algorithm is allowed to take, during training
        """

        # Mypy Checkings -- Traditional prediction is only called for search
        # where the following objects are created
        assert self._metric is not None
        assert self._logger is not None
        assert self._dask_client is not None

        self._logger.info("Start to create traditional classifier predictions.")

        # Initialise run history for the traditional classifiers
        run_history = RunHistory()

        available_classifiers = get_available_classifiers()
        dask_futures = []

        total_number_classifiers = len(available_classifiers)
        for n_r, classifier in enumerate(available_classifiers):

            # Only launch a task if there is time
            start_time = time.time()
            sufficient_time_available = (time_left >= self._func_eval_time_limit_secs)

            if sufficient_time_available:
                self._logger.info(f"{n_r}: Start fitting {classifier} with cutoff={self._func_eval_time_limit_secs}")
                ta = self._get_target_algorithm(time_left)
                dask_futures.append([
                    classifier,
                    self._dask_client.submit(
                        ta.run, config=classifier,
                        cutoff=self._func_eval_time_limit_secs,
                    )])

            if len(dask_futures) >= self.n_jobs:
                last_iteration = (n_r >= total_number_classifiers - 1)
                num_future_jobs = 1
                # If it is the last iteration, we have to run all the jobs
                if not sufficient_time_available or last_iteration:
                    num_future_jobs = len(dask_futures)

                self._parallel_worker_allocation(num_future_jobs=num_future_jobs,
                                                 run_history=run_history,
                                                 dask_futures=dask_futures)

            time_left -= int(time.time() - start_time)

            if time_left < self._func_eval_time_limit_secs:
                self._logger.warning("Not enough time to fit all traditional machine learning models."
                                     "Please consider increasing the run time to further improve performance.")
                break

        self._logger.debug("Run history traditional: {}".format(run_history))
        # add run history of traditional to api run history
        self.run_history.update(run_history, DataOrigin.EXTERNAL_SAME_INSTANCES)
        run_history.save_json(os.path.join(self._backend.internals_directory, 'traditional_run_history.json'),
                              save_external=True)
        return

    def _run_dummy_predictions(self) -> None:
        assert self._metric is not None
        assert self._logger is not None

        # For dummy estimator, we always expect the num_run to be 1
        num_run = 1

        dummy_task_name = 'runDummy'
        self._stopwatch.start_task(dummy_task_name)
        self._logger.info("Start to create dummy predictions.")
        ta = self._get_target_algorithm(self._total_walltime_limit)
        status, cost, runtime, additional_info = ta.run(num_run, cutoff=self._total_walltime_limit)
        if status == StatusType.SUCCESS:
            self._logger.info("Finish creating dummy predictions.")
        else:
            header = f"Dummy prediction failed with run state {str(status)}"
            self._logging_failed_prediction(additional_info=additional_info,
                                            header=header)
        self._stopwatch.stop_task(dummy_task_name)

    def _run_traditional_ml(self) -> None:
        """We would like to obtain training time for at least 1 Neural network in SMAC"""
        assert self._logger is not None

        if STRING_TO_TASK_TYPES[self.task_type] in REGRESSION_TASKS:
            self._logger.warning("Traditional Pipeline is not enabled for regression. Skipping...")
        else:
            traditional_task_name = 'runTraditional'
            self._stopwatch.start_task(traditional_task_name)
            elapsed_time = self._stopwatch.wall_elapsed(self.dataset_name)

            assert self._func_eval_time_limit_secs is not None
            time_for_traditional = int(
                self._total_walltime_limit - elapsed_time - self._func_eval_time_limit_secs
            )
            self._traditional_predictions(time_left=time_for_traditional)
            self._stopwatch.stop_task(traditional_task_name)

    def _run_ensemble(self, dataset: BaseDataset, optimize_metric: str,
                      precision: int) -> Optional[EnsembleBuilderManager]:

        assert self._logger is not None
        assert self._metric is not None

        elapsed_time = self._stopwatch.wall_elapsed(self.dataset_name)
        time_left_for_ensembles = max(0, self._total_walltime_limit - elapsed_time)
        proc_ensemble = None
        if time_left_for_ensembles <= 0 and self.ensemble_size > 0:
            raise ValueError("Could not run ensemble builder because there "
                             "is no time left. Try increasing the value "
                             "of total_walltime_limit.")
        elif self.ensemble_size <= 0:
            self._logger.info("Could not run ensemble builder as ensemble size is non-positive.")
        else:
            self._logger.info("Run ensemble")
            ensemble_task_name = 'ensemble'
            self._stopwatch.start_task(ensemble_task_name)
            proc_ensemble = EnsembleBuilderManager(
                start_time=time.time(),
                time_left_for_ensembles=time_left_for_ensembles,
                backend=copy.deepcopy(self._backend),
                dataset_name=dataset.dataset_name,
                output_type=STRING_TO_OUTPUT_TYPES[dataset.output_type],
                task_type=STRING_TO_TASK_TYPES[self.task_type],
                metrics=[self._metric],
                opt_metric=optimize_metric,
                ensemble_size=self.ensemble_size,
                ensemble_nbest=self.ensemble_nbest,
                max_models_on_disc=self.max_models_on_disc,
                seed=self.seed,
                max_iterations=None,
                read_at_most=sys.maxsize,
                ensemble_memory_limit=self._memory_limit,
                random_state=self.seed,
                precision=precision,
                logger_port=self._logger_port
            )
            self._stopwatch.stop_task(ensemble_task_name)

        return proc_ensemble

    def _get_budget_config(self, budget_type: Optional[str] = None,
                           budget: Optional[float] = None) -> Dict[str, Union[float, str]]:

        budget_config: Dict[str, Union[float, str]] = {}
        if budget_type is not None and budget is not None:
            budget_config['budget_type'] = budget_type
            budget_config[budget_type] = budget
        elif budget_type is not None or budget is not None:
            raise ValueError("budget type was not specified in budget_config")

        return budget_config

    def _start_smac(self, proc_smac: AutoMLSMBO) -> None:
        assert self._logger is not None

        try:
            run_history, self.trajectory, budget_type = \
                proc_smac.run_smbo()
            self.run_history.update(run_history, DataOrigin.INTERNAL)
            trajectory_filename = os.path.join(
                self._backend.get_smac_output_directory_for_run(self.seed),
                'trajectory.json')

            assert self.trajectory is not None

            saveable_trajectory = \
                [list(entry[:2]) + [entry[2].get_dictionary()] + list(entry[3:])
                 for entry in self.trajectory]
        except Exception as e:
            self._logger.exception(str(e))
            raise
        else:
            try:
                with open(trajectory_filename, 'w') as fh:
                    json.dump(saveable_trajectory, fh)
            except Exception as e:
                self._logger.warning(f"Could not save {trajectory_filename} due to {e}...")

    def _run_smac(self, dataset: BaseDataset, proc_ensemble: Optional[EnsembleBuilderManager],
                  budget_type: Optional[str] = None, budget: Optional[float] = None,
                  get_smac_object_callback: Optional[Callable] = None,
                  smac_scenario_args: Optional[Dict[str, Any]] = None) -> None:

        assert self._logger is not None

        smac_task_name = 'runSMAC'
        self._stopwatch.start_task(smac_task_name)
        elapsed_time = self._stopwatch.wall_elapsed(self.experiment_task_name)
        time_left_for_smac = max(0, self._total_walltime_limit - elapsed_time)

        self._logger.info(f"Run SMAC with {time_left_for_smac:.2f} sec time left")
        if time_left_for_smac <= 0:
            self._logger.warning(" Could not run SMAC because there is no time left")
        else:
            budget_config = self._get_budget_config(budget_type=budget_type, budget=budget)

            assert self._func_eval_time_limit_secs is not None
            assert self._metric is not None
            proc_smac = AutoMLSMBO(
                config_space=self.search_space,
                dataset_name=dataset.dataset_name,
                backend=self._backend,
                total_walltime_limit=self._total_walltime_limit,
                func_eval_time_limit_secs=self._func_eval_time_limit_secs,
                dask_client=self._dask_client,
                memory_limit=self._memory_limit,
                n_jobs=self.n_jobs,
                watcher=self._stopwatch,
                metric=self._metric,
                seed=self.seed,
                include=self.include_components,
                exclude=self.exclude_components,
                disable_file_output=self._disable_file_output,
                all_supported_metrics=self._all_supported_metrics,
                smac_scenario_args=smac_scenario_args,
                get_smac_object_callback=get_smac_object_callback,
                pipeline_config={**self.pipeline_options, **budget_config},
                ensemble_callback=proc_ensemble,
                logger_port=self._logger_port,
                start_num_run=self._backend.get_next_num_run(peek=True),
                search_space_updates=self.search_space_updates
            )

            self._start_smac(proc_smac)

    def _search_settings(self, dataset: BaseDataset, disable_file_output: List,
                         optimize_metric: str, memory_limit: Optional[int] = 4096,
                         func_eval_time_limit_secs: Optional[int] = None,
                         total_walltime_limit: int = 100,
                         all_supported_metrics: bool = True) -> None:

        """Initialise information needed for the experiment"""
        self.experiment_task_name = 'runSearch'
        dataset_requirements = get_dataset_requirements(
            info=self._get_required_dataset_properties(dataset))
        dataset_properties = dataset.get_dataset_properties(dataset_requirements)

        self._stopwatch.start_task(self.experiment_task_name)
        self.dataset_name = dataset.dataset_name
        self._all_supported_metrics = all_supported_metrics
        self._disable_file_output = disable_file_output
        self._memory_limit = memory_limit
        self._total_walltime_limit = total_walltime_limit
        self._func_eval_time_limit_secs = func_eval_time_limit_secs
        self._metric = get_metrics(
            names=[optimize_metric], dataset_properties=dataset_properties)[0]

        if self._logger is None:
            self._logger = self._get_logger(str(self.dataset_name))

        # Save start time to backend
        self._backend.save_start_time(str(self.seed))
        self._backend.save_datamanager(dataset)

        # Print debug information to log
        self._print_debug_info_to_log()

        self.search_space = self.get_search_space(dataset)

        # If no dask client was provided, we create one, so that we can
        # start a ensemble process in parallel to smbo optimize
        if (
            self._dask_client is None and (self.ensemble_size > 0 or self.n_jobs is not None and self.n_jobs > 1)
        ):
            self._create_dask_client()
        else:
            self._is_dask_client_internally_created = False

    def _adapt_time_resource_allocation(self) -> None:
        assert self._logger is not None

        # Handle time resource allocation
        elapsed_time = self._stopwatch.wall_elapsed(self.experiment_task_name)
        time_left_for_modelfit = int(max(0, self._total_walltime_limit - elapsed_time))
        if self._func_eval_time_limit_secs is None or self._func_eval_time_limit_secs > time_left_for_modelfit:
            self._logger.warning(
                'Time limit for a single run is higher than total time '
                'limit. Capping the limit for a single run to the total '
                'time given to SMAC (%f)' % time_left_for_modelfit
            )
            self._func_eval_time_limit_secs = time_left_for_modelfit

        # Make sure that at least 2 models are created for the ensemble process
        num_models = time_left_for_modelfit // self._func_eval_time_limit_secs
        if num_models < 2:
            self._func_eval_time_limit_secs = time_left_for_modelfit // 2
            self._logger.warning(
                "Capping the func_eval_time_limit_secs to {} to have "
                "time for a least 2 models to ensemble.".format(
                    self._func_eval_time_limit_secs
                )
            )

    def _save_ensemble_performance_history(self, proc_ensemble: EnsembleBuilderManager) -> None:
        assert self._logger is not None

        if len(proc_ensemble.futures) > 0:
            # Also add ensemble runs that did not finish within smac time
            # and add them into the ensemble history
            self._logger.info("Ensemble script still running, waiting for it to finish.")
            result = proc_ensemble.futures.pop().result()
            if result:
                ensemble_history, _, _, _ = result
                self.ensemble_performance_history.extend(ensemble_history)
            self._logger.info("Ensemble script finished, continue shutdown.")

        # save the ensemble performance history file
        if len(self.ensemble_performance_history) > 0:
            pd.DataFrame(self.ensemble_performance_history).to_json(
                os.path.join(self._backend.internals_directory, 'ensemble_history.json'))

    def _finish_experiment(self, proc_ensemble: Optional[EnsembleBuilderManager],
                           load_models: bool) -> None:

        assert self._logger is not None
        # Wait until the ensemble process is finished to avoid shutting down
        # while the ensemble builder tries to access the data
        self._logger.info("Start Shutdown")

        if proc_ensemble is not None:
            self.ensemble_performance_history = list(proc_ensemble.history)
            self._save_ensemble_performance_history(proc_ensemble)

        self._logger.info("Close the dask infrastructure")
        self._close_dask_client()
        self._logger.info("Finish closing the dask infrastructure")

        if load_models:
            self._logger.info("Load models...")
            self._load_models()
            self._logger.info("Finish loading models...")

        # Clean up the logger
        self._logger.info("Start to clean up the logger")
        self._clean_logger()

    def _search(
        self,
        optimize_metric: str,
        dataset: BaseDataset,
        budget_type: Optional[str] = None,
        budget: Optional[float] = None,
        total_walltime_limit: int = 100,
        func_eval_time_limit_secs: Optional[int] = None,
        enable_traditional_pipeline: bool = True,
        memory_limit: Optional[int] = 4096,
        smac_scenario_args: Optional[Dict[str, Any]] = None,
        get_smac_object_callback: Optional[Callable] = None,
        all_supported_metrics: bool = True,
        precision: int = 32,
        disable_file_output: List = [],
        load_models: bool = True
    ) -> 'BaseTask':
        """
        Search for the best pipeline configuration for the given dataset.

        Fit both optimizes the machine learning models and builds an ensemble out of them.
        To disable ensembling, set ensemble_size==0.
        using the optimizer.
        Args:
            dataset (Dataset):
                The argument that will provide the dataset splits. It is
                a subclass of the  base dataset object which can
                generate the splits based on different restrictions.
                Providing X_train, y_train and dataset together is not supported.
            optimize_metric (str): name of the metric that is used to
                evaluate a pipeline.
            budget_type (Optional[str]):
                Type of budget to be used when fitting the pipeline.
                Either 'epochs' or 'runtime'. If not provided, uses
                the default in the pipeline config ('epochs')
            budget (Optional[float]):
                Budget to fit a single run of the pipeline. If not
                provided, uses the default in the pipeline config
            total_walltime_limit (int), (default=100): Time limit
                in seconds for the search of appropriate models.
                By increasing this value, autopytorch has a higher
                chance of finding better models.
            func_eval_time_limit_secs (int), (default=None): Time limit
                for a single call to the machine learning model.
                Model fitting will be terminated if the machine
                learning algorithm runs over the time limit. Set
                this value high enough so that typical machine
                learning algorithms can be fit on the training
                data.
                When set to None, this time will automatically be set to
                total_walltime_limit // 2 to allow enough time to fit
                at least 2 individual machine learning algorithms.
                Set to np.inf in case no time limit is desired.
            enable_traditional_pipeline (bool), (default=True):
                We fit traditional machine learning algorithms
                (LightGBM, CatBoost, RandomForest, ExtraTrees, KNN, SVM)
                prior building PyTorch Neural Networks. You can disable this
                feature by turning this flag to False. All machine learning
                algorithms that are fitted during search() are considered for
                ensemble building.
            memory_limit (Optional[int]), (default=4096): Memory
                limit in MB for the machine learning algorithm. autopytorch
                will stop fitting the machine learning algorithm if it tries
                to allocate more than memory_limit MB. If None is provided,
                no memory limit is set. In case of multi-processing, memory_limit
                will be per job. This memory limit also applies to the ensemble
                creation process.
            smac_scenario_args (Optional[Dict]): Additional arguments inserted
                into the scenario of SMAC. See the
                [SMAC documentation] (https://automl.github.io/SMAC3/master/options.html?highlight=scenario#scenario)
                for a list of available arguments.
            get_smac_object_callback (Optional[Callable]): Callback function
                to create an object of class
                [smac.optimizer.smbo.SMBO](https://automl.github.io/SMAC3/master/apidoc/smac.optimizer.smbo.html).
                The function must accept the arguments scenario_dict,
                instances, num_params, runhistory, seed and ta. This is
                an advanced feature. Use only if you are familiar with
                [SMAC](https://automl.github.io/SMAC3/master/index.html).
            all_supported_metrics (bool), (default=True): if True, all
                metrics supporting current task will be calculated
                for each pipeline and results will be available via cv_results
            precision (int), (default=32): Numeric precision used when loading
                ensemble data. Can be either '16', '32' or '64'.
            disable_file_output (Union[bool, List]):
            load_models (bool), (default=True): Whether to load the
                models after fitting AutoPyTorch.

        Returns:
            self

        """

        if self.task_type != dataset.task_type:
            raise ValueError("Incompatible dataset entered for current task,"
                             "expected dataset to have task type :{} got "
                             ":{}".format(self.task_type, dataset.task_type))
        if self.task_type is None:
            raise ValueError("Cannot interpret task type from the dataset")
        if precision not in [16, 32, 64]:
            raise ValueError(f"precision must be either [16, 32, 64], but got {precision}")

        self._search_settings(dataset=dataset, disable_file_output=disable_file_output,
                              optimize_metric=optimize_metric, memory_limit=memory_limit,
                              all_supported_metrics=all_supported_metrics,
                              func_eval_time_limit_secs=func_eval_time_limit_secs,
                              total_walltime_limit=total_walltime_limit)

        self._adapt_time_resource_allocation()
        self._run_dummy_predictions()

        if enable_traditional_pipeline:
            self._run_traditional_ml()

        proc_ensemble = self._run_ensemble(dataset=dataset, precision=precision,
                                           optimize_metric=optimize_metric)

        self._run_smac(budget=budget, budget_type=budget_type, proc_ensemble=proc_ensemble,
                       dataset=dataset, get_smac_object_callback=get_smac_object_callback,
                       smac_scenario_args=smac_scenario_args)

        self._finish_experiment(proc_ensemble=proc_ensemble, load_models=load_models)

        return self

    def refit(
            self,
            dataset: BaseDataset,
            budget_config: Dict[str, Union[int, str]] = {},
            split_id: int = 0
    ) -> "BaseTask":
        """
        Refit all models found with fit to new data.

        Necessary when using cross-validation. During training, autoPyTorch
        fits each model k times on the dataset, but does not keep any trained
        model and can therefore not be used to predict for new data points.
        This methods fits all models found during a call to fit on the data
        given. This method may also be used together with holdout to avoid
        only using 66% of the training data to fit the final model.
        Args:
            dataset: (Dataset)
                The argument that will provide the dataset splits. It can either
                be a dictionary with the splits, or the dataset object which can
                generate the splits based on different restrictions.
            budget_config: (Optional[Dict[str, Union[int, str]]])
                can contain keys from 'budget_type' and the budget
                specified using 'epochs' or 'runtime'.
            split_id: (int)
                split id to fit on.
        Returns:
            self
        """
        if self.dataset_name == "":
            self.dataset_name = str(uuid.uuid1(clock_seq=os.getpid()))

        if self._logger is None:
            self._logger = self._get_logger(self.dataset_name)

        dataset_requirements = get_dataset_requirements(
            info=self._get_required_dataset_properties(dataset))
        dataset_properties = dataset.get_dataset_properties(dataset_requirements)
        self._backend.save_datamanager(dataset)

        X: Dict[str, Any] = dict({'dataset_properties': dataset_properties,
                                  'backend': self._backend,
                                  'X_train': dataset.train_tensors[0],
                                  'y_train': dataset.train_tensors[1],
                                  'X_test': dataset.test_tensors[0] if dataset.test_tensors is not None else None,
                                  'y_test': dataset.test_tensors[1] if dataset.test_tensors is not None else None,
                                  'train_indices': dataset.splits[split_id][0],
                                  'val_indices': dataset.splits[split_id][1],
                                  'split_id': split_id,
                                  'num_run': self._backend.get_next_num_run(),
                                  })
        X.update({**self.pipeline_options, **budget_config})
        if self.models_ is None or len(self.models_) == 0 or self.ensemble_ is None:
            self._load_models()

        # Refit is not applicable when ensemble_size is set to zero.
        if self.ensemble_ is None:
            raise ValueError("Refit can only be called if 'ensemble_size != 0'")

        for identifier in self.models_:
            model = self.models_[identifier]
            # It updates the model inplace, it can then later be used in
            # predict method

            # Fit the model to check if it fails.
            # If it fails, shuffle the data to alleviate
            # the ordering-of-the-data issue in algorithms
            fit_and_suppress_warnings(self._logger, model, X, y=None)

        self._clean_logger()

        return self

    def fit(self,
            dataset: BaseDataset,
            budget_config: Dict[str, Union[int, str]] = {},
            pipeline_config: Optional[Configuration] = None,
            split_id: int = 0) -> BasePipeline:
        """
        Fit a pipeline on the given task for the budget.
        A pipeline configuration can be specified if None,
        uses default
        Args:
            dataset: (Dataset)
                The argument that will provide the dataset splits. It can either
                be a dictionary with the splits, or the dataset object which can
                generate the splits based on different restrictions.
            budget_config: (Optional[Dict[str, Union[int, str]]])
                can contain keys from 'budget_type' and the budget
                specified using 'epochs' or 'runtime'.
            split_id: (int) (default=0)
                split id to fit on.
            pipeline_config: (Optional[Configuration])
                configuration to fit the pipeline with. If None,
                uses default

        Returns:
            (BasePipeline): fitted pipeline
        """
        if self.dataset_name == "":
            self.dataset_name = str(uuid.uuid1(clock_seq=os.getpid()))

        if self._logger is None:
            self._logger = self._get_logger(self.dataset_name)

        # get dataset properties
        dataset_requirements = get_dataset_requirements(
            info=self._get_required_dataset_properties(dataset))
        dataset_properties = dataset.get_dataset_properties(dataset_requirements)
        self._backend.save_datamanager(dataset)

        # build pipeline
        pipeline = self.build_pipeline(dataset_properties)
        if pipeline_config is not None:
            pipeline.set_hyperparameters(pipeline_config)

        # initialise fit dictionary
        X: Dict[str, Any] = dict({'dataset_properties': dataset_properties,
                                  'backend': self._backend,
                                  'X_train': dataset.train_tensors[0],
                                  'y_train': dataset.train_tensors[1],
                                  'X_test': dataset.test_tensors[0] if dataset.test_tensors is not None else None,
                                  'y_test': dataset.test_tensors[1] if dataset.test_tensors is not None else None,
                                  'train_indices': dataset.splits[split_id][0],
                                  'val_indices': dataset.splits[split_id][1],
                                  'split_id': split_id,
                                  'num_run': self._backend.get_next_num_run(),
                                  })
        X.update({**self.pipeline_options, **budget_config})

        fit_and_suppress_warnings(self._logger, pipeline, X, y=None)

        self._clean_logger()
        return pipeline

    def predict(
            self,
            X_test: np.ndarray,
            batch_size: Optional[int] = None,
            n_jobs: int = 1
    ) -> np.ndarray:
        """Generate the estimator predictions.
        Generate the predictions based on the given examples from the test set.
        Args:
        X_test: (np.ndarray)
            The test set examples.
        Returns:
            Array with estimator predictions.
        """

        # Parallelize predictions across models with n_jobs processes.
        # Each process computes predictions in chunks of batch_size rows.
        if self._logger is None:
            self._logger = self._get_logger("Predict-Logger")

        if self.ensemble_ is None and not self._load_models():
            raise ValueError("No ensemble found. Either fit has not yet "
                             "been called or no ensemble was fitted")

        # Mypy assert
        assert self.ensemble_ is not None, "Load models should error out if no ensemble"
        self.ensemble_ = cast(Union[SingleBest, EnsembleSelection], self.ensemble_)

        if isinstance(self.resampling_strategy, HoldoutValTypes):
            models = self.models_
        elif isinstance(self.resampling_strategy, CrossValTypes):
            models = self.cv_models_

        all_predictions = joblib.Parallel(n_jobs=n_jobs)(
            joblib.delayed(_pipeline_predict)(
                models[identifier], X_test, batch_size, self._logger, STRING_TO_TASK_TYPES[self.task_type]
            )
            for identifier in self.ensemble_.get_selected_model_identifiers()
        )

        if len(all_predictions) == 0:
            raise ValueError('Something went wrong generating the predictions. '
                             'The ensemble should consist of the following '
                             'models: %s, the following models were loaded: '
                             '%s' % (str(list(self.ensemble_.indices_)),
                                     str(list(self.models_))))

        predictions = self.ensemble_.predict(all_predictions)

        self._clean_logger()

        return predictions

    def score(
            self,
            y_pred: np.ndarray,
            y_test: Union[np.ndarray, pd.DataFrame]
    ) -> Dict[str, float]:
        """Calculate the score on the test set.
        Calculate the evaluation measure on the test set.
        Args:
        y_pred: (np.ndarray)
            The test predictions
        y_test: (np.ndarray)
            The test ground truth labels.
        Returns:
            Dict[str, float]: Value of the evaluation metric calculated on the test set.
        """
        if self._metric is None:
            raise ValueError("No metric found. Either fit/search has not been called yet "
                             "or AutoPyTorch failed to infer a metric from the dataset ")
        if self.task_type is None:
            raise ValueError("AutoPytorch failed to infer a task type from the dataset "
                             "Please check the log file for related errors. ")
        return calculate_score(target=y_test, prediction=y_pred,
                               task_type=STRING_TO_TASK_TYPES[self.task_type],
                               metrics=[self._metric])

    def __getstate__(self) -> Dict[str, Any]:
        # Cannot serialize a client!
        self._dask_client = None
        self.logging_server = None  # type: ignore [assignment]
        self.stop_logging_server = None
        return self.__dict__

    def __del__(self) -> None:
        # Clean up the logger
        self._clean_logger()

        self._close_dask_client()

        # When a multiprocessing work is done, the
        # objects are deleted. We don't want to delete run areas
        # until the estimator is deleted
        self._backend.context.delete_directories(force=False)

    @typing.no_type_check
    def get_incumbent_results(self):
        pass

    @typing.no_type_check
    def get_incumbent_config(self):
        pass

    def get_models_with_weights(self) -> List:
        if self.models_ is None or len(self.models_) == 0 or \
                self.ensemble_ is None:
            self._load_models()

        assert self.ensemble_ is not None
        return self.ensemble_.get_models_with_weights(self.models_)

    def show_models(self) -> str:
        df = []
        for weight, model in self.get_models_with_weights():
            representation = model.get_pipeline_representation()
            representation.update({'Weight': weight})
            df.append(representation)
        return pd.DataFrame(df).to_markdown()

    def _print_debug_info_to_log(self) -> None:
        """
        Prints to the log file debug information about the current estimator
        """
        assert self._logger is not None
        self._logger.debug("Start to print environment information")
        self._logger.debug('  Python version: %s', sys.version.split('\n'))
        self._logger.debug('  System: %s', platform.system())
        self._logger.debug('  Machine: %s', platform.machine())
        self._logger.debug('  Platform: %s', platform.platform())
        for key, value in vars(self).items():
            self._logger.debug(f"\t{key}->{value}")
