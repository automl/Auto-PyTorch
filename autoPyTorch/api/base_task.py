import copy
import json
import logging.handlers
import math
import multiprocessing
import os
import platform
import sys
import tempfile
import time
import typing
import unittest.mock
import warnings
from abc import abstractmethod
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, cast

from ConfigSpace.configuration_space import Configuration, ConfigurationSpace

import dask

import joblib

import numpy as np

import pandas as pd

from smac.runhistory.runhistory import DataOrigin, RunHistory, RunInfo, RunValue
from smac.stats.stats import Stats
from smac.tae import StatusType

from autoPyTorch.constants import (
    REGRESSION_TASKS,
    STRING_TO_OUTPUT_TYPES,
    STRING_TO_TASK_TYPES,
)
from autoPyTorch.datasets.base_dataset import BaseDataset
from autoPyTorch.datasets.resampling_strategy import CrossValTypes, HoldoutValTypes, NoResamplingStrategyTypes
from autoPyTorch.ensemble.ensemble_builder import EnsembleBuilderManager
from autoPyTorch.ensemble.ensemble_selection import EnsembleSelection
from autoPyTorch.ensemble.singlebest_ensemble import SingleBest
from autoPyTorch.evaluation.abstract_evaluator import fit_and_suppress_warnings
from autoPyTorch.evaluation.tae import ExecuteTaFuncWithQueue, get_cost_of_crash
from autoPyTorch.optimizer.smbo import AutoMLSMBO
from autoPyTorch.pipeline.base_pipeline import BasePipeline
from autoPyTorch.pipeline.components.setup.traditional_ml.classifier_models import get_available_classifiers
from autoPyTorch.pipeline.components.training.metrics.base import autoPyTorchMetric
from autoPyTorch.pipeline.components.training.metrics.utils import calculate_score, get_metrics
from autoPyTorch.utils.backend import Backend, create
from autoPyTorch.utils.common import FitRequirement, replace_string_bool_to_bool
from autoPyTorch.utils.hyperparameter_search_space_update import HyperparameterSearchSpaceUpdates
from autoPyTorch.utils.logging_ import (
    PicklableClientLogger,
    get_named_client_logger,
    setup_logger,
    start_log_server,
)
from autoPyTorch.utils.pipeline import get_configuration_space, get_dataset_requirements
from autoPyTorch.utils.stopwatch import StopWatch


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
        search_space_updates (Optional[HyperparameterSearchSpaceUpdates]): updates to be made
            to the hyperparameter search space of the pipeline
        resampling_strategy (Union[CrossValTypes, HoldoutValTypes, NoResamplingStrategyTypes]),
                (default=HoldoutValTypes.holdout_validation):
                strategy to split the training data.
        resampling_strategy_args (Optional[Dict[str, Any]]): arguments
            required for the chosen resampling strategy. If None, uses
            the default values provided in DEFAULT_RESAMPLING_PARAMETERS
            in ```datasets/resampling_strategy.py```.
        task_type (str): The task of the experiment as a string. Currently, supported
            tasks are 'tabular_classification' and 'tabular_regression'
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
        resampling_strategy: Union[CrossValTypes,
                                   HoldoutValTypes,
                                   NoResamplingStrategyTypes] = HoldoutValTypes.holdout_validation,
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
        self._dataset_requirements: Optional[List[FitRequirement]] = None
        self._metric: Optional[autoPyTorchMetric] = None
        self._logger: Optional[PicklableClientLogger] = None
        self.run_history: RunHistory = RunHistory()
        self.trajectory: Optional[List] = None
        self.dataset_name: Optional[str] = None
        self.cv_models_: Dict = {}

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
    def build_pipeline(self, dataset_properties: Dict[str, Any],
                       include_components: Optional[Dict] = None,
                       exclude_components: Optional[Dict] = None,
                       search_space_updates: Optional[HyperparameterSearchSpaceUpdates] = None
                       ) -> BasePipeline:
        """
        Build pipeline according to current task
        and for the passed dataset properties
        Args:
            dataset_properties (Dict[str,Any]):

        Returns:

        """

        raise NotImplementedError("Function called on BaseTask, this can only be called by "
                                  "specific task which is a child of the BaseTask")

    @abstractmethod
    def get_dataset(self,
                    X_train: Union[List, pd.DataFrame, np.ndarray],
                    y_train: Union[List, pd.DataFrame, np.ndarray],
                    X_test: Union[List, pd.DataFrame, np.ndarray],
                    y_test: Union[List, pd.DataFrame, np.ndarray],
                    resampling_strategy: Optional[Union[CrossValTypes,
                                                        HoldoutValTypes,
                                                        NoResamplingStrategyTypes]] = None,
                    resampling_strategy_args: Optional[Dict[str, Any]] = None,
                    dataset_name: Optional[str] = None,
                    return_only: Optional[bool] = False
                    ) -> BaseDataset:
        raise NotImplementedError("Function called on BaseTask, this can only be called by "
                                  "specific task which is a child of the BaseTask")

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

        if isinstance(self._disable_file_output, List):
            disabled_file_outputs = self._disable_file_output
            disable_file_output = False
        elif isinstance(self._disable_file_output, bool):
            disable_file_output = self._disable_file_output
            disabled_file_outputs = []

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

        elif disable_file_output or 'pipeline' not in disabled_file_outputs:
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
        if self._logger is None:
            warnings.warn(
                "No valid ensemble was created. Please check the log"
                "file for errors. Default to the best individual estimator:{}".format(
                    ensemble.identifiers_
                )
            )
        else:
            self._logger.exception(
                "No valid ensemble was created. Please check the log"
                "file for errors. Default to the best individual estimator:{}".format(
                    ensemble.identifiers_
                )
            )

        return ensemble

    def _do_dummy_prediction(self) -> None:

        assert self._metric is not None
        assert self._logger is not None

        # For dummy estimator, we always expect the num_run to be 1
        num_run = 1

        self._logger.info("Starting to create dummy predictions.")

        memory_limit = self._memory_limit
        if memory_limit is not None:
            memory_limit = int(math.ceil(memory_limit))

        scenario_mock = unittest.mock.Mock()
        scenario_mock.wallclock_limit = self._time_for_task
        # This stats object is a hack - maybe the SMAC stats object should
        # already be generated here!
        stats = Stats(scenario_mock)
        stats.start_timing()
        ta = ExecuteTaFuncWithQueue(
            backend=self._backend,
            seed=self.seed,
            metric=self._metric,
            logger_port=self._logger_port,
            cost_for_crash=get_cost_of_crash(self._metric),
            abort_on_first_run_crash=False,
            initial_num_run=num_run,
            stats=stats,
            memory_limit=memory_limit,
            disable_file_output=self._disable_file_output,
            all_supported_metrics=self._all_supported_metrics
        )

        status, cost, runtime, additional_info = ta.run(num_run, cutoff=self._time_for_task)
        if status == StatusType.SUCCESS:
            self._logger.info("Finished creating dummy predictions.")
        else:
            if additional_info.get('exitcode') == -6:
                self._logger.error(
                    "Dummy prediction failed with run state %s. "
                    "The error suggests that the provided memory limits were too tight. Please "
                    "increase the 'ml_memory_limit' and try again. If this does not solve your "
                    "problem, please open an issue and paste the additional output. "
                    "Additional output: %s.",
                    str(status), str(additional_info),
                )
                # Fail if dummy prediction fails.
                raise ValueError(
                    "Dummy prediction failed with run state %s. "
                    "The error suggests that the provided memory limits were too tight. Please "
                    "increase the 'ml_memory_limit' and try again. If this does not solve your "
                    "problem, please open an issue and paste the additional output. "
                    "Additional output: %s." %
                    (str(status), str(additional_info)),
                )

            else:
                self._logger.error(
                    "Dummy prediction failed with run state %s and additional output: %s.",
                    str(status), str(additional_info),
                )
                # Fail if dummy prediction fails.
                raise ValueError(
                    "Dummy prediction failed with run state %s and additional output: %s."
                    % (str(status), str(additional_info))
                )

    def _do_traditional_prediction(self, time_left: int, func_eval_time_limit_secs: int) -> None:
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

        self._logger.info("Starting to create traditional classifier predictions.")

        # Initialise run history for the traditional classifiers
        run_history = RunHistory()
        memory_limit = self._memory_limit
        if memory_limit is not None:
            memory_limit = int(math.ceil(memory_limit))
        available_classifiers = get_available_classifiers()
        dask_futures = []

        total_number_classifiers = len(available_classifiers)
        for n_r, classifier in enumerate(available_classifiers):

            # Only launch a task if there is time
            start_time = time.time()
            if time_left >= func_eval_time_limit_secs:
                self._logger.info(f"{n_r}: Started fitting {classifier} with cutoff={func_eval_time_limit_secs}")
                scenario_mock = unittest.mock.Mock()
                scenario_mock.wallclock_limit = time_left
                # This stats object is a hack - maybe the SMAC stats object should
                # already be generated here!
                stats = Stats(scenario_mock)
                stats.start_timing()
                ta = ExecuteTaFuncWithQueue(
                    backend=self._backend,
                    seed=self.seed,
                    metric=self._metric,
                    logger_port=self._logger_port,
                    cost_for_crash=get_cost_of_crash(self._metric),
                    abort_on_first_run_crash=False,
                    initial_num_run=self._backend.get_next_num_run(),
                    stats=stats,
                    memory_limit=memory_limit,
                    disable_file_output=self._disable_file_output,
                    all_supported_metrics=self._all_supported_metrics
                )
                dask_futures.append([
                    classifier,
                    self._dask_client.submit(
                        ta.run, config=classifier,
                        cutoff=func_eval_time_limit_secs,
                    )
                ])

            # When managing time, we need to take into account the allocated time resources,
            # which are dependent on the number of cores. 'dask_futures' is a proxy to the number
            # of workers /n_jobs that we have, in that if there are 4 cores allocated, we can run at most
            # 4 task in parallel. Every 'cutoff' seconds, we generate up to 4 tasks.
            # If we only have 4 workers and there are 4 futures in dask_futures, it means that every
            # worker has a task. We would not like to launch another job until a worker is available. To this
            # end, the following if-statement queries the number of active jobs, and forces to wait for a job
            # completion via future.result(), so that a new worker is available for the next iteration.
            if len(dask_futures) >= self.n_jobs:

                # How many workers to wait before starting fitting the next iteration
                workers_to_wait = 1
                if n_r >= total_number_classifiers - 1 or time_left <= func_eval_time_limit_secs:
                    # If on the last iteration, flush out all tasks
                    workers_to_wait = len(dask_futures)

                while workers_to_wait >= 1:
                    workers_to_wait -= 1
                    # We launch dask jobs only when there are resources available.
                    # This allow us to control time allocation properly, and early terminate
                    # the traditional machine learning pipeline
                    cls, future = dask_futures.pop(0)
                    status, cost, runtime, additional_info = future.result()
                    if status == StatusType.SUCCESS:
                        self._logger.info(
                            f"Fitting {cls} took {runtime}s, performance:{cost}/{additional_info}")
                        configuration = additional_info['pipeline_configuration']
                        origin = additional_info['configuration_origin']
                        run_history.add(config=configuration, cost=cost,
                                        time=runtime, status=status, seed=self.seed,
                                        origin=origin)
                    else:
                        if additional_info.get('exitcode') == -6:
                            self._logger.error(
                                "Traditional prediction for %s failed with run state %s. "
                                "The error suggests that the provided memory limits were too tight. Please "
                                "increase the 'ml_memory_limit' and try again. If this does not solve your "
                                "problem, please open an issue and paste the additional output. "
                                "Additional output: %s.",
                                cls, str(status), str(additional_info),
                            )
                        else:
                            self._logger.error(
                                "Traditional prediction for %s failed with run state %s and additional output: %s.",
                                cls, str(status), str(additional_info),
                            )

            # In the case of a serial execution, calling submit halts the run for a resource
            # dynamically adjust time in this case
            time_left -= int(time.time() - start_time)

            # Exit if no more time is available for a new classifier
            if time_left < func_eval_time_limit_secs:
                self._logger.warning("Not enough time to fit all traditional machine learning models."
                                     "Please consider increasing the run time to further improve performance.")
                break

        self._logger.debug("Run history traditional: {}".format(run_history))
        # add run history of traditional to api run history
        self.run_history.update(run_history, DataOrigin.EXTERNAL_SAME_INSTANCES)
        run_history.save_json(os.path.join(self._backend.internals_directory, 'traditional_run_history.json'),
                              save_external=True)
        return

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
        disable_file_output: Union[bool, List] = False,
        load_models: bool = True,
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

        # Initialise information needed for the experiment
        experiment_task_name: str = 'runSearch'
        dataset_requirements = get_dataset_requirements(
            info=self._get_required_dataset_properties(dataset))
        self._dataset_requirements = dataset_requirements
        dataset_properties = dataset.get_dataset_properties(dataset_requirements)
        self._stopwatch.start_task(experiment_task_name)
        self.dataset_name = dataset.dataset_name
        assert self.dataset_name is not None

        if self._logger is None:
            self._logger = self._get_logger(self.dataset_name)
        self._all_supported_metrics = all_supported_metrics
        self._disable_file_output = disable_file_output
        self._memory_limit = memory_limit
        self._time_for_task = total_walltime_limit
        # Save start time to backend
        self._backend.save_start_time(str(self.seed))

        self._backend.save_datamanager(dataset)

        # Print debug information to log
        self._print_debug_info_to_log()

        self._metric = get_metrics(
            names=[optimize_metric], dataset_properties=dataset_properties)[0]

        self.search_space = self.get_search_space(dataset)

        budget_config: Dict[str, Union[float, str]] = {}
        if budget_type is not None and budget is not None:
            budget_config['budget_type'] = budget_type
            budget_config[budget_type] = budget
        elif budget_type is not None or budget is not None:
            raise ValueError(
                "budget type was not specified in budget_config"
            )

        if self.task_type is None:
            raise ValueError("Cannot interpret task type from the dataset")

        # If no dask client was provided, we create one, so that we can
        # start a ensemble process in parallel to smbo optimize
        if (
            self._dask_client is None and (self.ensemble_size > 0 or self.n_jobs is not None and self.n_jobs > 1)
        ):
            self._create_dask_client()
        else:
            self._is_dask_client_internally_created = False

        # Handle time resource allocation
        elapsed_time = self._stopwatch.wall_elapsed(experiment_task_name)
        time_left_for_modelfit = int(max(0, total_walltime_limit - elapsed_time))
        if func_eval_time_limit_secs is None or func_eval_time_limit_secs > time_left_for_modelfit:
            self._logger.warning(
                'Time limit for a single run is higher than total time '
                'limit. Capping the limit for a single run to the total '
                'time given to SMAC (%f)' % time_left_for_modelfit
            )
            func_eval_time_limit_secs = time_left_for_modelfit

        # Make sure that at least 2 models are created for the ensemble process
        num_models = time_left_for_modelfit // func_eval_time_limit_secs
        if num_models < 2:
            func_eval_time_limit_secs = time_left_for_modelfit // 2
            self._logger.warning(
                "Capping the func_eval_time_limit_secs to {} to have "
                "time for a least 2 models to ensemble.".format(
                    func_eval_time_limit_secs
                )
            )

        # ============> Run dummy predictions
        dummy_task_name = 'runDummy'
        self._stopwatch.start_task(dummy_task_name)
        self._do_dummy_prediction()
        self._stopwatch.stop_task(dummy_task_name)

        # ============> Run traditional ml

        if enable_traditional_pipeline:
            if STRING_TO_TASK_TYPES[self.task_type] in REGRESSION_TASKS:
                self._logger.warning("Traditional Pipeline is not enabled for regression. Skipping...")
            else:
                traditional_task_name = 'runTraditional'
                self._stopwatch.start_task(traditional_task_name)
                elapsed_time = self._stopwatch.wall_elapsed(self.dataset_name)
                # We want time for at least 1 Neural network in SMAC
                time_for_traditional = int(
                    self._time_for_task - elapsed_time - func_eval_time_limit_secs
                )
                self._do_traditional_prediction(
                    func_eval_time_limit_secs=func_eval_time_limit_secs,
                    time_left=time_for_traditional,
                )
                self._stopwatch.stop_task(traditional_task_name)

        # ============> Starting ensemble
        elapsed_time = self._stopwatch.wall_elapsed(self.dataset_name)
        time_left_for_ensembles = max(0, total_walltime_limit - elapsed_time)
        proc_ensemble = None
        if time_left_for_ensembles <= 0:
            # Fit only raises error when ensemble_size is not zero but
            # time_left_for_ensembles is zero.
            if self.ensemble_size > 0:
                raise ValueError("Not starting ensemble builder because there "
                                 "is no time left. Try increasing the value "
                                 "of time_left_for_this_task.")
        elif self.ensemble_size <= 0:
            self._logger.info("Not starting ensemble builder as ensemble size is 0")
        else:
            self._logger.info("Starting ensemble")
            ensemble_task_name = 'ensemble'
            self._stopwatch.start_task(ensemble_task_name)
            proc_ensemble = EnsembleBuilderManager(
                start_time=time.time(),
                time_left_for_ensembles=time_left_for_ensembles,
                backend=copy.deepcopy(self._backend),
                dataset_name=str(dataset.dataset_name),
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
                logger_port=self._logger_port,
            )
            self._stopwatch.stop_task(ensemble_task_name)

        # ==> Run SMAC
        smac_task_name: str = 'runSMAC'
        self._stopwatch.start_task(smac_task_name)
        elapsed_time = self._stopwatch.wall_elapsed(experiment_task_name)
        time_left_for_smac = max(0, total_walltime_limit - elapsed_time)

        self._logger.info("Starting SMAC with %5.2f sec time left" % time_left_for_smac)
        if time_left_for_smac <= 0:
            self._logger.warning(" Not starting SMAC because there is no time left")
        else:

            _proc_smac = AutoMLSMBO(
                config_space=self.search_space,
                dataset_name=str(dataset.dataset_name),
                backend=self._backend,
                total_walltime_limit=total_walltime_limit,
                func_eval_time_limit_secs=func_eval_time_limit_secs,
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
                # We do not increase the num_run here, this is something
                # smac does internally
                start_num_run=self._backend.get_next_num_run(peek=True),
                search_space_updates=self.search_space_updates
            )
            try:
                run_history, self.trajectory, budget_type = \
                    _proc_smac.run_smbo()
                self.run_history.update(run_history, DataOrigin.INTERNAL)
                trajectory_filename = os.path.join(
                    self._backend.get_smac_output_directory_for_run(self.seed),
                    'trajectory.json')
                saveable_trajectory = \
                    [list(entry[:2]) + [entry[2].get_dictionary()] + list(entry[3:])
                     for entry in self.trajectory]
                try:
                    with open(trajectory_filename, 'w') as fh:
                        json.dump(saveable_trajectory, fh)
                except Exception as e:
                    self._logger.warning(f"Cannot save {trajectory_filename} due to {e}...")
            except Exception as e:
                self._logger.exception(str(e))
                raise
        # Wait until the ensemble process is finished to avoid shutting down
        # while the ensemble builder tries to access the data
        self._logger.info("Starting Shutdown")

        if proc_ensemble is not None:
            self.ensemble_performance_history = list(proc_ensemble.history)

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

        self._logger.info("Closing the dask infrastructure")
        self._close_dask_client()
        self._logger.info("Finished closing the dask infrastructure")

        if load_models:
            self._logger.info("Loading models...")
            self._load_models()
            self._logger.info("Finished loading models...")

        # Clean up the logger
        self._logger.info("Starting to clean up the logger")
        self._clean_logger()

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

        self.dataset_name = dataset.dataset_name

        if self._logger is None:
            self._logger = self._get_logger(str(self.dataset_name))

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
            # this updates the model inplace, it can then later be used in
            # predict method

            # try to fit the model. If it fails, shuffle the data. This
            # could alleviate the problem in algorithms that depend on
            # the ordering of the data.
            fit_and_suppress_warnings(self._logger, model, X, y=None)

        self._clean_logger()

        return self

    def fit_pipeline(self,
                     configuration: Configuration,
                     dataset: Optional[BaseDataset] = None,
                     X_train: Optional[Union[List, pd.DataFrame, np.ndarray]] = None,
                     y_train: Optional[Union[List, pd.DataFrame, np.ndarray]] = None,
                     X_test: Optional[Union[List, pd.DataFrame, np.ndarray]] = None,
                     y_test: Optional[Union[List, pd.DataFrame, np.ndarray]] = None,
                     dataset_name: Optional[str] = None,
                     resampling_strategy: Optional[Union[HoldoutValTypes,
                                                         CrossValTypes,
                                                         NoResamplingStrategyTypes]] = None,
                     resampling_strategy_args: Optional[Dict[str, Any]] = None,
                     run_time_limit_secs: int = 60,
                     memory_limit: Optional[int] = None,
                     eval_metric: Optional[str] = None,
                     all_supported_metrics: bool = False,
                     budget_type: Optional[str] = None,
                     include_components: Optional[Dict] = None,
                     exclude_components: Optional[Dict] = None,
                     search_space_updates: Optional[HyperparameterSearchSpaceUpdates] = None,
                     budget: float = 50,
                     pipeline_options: Optional[Dict] = None,
                     disable_file_output: Optional[Union[bool, List]] = False,
                     return_dataset: bool = True
                     ) -> Tuple[Optional[BasePipeline], RunInfo, RunValue, Optional[BaseDataset]]:

        """
        Fit a pipeline on the given task for the budget.
        A pipeline configuration can be specified if None,
        uses default

        Args:
            X_train, y_train, X_test, y_test: Union[np.ndarray, List, pd.DataFrame]
                A pair of features (X_train) and targets (y_train) used to fit a
                pipeline. Additionally, a holdout of this pairs (X_test, y_test) can
                be provided to track the generalization performance of each stage.
            dataset_name (Optional[str]):
                Name of the dataset, if None, random value is used.
            resampling_strategy (Union[CrossValTypes, HoldoutValTypes, NoResamplingStrategyTypes]),
                (default=HoldoutValTypes.holdout_validation):
                strategy to split the training data.
            resampling_strategy_args (Optional[Dict[str, Any]]): arguments
                required for the chosen resampling strategy. If None, uses
                the default values provided in DEFAULT_RESAMPLING_PARAMETERS
                in ```datasets/resampling_strategy.py```.
            run_time_limit_secs (int), (default=120): Time limit
                for a single call to the machine learning model.
                Model fitting will be terminated if the machine
                learning algorithm runs over the time limit. Set
                this value high enough so that typical machine
                learning algorithms can be fit on the training
                data.
            memory_limit (Optional[int]), (default=None): Memory
                limit in MB for the machine learning algorithm. autopytorch
                will stop fitting the machine learning algorithm if it tries
                to allocate more than memory_limit MB. If None is provided,
                no memory limit is set. In case of multi-processing, memory_limit
                will be per job. This memory limit also applies to the ensemble
                creation process.
            eval_metric (str): name of the metric that is used to
                evaluate a pipeline.
            all_supported_metrics (bool), (default=True): if True, all
                metrics supporting current task will be calculated
                for each pipeline and results will be available via cv_results
            budget_type (Optional[str]):
                Type of budget to be used when fitting the pipeline.
                Either 'epochs' or 'runtime'. If not provided, uses
                the default in the pipeline config ('epochs')
            include_components (Optional[Dict]): If None, all possible components are used.
                Otherwise specifies set of components to use.
            exclude_components (Optional[Dict]): If None, all possible components are used.
                Otherwise specifies set of components not to use. Incompatible with include
                components
            search_space_updates(Optional[HyperparameterSearchSpaceUpdates]): updates to be made
                to the hyperparameter search space of the pipeline
            budget (Optional[float]):
                Budget to fit a single run of the pipeline. If not
                provided, uses the default in the pipeline config
            pipeline_options (Optional[Dict]):
                Valid config options include "device",
                "torch_num_threads", "early_stopping", "use_tensorboard_logger",
                "metrics_during_training"
            disable_file_output (Optional[Union[bool, List]]):
                By default, the model, it's predictions and other metadata is stored on disk
                for each finished configuration. This argument allows the user to skip
                saving certain file type, for example the model, from being written to disk.
            configuration: (Optional[Configuration])
                configuration to fit the pipeline with.

        Returns:
            (BasePipeline): fitted pipeline
            (RunInfo): Run information
            (RunValue): Result of fitting the pipeline
            (BaseDataset): Dataset created from the given tensors
        """
        if dataset is None:
            assert X_train is not None or \
                   y_train is not None or \
                   X_test is not None or \
                   y_test is not None, "No dataset provided, must provide X_train, y_train, X_test, y_test tensors"
            dataset = self.get_dataset(X_train=X_train,
                                       y_train=y_train,
                                       X_test=X_test,
                                       y_test=y_test,
                                       resampling_strategy=resampling_strategy,
                                       resampling_strategy_args=resampling_strategy_args,
                                       dataset_name=dataset_name,
                                       return_only=True)

        # TAE expects each configuration to have a config_id.
        # For fitting a pipeline as it is not part of the
        # search process, it makes sense to set it to 0
        if hasattr(configuration, 'config_id') or configuration.config_id is None:
            configuration.__setattr__('config_id', 0)

        # get dataset properties
        dataset_requirements = get_dataset_requirements(
            info=self._get_required_dataset_properties(dataset))
        dataset_properties = dataset.get_dataset_properties(dataset_requirements)
        self._backend.save_datamanager(dataset)

        if self._logger is None:
            self._logger = self._get_logger(dataset.dataset_name)

        # build pipeline
        if include_components is None:
            include_components = self.include_components
        if exclude_components is None:
            exclude_components = self.exclude_components
        if search_space_updates is None:
            search_space_updates = self.search_space_updates

        scenario_mock = unittest.mock.Mock()
        scenario_mock.wallclock_limit = run_time_limit_secs
        # This stats object is a hack - maybe the SMAC stats object should
        # already be generated here!
        stats = Stats(scenario_mock)

        if memory_limit is None:
            if hasattr(self, '_memory_limit') and self._memory_limit is not None:
                memory_limit = self._memory_limit

        metric = get_metrics(dataset_properties=dataset_properties,
                             names=[eval_metric] if eval_metric is not None else None,
                             all_supported_metrics=False).pop()

        pipeline_options = self.pipeline_options.copy().update(pipeline_options) if pipeline_options is not None \
            else self.pipeline_options.copy()
        if budget_type is not None:
            assert pipeline_options is not None
            pipeline_options.update({'budget_type': budget_type})
        if disable_file_output is None:
            disable_file_output = self._disable_file_output
        stats.start_timing()
        tae = ExecuteTaFuncWithQueue(
            backend=self._backend,
            seed=self.seed,
            metric=metric,
            logger_port=self._logger_port,
            cost_for_crash=get_cost_of_crash(metric),
            abort_on_first_run_crash=False,
            initial_num_run=self._backend.get_next_num_run(),
            stats=stats,
            memory_limit=memory_limit,
            disable_file_output=disable_file_output,
            all_supported_metrics=all_supported_metrics,
            budget_type=budget_type,
            include=include_components,
            exclude=exclude_components,
            search_space_updates=search_space_updates,
            pipeline_config=pipeline_options
        )

        run_info, run_value = tae.run_wrapper(
            RunInfo(config=configuration,
                    budget=budget,
                    seed=self.seed,
                    cutoff=run_time_limit_secs,
                    capped=False,
                    instance_specific=None,
                    instance=None)
        )
        disabled_file_outputs: List = []
        if isinstance(disable_file_output, bool):
            disable_file_output = disable_file_output
        elif isinstance(disable_file_output, List):
            disabled_file_outputs = disable_file_output
        else:
            raise ValueError('disable_file_output should be either a bool or a list')

        fitted_pipeline: Optional[BasePipeline] = None
        if disable_file_output or 'pipeline' in disabled_file_outputs:
            self._logger.warning("File output is disabled. No pipeline can returned")
        elif run_value.status == StatusType.SUCCESS:
            if self.resampling_strategy in CrossValTypes:
                load_function = self._backend.load_cv_model_by_seed_and_id_and_budget
            else:
                load_function = self._backend.load_model_by_seed_and_id_and_budget
            fitted_pipeline = load_function(
                seed=self.seed,
                idx=run_info.config.config_id + tae.initial_num_run,
                budget=float(run_info.budget),
            )

        self._clean_logger()

        if not return_dataset:
            dataset = None  # type: ignore [assignment]

        return fitted_pipeline, run_info, run_value, dataset

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
        if hasattr(self, '_backend'):
            self._backend.context.delete_directories(force=False)

    @typing.no_type_check
    def get_incumbent_results(
        self
    ):
        pass

    @typing.no_type_check
    def get_incumbent_config(
        self
    ):
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
        self._logger.debug("Starting to print environment information")
        self._logger.debug('  Python version: %s', sys.version.split('\n'))
        self._logger.debug('  System: %s', platform.system())
        self._logger.debug('  Machine: %s', platform.machine())
        self._logger.debug('  Platform: %s', platform.platform())
        self._logger.debug(f" Workers: {self._dask_client.dashboard_link}")
        for key, value in vars(self).items():
            self._logger.debug(f"\t{key}->{value}")
