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
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from ConfigSpace.configuration_space import Configuration, ConfigurationSpace

import dask
import dask.distributed

import joblib

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

from smac.runhistory.runhistory import DataOrigin, RunHistory
from smac.stats.stats import Stats
from smac.tae import StatusType

from autoPyTorch import metrics
from autoPyTorch.api.results_manager import MetricResults, ResultsManager, SearchResults
from autoPyTorch.api.run_history_visualizer import ColorLabelSettings, PlotSettingParams, RunHistoryVisualizer
from autoPyTorch.automl_common.common.utils.backend import Backend, create
from autoPyTorch.constants import (
    REGRESSION_TASKS,
    STRING_TO_OUTPUT_TYPES,
    STRING_TO_TASK_TYPES,
)
from autoPyTorch.data.base_validator import BaseInputValidator
from autoPyTorch.datasets.base_dataset import BaseDataset, BaseDatasetPropertiesType
from autoPyTorch.datasets.resampling_strategy import CrossValTypes, HoldoutValTypes
from autoPyTorch.ensemble.ensemble_builder import EnsembleBuilderManager
from autoPyTorch.ensemble.singlebest_ensemble import SingleBest
from autoPyTorch.evaluation.abstract_evaluator import fit_and_suppress_warnings
from autoPyTorch.evaluation.tae import ExecuteTaFuncWithQueue, get_cost_of_crash
from autoPyTorch.optimizer.smbo import AutoMLSMBO
from autoPyTorch.pipeline.base_pipeline import BasePipeline
from autoPyTorch.pipeline.components.setup.traditional_ml.traditional_learner import get_available_traditional_learners
from autoPyTorch.pipeline.components.training.metrics.base import autoPyTorchMetric
from autoPyTorch.pipeline.components.training.metrics.utils import calculate_score, get_metrics
from autoPyTorch.utils.common import FitRequirement, dict_repr, replace_string_bool_to_bool
from autoPyTorch.utils.hyperparameter_search_space_update import HyperparameterSearchSpaceUpdates
from autoPyTorch.utils.logging_ import (
    PicklableClientLogger,
    get_named_client_logger,
    setup_logger,
    start_log_server,
)
from autoPyTorch.utils.parallel import preload_modules
from autoPyTorch.utils.pipeline import get_configuration_space, get_dataset_requirements
from autoPyTorch.utils.single_thread_client import SingleThreadedClient
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
        seed (int: default=1):
            Seed to be used for reproducibility.
        n_jobs (int: default=1):
            Number of consecutive processes to spawn.
        n_threads (int: default=1):
            Number of threads to use for each process.
        logging_config (Optional[Dict]):
            Specifies configuration for logging, if None, it is loaded from the logging.yaml
        ensemble_size (int: default=50):
            Number of models added to the ensemble built by
            Ensemble selection from libraries of models.
            Models are drawn with replacement.
        ensemble_nbest (int: default=50):
            Only consider the ensemble_nbest models to build the ensemble
        max_models_on_disc (int: default=50):
            Maximum number of models saved to disc. It also controls the size of
            the ensemble as any additional models will be deleted.
            Must be greater than or equal to 1.
        temporary_directory (str):
            Folder to store configuration output and log file
        output_directory (str):
            Folder to store predictions for optional test set
        delete_tmp_folder_after_terminate (bool):
            Determines whether to delete the temporary directory,
            when finished
        include_components (Optional[Dict]):
            If None, all possible components are used.
            Otherwise specifies set of components to use.
        exclude_components (Optional[Dict]):
            If None, all possible components are used.
            Otherwise specifies set of components not to use.
            Incompatible with include components
        search_space_updates (Optional[HyperparameterSearchSpaceUpdates]):
            Search space updates that can be used to modify the search
            space of particular components or choice modules of the pipeline
    """

    def __init__(
        self,
        seed: int = 1,
        n_jobs: int = 1,
        n_threads: int = 1,
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
        self.n_threads = n_threads
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
                prefix='autoPyTorch',
                temporary_directory=self._temporary_directory,
                output_directory=self._output_directory,
                delete_tmp_folder_after_terminate=delete_tmp_folder_after_terminate,
                delete_output_folder_after_terminate=delete_output_folder_after_terminate,
            )
        self.task_type = task_type or ""
        self._stopwatch = StopWatch()

        self.pipeline_options = replace_string_bool_to_bool(json.load(open(
            os.path.join(os.path.dirname(__file__), '../configs/default_pipeline_options.json'))))

        self._visualizer = RunHistoryVisualizer()

        self.search_space: Optional[ConfigurationSpace] = None
        self._dataset_requirements: Optional[List[FitRequirement]] = None
        self._metric: Optional[autoPyTorchMetric] = None
        self._scoring_functions: Optional[List[autoPyTorchMetric]] = None
        self._logger: Optional[PicklableClientLogger] = None
        self.dataset_name: Optional[str] = None
        self.cv_models_: Dict = {}

        self._results_manager = ResultsManager()

        # By default try to use the TCP logging port or get a new port
        self._logger_port = logging.handlers.DEFAULT_TCP_LOGGING_PORT

        # Store the resampling strategy from the dataset, to load models as needed
        self.resampling_strategy = resampling_strategy
        self.resampling_strategy_args = resampling_strategy_args

        self.stop_logging_server: Optional[multiprocessing.synchronize.Event] = None

        # Single core, local runs should use fork
        # to prevent the __main__ requirements in
        # examples. Nevertheless, multi-process runs
        # have spawn as requirement to reduce the
        # possibility of a deadlock
        self._dask_client: Optional[dask.distributed.Client] = None
        self._multiprocessing_context = 'forkserver'
        if self.n_jobs == 1:
            self._multiprocessing_context = 'fork'

        self.InputValidator: Optional[BaseInputValidator] = None

        self.search_space_updates = search_space_updates
        if search_space_updates is not None:
            if not isinstance(self.search_space_updates,
                              HyperparameterSearchSpaceUpdates):
                raise ValueError("Expected search space updates to be of instance"
                                 " HyperparameterSearchSpaceUpdates got {}".format(type(self.search_space_updates)))

    @abstractmethod
    def build_pipeline(self, dataset_properties: Dict[str, Any]) -> BasePipeline:
        """
        Build pipeline according to current task
        and for the passed dataset properties

        Args:
            dataset_properties (Dict[str,Any])

        Returns:

        """
        raise NotImplementedError

    @property
    def run_history(self) -> RunHistory:
        return self._results_manager.run_history

    @property
    def ensemble_performance_history(self) -> List[Dict[str, Any]]:
        return self._results_manager.ensemble_performance_history

    @property
    def trajectory(self) -> Optional[List]:
        return self._results_manager.trajectory

    def set_pipeline_config(self, **pipeline_config_kwargs: Any) -> None:
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

    def get_search_space(self, dataset: BaseDataset = None) -> ConfigurationSpace:
        """
        Returns the current search space as ConfigurationSpace object.
        """
        if self.search_space is not None:
            return self.search_space
        elif dataset is not None:
            dataset_requirements = get_dataset_requirements(
                info=dataset.get_required_dataset_info(),
                include=self.include_components,
                exclude=self.exclude_components,
                search_space_updates=self.search_space_updates)
            return get_configuration_space(info=dataset.get_dataset_properties(dataset_requirements),
                                           include=self.include_components,
                                           exclude=self.exclude_components,
                                           search_space_updates=self.search_space_updates)
        raise ValueError("No search space initialised and no dataset passed. "
                         "Can't create default search space without the dataset")

    def _get_logger(self, name: str) -> PicklableClientLogger:
        """
        Instantiates the logger used throughout the experiment

        Args:
            name (str):
                Name of the log file, usually the dataset name

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
        context = multiprocessing.get_context(self._multiprocessing_context)
        preload_modules(context)
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
            None
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

        Returns:
            SingleBest:
                Ensemble made with incumbent pipeline
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
            pynisher_context=self._multiprocessing_context,
            backend=self._backend,
            seed=self.seed,
            metric=self._metric,
            logger_port=self._logger_port,
            cost_for_crash=get_cost_of_crash(self._metric),
            abort_on_first_run_crash=False,
            initial_num_run=num_run,
            stats=stats,
            memory_limit=memory_limit,
            disable_file_output=True if len(self._disable_file_output) > 0 else False,
            all_supported_metrics=self._all_supported_metrics
        )

        status, _, _, additional_info = ta.run(num_run, cutoff=self._time_for_task)
        if status == StatusType.SUCCESS:
            self._logger.info("Finished creating dummy predictions.")
        else:
            if additional_info.get('exitcode') == -6:
                err_msg = "Dummy prediction failed with run state {},\n" \
                          "because the provided memory limits were too tight.\n" \
                          "Please increase the 'ml_memory_limit' and try again.\n" \
                          "If you still get the problem, please open an issue and\n" \
                          "paste the additional info.\n" \
                          "Additional info:\n{}.".format(str(status), dict_repr(additional_info))
                self._logger.error(err_msg)
                # Fail if dummy prediction fails.
                raise ValueError(err_msg)

            else:
                err_msg = "Dummy prediction failed with run state {} and additional info:\n{}.".format(
                    str(status), dict_repr(additional_info)
                )
                self._logger.error(err_msg)
                # Fail if dummy prediction fails.
                raise ValueError(err_msg)

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
        starttime = time.time()

        # Initialise run history for the traditional classifiers
        run_history = RunHistory()
        memory_limit = self._memory_limit
        if memory_limit is not None:
            memory_limit = int(math.ceil(memory_limit))
        available_classifiers = get_available_traditional_learners()
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
                    pynisher_context=self._multiprocessing_context,
                    backend=self._backend,
                    seed=self.seed,
                    metric=self._metric,
                    logger_port=self._logger_port,
                    cost_for_crash=get_cost_of_crash(self._metric),
                    abort_on_first_run_crash=False,
                    initial_num_run=self._backend.get_next_num_run(),
                    stats=stats,
                    memory_limit=memory_limit,
                    disable_file_output=True if len(self._disable_file_output) > 0 else False,
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
                            "Fitting {} took {} [sec] and got performance: {}.\n"
                            "additional info:\n{}".format(cls, runtime, cost, dict_repr(additional_info))
                        )
                        configuration = additional_info['pipeline_configuration']
                        origin = additional_info['configuration_origin']
                        additional_info.pop('pipeline_configuration')
                        run_history.add(config=configuration, cost=cost,
                                        time=runtime, status=status, seed=self.seed,
                                        starttime=starttime, endtime=starttime + runtime,
                                        origin=origin, additional_info=additional_info)
                    else:
                        if additional_info.get('exitcode') == -6:
                            self._logger.error(
                                "Traditional prediction for {} failed with run state {},\n"
                                "because the provided memory limits were too tight.\n"
                                "Please increase the 'ml_memory_limit' and try again.\n"
                                "If you still get the problem, please open an issue\n"
                                "and paste the additional info.\n"
                                "Additional info:\n{}".format(cls, str(status), dict_repr(additional_info))
                            )
                        else:
                            self._logger.error(
                                "Traditional prediction for {} failed with run state {}.\nAdditional info:\n{}".format(
                                    cls, str(status), dict_repr(additional_info)
                                )
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
        budget_type: str = 'epochs',
        min_budget: int = 5,
        max_budget: int = 50,
        total_walltime_limit: int = 100,
        func_eval_time_limit_secs: Optional[int] = None,
        enable_traditional_pipeline: bool = True,
        memory_limit: Optional[int] = 4096,
        smac_scenario_args: Optional[Dict[str, Any]] = None,
        get_smac_object_callback: Optional[Callable] = None,
        tae_func: Optional[Callable] = None,
        all_supported_metrics: bool = True,
        precision: int = 32,
        disable_file_output: List = [],
        load_models: bool = True,
        portfolio_selection: Optional[str] = None,
        dask_client: Optional[dask.distributed.Client] = None
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
            budget_type (str):
                Type of budget to be used when fitting the pipeline.
                It can be one of:

                + `epochs`: The training of each pipeline will be terminated after
                    a number of epochs have passed. This number of epochs is determined by the
                    budget argument of this method.
                + `runtime`: The training of each pipeline will be terminated after
                    a number of seconds have passed. This number of seconds is determined by the
                    budget argument of this method. The overall fitting time of a pipeline is
                    controlled by func_eval_time_limit_secs. 'runtime' only controls the allocated
                    time to train a pipeline, but it does not consider the overall time it takes
                    to create a pipeline (data loading and preprocessing, other i/o operations, etc.).
                    budget_type will determine the units of min_budget/max_budget. If budget_type=='epochs'
                    is used, min_budget will refer to epochs whereas if budget_type=='runtime' then
                    min_budget will refer to seconds.
            min_budget (int):
                Auto-PyTorch uses `Hyperband <https://arxiv.org/abs/1603.06560>`_ to
                trade-off resources between running many pipelines at min_budget and
                running the top performing pipelines on max_budget.
                min_budget states the minimum resource allocation a pipeline should have
                so that we can compare and quickly discard bad performing models.
                For example, if the budget_type is epochs, and min_budget=5, then we will
                run every pipeline to a minimum of 5 epochs before performance comparison.
            max_budget (int):
                Auto-PyTorch uses `Hyperband <https://arxiv.org/abs/1603.06560>`_ to
                trade-off resources between running many pipelines at min_budget and
                running the top performing pipelines on max_budget.
                max_budget states the maximum resource allocation a pipeline is going to
                be ran. For example, if the budget_type is epochs, and max_budget=50,
                then the pipeline training will be terminated after 50 epochs.
            total_walltime_limit (int: default=100):
                Time limit in seconds for the search of appropriate models.
                By increasing this value, autopytorch has a higher
                chance of finding better models.
            func_eval_time_limit_secs (Optional[int]):
                Time limit for a single call to the machine learning model.
                Model fitting will be terminated if the machine
                learning algorithm runs over the time limit. Set
                this value high enough so that typical machine
                learning algorithms can be fit on the training
                data.
                When set to None, this time will automatically be set to
                total_walltime_limit // 2 to allow enough time to fit
                at least 2 individual machine learning algorithms.
                Set to np.inf in case no time limit is desired.
            enable_traditional_pipeline (bool: default=True):
                We fit traditional machine learning algorithms
                (LightGBM, CatBoost, RandomForest, ExtraTrees, KNN, SVM)
                prior building PyTorch Neural Networks. You can disable this
                feature by turning this flag to False. All machine learning
                algorithms that are fitted during search() are considered for
                ensemble building.
            memory_limit (Optional[int]: default=4096):
                Memory limit in MB for the machine learning algorithm.
                Autopytorch will stop fitting the machine learning algorithm
                if it tries to allocate more than memory_limit MB. If None
                is provided, no memory limit is set. In case of multi-processing,
                memory_limit will be per job. This memory limit also applies to
                the ensemble creation process.
            smac_scenario_args (Optional[Dict]):
                Additional arguments inserted into the scenario of SMAC. See the
                `SMAC documentation <https://automl.github.io/SMAC3/master/options.html?highlight=scenario#scenario>`_
                for a list of available arguments.
            get_smac_object_callback (Optional[Callable]):
                Callback function to create an object of class
                `smac.optimizer.smbo.SMBO <https://automl.github.io/SMAC3/master/apidoc/smac.optimizer.smbo.html>`_.
                The function must accept the arguments scenario_dict,
                instances, num_params, runhistory, seed and ta. This is
                an advanced feature. Use only if you are familiar with
                `SMAC <https://automl.github.io/SMAC3/master/index.html>`_.
            tae_func (Optional[Callable]):
                TargetAlgorithm to be optimised. If None, `eval_function`
                available in autoPyTorch/evaluation/train_evaluator is used.
                Must be child class of AbstractEvaluator.
            all_supported_metrics (bool: default=True):
                If True, all metrics supporting current task will be calculated
                for each pipeline and results will be available via cv_results
            precision (int: default=32):
                Numeric precision used when loading ensemble data.
                Can be either '16', '32' or '64'.
            disable_file_output (Union[bool, List]):
                If True, disable model and prediction output.
                Can also be used as a list to pass more fine-grained
                information on what to save. Allowed elements in the list are:

                + `y_optimization`:
                    do not save the predictions for the optimization set,
                    which would later on be used to build an ensemble. Note that SMAC
                    optimizes a metric evaluated on the optimization set.
                + `pipeline`:
                    do not save any individual pipeline files
                + `pipelines`:
                    In case of cross validation, disables saving the joint model of the
                    pipelines fit on each fold.
                + `y_test`:
                    do not save the predictions for the test set.
            load_models (bool: default=True):
                Whether to load the models after fitting AutoPyTorch.
            portfolio_selection (Optional[str]):
                This argument controls the initial configurations that
                AutoPyTorch uses to warm start SMAC for hyperparameter
                optimization. By default, no warm-starting happens.
                The user can provide a path to a json file containing
                configurations, similar to (...herepathtogreedy...).
                Additionally, the keyword 'greedy' is supported,
                which would use the default portfolio from
                `AutoPyTorch Tabular <https://arxiv.org/abs/2006.13799>`_

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
            info=dataset.get_required_dataset_info(),
            include=self.include_components,
            exclude=self.exclude_components,
            search_space_updates=self.search_space_updates)
        self._dataset_requirements = dataset_requirements
        dataset_properties = dataset.get_dataset_properties(dataset_requirements)
        self._stopwatch.start_task(experiment_task_name)
        self.dataset_name = dataset.dataset_name
        assert self.dataset_name is not None

        if self._logger is None:
            self._logger = self._get_logger(self.dataset_name)

        # Setup the logger for the backend
        self._backend.setup_logger(port=self._logger_port)

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

        self.pipeline_options['optimize_metric'] = optimize_metric

        if all_supported_metrics:
            self._scoring_functions = get_metrics(dataset_properties=dataset_properties,
                                                  all_supported_metrics=True)
        else:
            self._scoring_functions = [self._metric]

        self.search_space = self.get_search_space(dataset)

        # Incorporate budget to pipeline config
        if budget_type not in ('epochs', 'runtime'):
            raise ValueError("Budget type must be one ('epochs', 'runtime')"
                             f" yet {budget_type} was provided")
        self.pipeline_options['budget_type'] = budget_type

        # Here the budget is set to max because the SMAC intensifier can be:
        # Hyperband: in this case the budget is determined on the fly and overwritten
        #            by the ExecuteTaFuncWithQueue
        # SimpleIntensifier (and others): in this case, we use max_budget as a target
        #                                 budget, and hece the below line is honored
        self.pipeline_options[budget_type] = max_budget

        if self.task_type is None:
            raise ValueError("Cannot interpret task type from the dataset")

        # If no dask client was provided, we create one, so that we can
        # start a ensemble process in parallel to smbo optimize
        if self.n_jobs == 1:
            self._dask_client = SingleThreadedClient()
        elif dask_client is None:
            self._create_dask_client()
        else:
            self._dask_client = dask_client
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
        if num_models < 2 and self.ensemble_size > 0:
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
                pynisher_context=self._multiprocessing_context,
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
                pipeline_config=self.pipeline_options,
                min_budget=min_budget,
                max_budget=max_budget,
                ensemble_callback=proc_ensemble,
                logger_port=self._logger_port,
                # We do not increase the num_run here, this is something
                # smac does internally
                start_num_run=self._backend.get_next_num_run(peek=True),
                search_space_updates=self.search_space_updates,
                portfolio_selection=portfolio_selection,
                pynisher_context=self._multiprocessing_context,
            )
            try:
                run_history, self._results_manager.trajectory, budget_type = \
                    _proc_smac.run_smbo(func=tae_func)
                self.run_history.update(run_history, DataOrigin.INTERNAL)
                trajectory_filename = os.path.join(
                    self._backend.get_smac_output_directory_for_run(self.seed),
                    'trajectory.json')

                assert self.trajectory is not None  # mypy check
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
            self._results_manager.ensemble_performance_history = list(proc_ensemble.history)

            if len(proc_ensemble.futures) > 0:
                # Also add ensemble runs that did not finish within smac time
                # and add them into the ensemble history
                self._logger.info("Ensemble script still running, waiting for it to finish.")
                result = proc_ensemble.futures.pop().result()
                if result:
                    ensemble_history, _, _, _ = result
                    self._results_manager.ensemble_performance_history.extend(ensemble_history)
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

    def _get_fit_dictionary(
        self,
        dataset_properties: Dict[str, BaseDatasetPropertiesType],
        dataset: BaseDataset,
        split_id: int = 0
    ) -> Dict[str, Any]:
        X_test = dataset.test_tensors[0].copy() if dataset.test_tensors is not None else None
        y_test = dataset.test_tensors[1].copy() if dataset.test_tensors is not None else None
        X: Dict[str, Any] = dict({'dataset_properties': dataset_properties,
                                  'backend': self._backend,
                                  'X_train': dataset.train_tensors[0].copy(),
                                  'y_train': dataset.train_tensors[1].copy(),
                                  'X_test': X_test,
                                  'y_test': y_test,
                                  'train_indices': dataset.splits[split_id][0],
                                  'val_indices': dataset.splits[split_id][1],
                                  'split_id': split_id,
                                  'num_run': self._backend.get_next_num_run(),
                                  })
        X.update(self.pipeline_options)
        return X

    def refit(
        self,
        dataset: BaseDataset,
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

        Refit uses the estimator pipeline_config attribute, which the user
        can interact via the get_pipeline_config()/set_pipeline_config()
        methods.

        Args:
            dataset (Dataset):
                The argument that will provide the dataset splits. It can either
                be a dictionary with the splits, or the dataset object which can
                generate the splits based on different restrictions.
            split_id (int):
                split id to fit on.
        Returns:
            self
        """

        self.dataset_name = dataset.dataset_name

        if self._logger is None:
            self._logger = self._get_logger(str(self.dataset_name))

        dataset_requirements = get_dataset_requirements(
            info=dataset.get_required_dataset_info(),
            include=self.include_components,
            exclude=self.exclude_components,
            search_space_updates=self.search_space_updates)
        dataset_properties = dataset.get_dataset_properties(dataset_requirements)
        self._backend.save_datamanager(dataset)

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
            X = self._get_fit_dictionary(
                dataset_properties=dataset_properties,
                dataset=dataset,
                split_id=split_id)
            fit_and_suppress_warnings(self._logger, model, X, y=None)

        self._clean_logger()

        return self

    def fit(self,
            dataset: BaseDataset,
            pipeline_config: Optional[Configuration] = None,
            split_id: int = 0) -> BasePipeline:
        """
        Fit a pipeline on the given task for the budget.
        A pipeline configuration can be specified if None,
        uses default

        Fit uses the estimator pipeline_config attribute, which the user
        can interact via the get_pipeline_config()/set_pipeline_config()
        methods.

        Args:
            dataset (Dataset):
                The argument that will provide the dataset splits. It can either
                be a dictionary with the splits, or the dataset object which can
                generate the splits based on different restrictions.
            split_id (int: default=0):
                split id to fit on.
            pipeline_config (Optional[Configuration]):
                configuration to fit the pipeline with. If None,
                uses default

        Returns:
            BasePipeline:
                fitted pipeline
        """
        self.dataset_name = dataset.dataset_name

        if self._logger is None:
            self._logger = self._get_logger(str(self.dataset_name))

        # get dataset properties
        dataset_requirements = get_dataset_requirements(
            info=dataset.get_required_dataset_info(),
            include=self.include_components,
            exclude=self.exclude_components,
            search_space_updates=self.search_space_updates)
        dataset_properties = dataset.get_dataset_properties(dataset_requirements)
        self._backend.save_datamanager(dataset)

        # build pipeline
        pipeline = self.build_pipeline(dataset_properties)
        if pipeline_config is not None:
            pipeline.set_hyperparameters(pipeline_config)

        # initialise fit dictionary
        X = self._get_fit_dictionary(
            dataset_properties=dataset_properties,
            dataset=dataset,
            split_id=split_id)

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
            X_test (np.ndarray):
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
            y_pred (np.ndarray):
                The test predictions
            y_test (np.ndarray):
                The test ground truth labels.

        Returns:
            Dict[str, float]:
                Value of the evaluation metric calculated on the test set.
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

        if self._metric is None:
            raise RuntimeError("`search_results` is only available after a search has finished.")

        return self._results_manager.get_incumbent_results(metric=self._metric, include_traditional=include_traditional)

    def get_models_with_weights(self) -> List:
        if self.models_ is None or len(self.models_) == 0 or \
                self.ensemble_ is None:
            self._load_models()

        assert self.ensemble_ is not None
        models_with_weights: List[Tuple[float, BasePipeline]] = self.ensemble_.get_models_with_weights(self.models_)
        return models_with_weights

    def show_models(self) -> str:
        """
        Returns a Markdown containing details about the final ensemble/configuration.

        Returns:
            str:
                Markdown table of models.
        """
        df = []
        for weight, model in self.get_models_with_weights():
            representation = model.get_pipeline_representation()
            representation.update({'Weight': weight})
            df.append(representation)
        models_markdown: str = pd.DataFrame(df).to_markdown()
        return models_markdown

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
        self._logger.debug('  multiprocessing_context: %s', str(self._multiprocessing_context))
        for key, value in vars(self).items():
            self._logger.debug(f"\t{key}->{value}")

    def get_search_results(self) -> SearchResults:
        """
        Get the interface to obtain the search results easily.
        """
        if self._scoring_functions is None or self._metric is None:
            raise RuntimeError("`search_results` is only available after a search has finished.")

        return self._results_manager.get_search_results(
            metric=self._metric,
            scoring_functions=self._scoring_functions
        )

    def sprint_statistics(self) -> str:
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

        Returns:
            (str):
                Formatted string with statistics
        """
        if self._scoring_functions is None or self._metric is None:
            raise RuntimeError("`search_results` is only available after a search has finished.")

        assert self.dataset_name is not None  # my check
        return self._results_manager.sprint_statistics(
            dataset_name=self.dataset_name,
            scoring_functions=self._scoring_functions,
            metric=self._metric
        )

    def plot_perf_over_time(
        self,
        metric_name: str,
        ax: Optional[plt.Axes] = None,
        plot_setting_params: PlotSettingParams = PlotSettingParams(),
        color_label_settings: ColorLabelSettings = ColorLabelSettings(),
        *args: Any,
        **kwargs: Any
    ) -> None:
        """
        Visualize the performance over time using matplotlib.
        The plot relate arguments are based on matplotlib.
        Please refer to the matplotlib documentation for more details.

        Args:
            metric_name (str):
                The name of metric to visualize.
                The names are available in
                    * autoPyTorch.metrics.CLASSIFICATION_METRICS
                    * autoPyTorch.metrics.REGRESSION_METRICS
            ax (Optional[plt.Axes]):
                axis to plot (subplots of matplotlib).
                If None, it will be created automatically.
            plot_setting_params (PlotSettingParams):
                Parameters for the plot.
            color_label_settings (ColorLabelSettings):
                The settings of a pair of color and label for each plot.
            args, kwargs (Any):
                Arguments for the ax.plot.
        """

        if not hasattr(metrics, metric_name):
            raise ValueError(
                f'metric_name must be in {list(metrics.CLASSIFICATION_METRICS.keys())} '
                f'or {list(metrics.REGRESSION_METRICS.keys())}, but got {metric_name}'
            )

        results = MetricResults(
            metric=getattr(metrics, metric_name),
            run_history=self.run_history,
            ensemble_performance_history=self.ensemble_performance_history
        )

        colors, labels = {}, {}

        for key, color_label in vars(color_label_settings).items():
            if color_label is None:
                continue

            prefix = '::'.join(key.split('_'))
            try:
                new_key = [key for key in results.data.keys() if key.startswith(prefix)][0]
                colors[new_key], labels[new_key] = color_label
            except IndexError:  # ensemble does not always have results
                pass

        self._visualizer.plot_perf_over_time(
            results=results, plot_setting_params=plot_setting_params,
            colors=colors, labels=labels, ax=ax,
            *args, **kwargs  # type: ignore
        )
