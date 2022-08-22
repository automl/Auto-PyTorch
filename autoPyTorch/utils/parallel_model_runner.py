import logging
import math
import time
import unittest
from typing import Dict, List, Optional, Tuple, Union

from ConfigSpace.configuration_space import Configuration, ConfigurationSpace

import dask.distributed

from smac.runhistory.runhistory import RunHistory
from smac.stats.stats import Stats
from smac.tae import StatusType

from autoPyTorch.automl_common.common.utils.backend import Backend
from autoPyTorch.evaluation.tae import ExecuteTaFuncWithQueue, get_cost_of_crash
from autoPyTorch.evaluation.utils import DisableFileOutputParameters
from autoPyTorch.pipeline.components.training.metrics.base import autoPyTorchMetric
from autoPyTorch.utils.common import dict_repr
from autoPyTorch.utils.hyperparameter_search_space_update import HyperparameterSearchSpaceUpdates
from autoPyTorch.utils.logging_ import PicklableClientLogger


def run_models_on_dataset(
    time_left: int,
    func_eval_time_limit_secs: int,
    model_configs: List[Tuple[str, Configuration]],
    logger: PicklableClientLogger,
    metric: autoPyTorchMetric,
    dask_client: dask.distributed.Client,
    backend: Backend,
    seed: int,
    multiprocessing_context: str,
    current_search_space: ConfigurationSpace,
    n_jobs: int = 1,
    initial_num_run: int = 1,
    all_supported_metrics: bool = False,
    include: Optional[Dict[str, List[str]]] = None,
    exclude: Optional[Dict[str, List[str]]] = None,
    search_space_updates: Optional[HyperparameterSearchSpaceUpdates] = None,
    logger_port: Optional[int] = logging.handlers.DEFAULT_TCP_LOGGING_PORT,
    memory_limit: Optional[int] = None,
    disable_file_output: Optional[List[Union[str, DisableFileOutputParameters]]] = None,
    pipeline_options: Optional[Dict] = None,
) -> RunHistory:
    """
    Runs models specified by `model_configs` on dask parallel infrastructure.

    Args:
        time_left (int):
            Time limit in seconds for the search of appropriate models.
            By increasing this value, autopytorch has a higher
            chance of finding better models.
        func_eval_time_limit_secs (int):
            Time limit for a single call to the machine learning model.
            Model fitting will be terminated if the machine
            learning algorithm runs over the time limit. Set
            this value high enough so that typical machine
            learning algorithms can be fit on the training
            data.
            Set to np.inf in case no time limit is desired.
        model_configs (List[Tuple[str, Configuration]]):
            List containing the configuration and the budget for the model to be evaluated.
        logger (PicklableClientLogger):
            Logger
        metric (autoPyTorchMetric):
            autoPyTorchMetric to be used for evaluation.
        dask_client (dask.distributed.Client):
            dask client where the function evaluation jobs are submitted.
        backend (Backend):
            Current backend object where the data is stored. The backend
            is used to interact with the disk.
        all_supported_metrics (bool):
            If True, all metrics supporting current task will be calculated
            for each pipeline.
        seed (int):
            Seed to be used for reproducibility.
        multiprocessing_context (str):
            context used for spawning child processes.
        n_jobs (int):
            Number of consecutive processes to spawn.
        current_search_space (ConfigurationSpace):
            The search space of the neural networks which will be used to instantiate Configuration objects.
        initial_num_run (int):
            Initial num run for running the models.
        include (Optional[Dict[str, List[str]]]):
            Dictionary containing components to include. Key is the node
            name and Value is an Iterable of the names of the components
            to include. Only these components will be present in the
            search space. Defaults to None.
        exclude (Optional[Dict[str, List[str]]]):
            Dictionary containing components to exclude. Key is the node
            name and Value is an Iterable of the names of the components
            to exclude. All except these components will be present in
            the search space. Defaults to None.
        search_space_updates (Optional[HyperparameterSearchSpaceUpdates]):
            Search space updates that can be used to modify the search
            space of particular components or choice modules of the pipeline.
            Defaults to None.
        logger_port (Optional[int]):
            Port used to create the logging server. Defaults to logging.handlers.DEFAULT_TCP_LOGGING_PORT.
        memory_limit (Optional[int]):
            Memory limit in MB for the machine learning algorithm.
            Autopytorch will stop fitting the machine learning algorithm
            if it tries to allocate more than memory_limit MB. If None
            is provided, no memory limit is set. In case of multi-processing,
            memory_limit will be per job. This memory limit also applies to
            the ensemble creation process. Defaults to None.
        disable_file_output (Optional[List[Union[str, DisableFileOutputParameters]]]):
                Used as a list to pass more fine-grained
                information on what to save. Must be a member of `DisableFileOutputParameters`.
                Allowed elements in the list are:

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
                + `all`:
                    do not save any of the above.
                For more information check `autoPyTorch.evaluation.utils.DisableFileOutputParameters`.
                Defaults to None.
        pipeline_options (Optional[Dict]):
            Valid config options include "device",
                "torch_num_threads", "early_stopping", "use_tensorboard_logger",
                "metrics_during_training".

    Returns:
        RunHistory:
            run_history:
                Run History of training all the models in model_configs
    """
    starttime = time.time()
    run_history = RunHistory()
    if memory_limit is not None:
        memory_limit = int(math.ceil(memory_limit))
    total_models = len(model_configs)
    dask_futures: List[dask.distributed.Future] = []
    for n_r, (config, budget) in enumerate(model_configs):

        # Only launch a task if there is time
        start_time = time.time()
        if time_left >= func_eval_time_limit_secs:
            logger.info(f"{n_r}: Started fitting {config} with cutoff={func_eval_time_limit_secs}")
            scenario_mock = unittest.mock.Mock()
            scenario_mock.wallclock_limit = time_left
            # This stats object is a hack - maybe the SMAC stats object should
            # already be generated here!
            stats = Stats(scenario_mock)
            stats.start_timing()

            if isinstance(config, Configuration):
                config.config_id = n_r
                init_num_run = initial_num_run
            else:
                init_num_run = initial_num_run + n_r

            ta = ExecuteTaFuncWithQueue(
                pynisher_context=multiprocessing_context,
                backend=backend,
                seed=seed,
                metric=metric,
                multi_objectives=["cost"],
                logger_port=logger_port,
                pipeline_options=pipeline_options,
                cost_for_crash=get_cost_of_crash(metric),
                abort_on_first_run_crash=False,
                initial_num_run=init_num_run,
                stats=stats,
                memory_limit=memory_limit,
                disable_file_output=disable_file_output,
                all_supported_metrics=all_supported_metrics,
                include=include,
                exclude=exclude,
                search_space_updates=search_space_updates
            )
            dask_futures.append([
                config,
                dask_client.submit(
                    ta.run, config=config,
                    cutoff=func_eval_time_limit_secs,
                    budget=budget
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
        if len(dask_futures) >= n_jobs:

            # How many workers to wait before starting fitting the next iteration
            workers_to_wait = 1
            if n_r >= total_models - 1 or time_left <= func_eval_time_limit_secs:
                # If on the last iteration, flush out all tasks
                workers_to_wait = len(dask_futures)

            while workers_to_wait >= 1:
                workers_to_wait -= 1
                # We launch dask jobs only when there are resources available.
                # This allow us to control time allocation properly, and early terminate
                # the traditional machine learning pipeline
                _process_result(
                    current_search_space=current_search_space,
                    dask_futures=dask_futures,
                    run_history=run_history,
                    seed=seed,
                    starttime=starttime,
                    logger=logger)

        # In the case of a serial execution, calling submit halts the run for a resource
        # dynamically adjust time in this case
        time_left -= int(time.time() - start_time)

        # Exit if no more time is available for a new classifier
        if time_left < func_eval_time_limit_secs:
            logger.warning("Not enough time to fit all machine learning models."
                           "Please consider increasing the run time to further improve performance.")
            break

    return run_history


def _process_result(
    dask_futures: List[dask.distributed.Future],
    current_search_space: ConfigurationSpace,
    run_history: RunHistory,
    seed: int,
    starttime: float,
    logger: PicklableClientLogger
) -> None:
    """
    Update run_history in-place using results of the
    latest finishing model.

    Args:
        dask_futures (List[dask.distributed.Future]):
            List of dask futures which are used to get the results of a finished run.
        run_history (RunHistory):
            RunHistory object to be appended with the finished run
        seed (int):
            Seed used for reproducibility.
        starttime (float):
            starttime of the runs.
        logger (PicklableClientLogger):
            Logger.
    """
    cls, future = dask_futures.pop(0)
    status, cost, runtime, additional_info = future.result()
    if status == StatusType.SUCCESS:
        logger.info(
            "Fitting {} took {} [sec] and got performance: {}.\n"
            "additional info:\n{}".format(cls, runtime, cost, dict_repr(additional_info))
        )
        origin: str = additional_info['configuration_origin']
        current_config: Union[str, dict] = additional_info['configuration']

        # indicates the finished model is part of autopytorch search space
        if isinstance(current_config, dict):
            configuration = Configuration(current_search_space, current_config)  # type: ignore[misc]
        else:
            # we assume that it is a traditional model and `pipeline_configuration`
            # specifies the configuration.
            configuration = additional_info.pop('pipeline_configuration', None)

        if configuration is not None:
            run_history.add(config=configuration, cost=cost,
                            time=runtime, status=status, seed=seed,
                            starttime=starttime, endtime=starttime + runtime,
                            origin=origin, additional_info=additional_info)
        else:
            logger.warning(f"Something went wrong while processing the results of {current_config}."
                           f"with additional_info: {additional_info} and status_type: {status}. "
                           f"Refer to the log file for more information.\nSkipping for now.")
    else:
        if additional_info.get('exitcode') == -6:
            logger.error(
                "Prediction for {} failed with run state {},\n"
                "because the provided memory limits were too tight.\n"
                "Please increase the 'ml_memory_limit' and try again.\n"
                "If you still get the problem, please open an issue\n"
                "and paste the additional info.\n"
                "Additional info:\n{}".format(cls, str(status), dict_repr(additional_info))
            )
        else:
            logger.error(
                "Prediction for {} failed with run state {}.\nAdditional info:\n{}".format(
                    cls, str(status), dict_repr(additional_info)
                )
            )
