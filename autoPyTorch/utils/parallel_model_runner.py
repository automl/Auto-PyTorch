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
    all_supported_metrics: bool,
    seed: int,
    multiprocessing_context: str,
    n_jobs: int,
    current_search_space: ConfigurationSpace,
    smac_initial_run: int,
    include: Optional[Dict[str, List[str]]] = None,
    exclude: Optional[Dict[str, List[str]]] = None,
    search_space_updates: Optional[HyperparameterSearchSpaceUpdates] = None,
    logger_port: Optional[int] = logging.handlers.DEFAULT_TCP_LOGGING_PORT,
    memory_limit: Optional[int] = None,
    disable_file_output: Optional[List[Union[str, DisableFileOutputParameters]]] = None,
    pipeline_options: Optional[Dict] = None,
) -> Tuple[RunHistory, List[Optional[Tuple[int, int, float]]]]:
    """
    Runs models specified by `model_configs` on dask parallel infrastructure.

    Args:
        time_left (int): _description_
        func_eval_time_limit_secs (int): _description_
        model_configs (List[Tuple[str, Configuration]]): _description_
        logger (PicklableClientLogger): _description_
        metric (autoPyTorchMetric): _description_
        dask_client (dask.distributed.Client): _description_
        backend (Backend): _description_
        memory_limit (int): _description_
        disable_file_output (_type_): _description_
        all_supported_metrics (bool): _description_
        pipeline_options (_type_): _description_
        seed (int): _description_
        multiprocessing_context (str): _description_
        n_jobs (int): _description_
        current_search_space (ConfigurationSpace): _description_
        smac_initial_run (int): _description_
        include (Optional[Dict[str, List[str]]], optional): _description_. Defaults to None.
        exclude (Optional[Dict[str, List[str]]], optional): _description_. Defaults to None.
        search_space_updates (Optional[HyperparameterSearchSpaceUpdates], optional): _description_. Defaults to None.
        logger_port (Optional[int], optional): _description_. Defaults to logging.handlers.DEFAULT_TCP_LOGGING_PORT.

    Returns:
        Union[RunHistory, List[Tuple[int, int, float]]]: _description_
    """

    starttime = time.time()
    run_history = RunHistory()
    memory_limit = memory_limit
    if memory_limit is not None:
        memory_limit = int(math.ceil(memory_limit))
    model_identifiers: List[Optional[Tuple[int, int, float]]] = []
    total_models = len(model_configs)
    dask_futures = []
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
                init_num_run = smac_initial_run
            else:
                init_num_run = smac_initial_run + n_r

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
                cls, future = dask_futures.pop(0)
                status, cost, runtime, additional_info = future.result()

                if status == StatusType.SUCCESS:
                    logger.info(
                        "Fitting {} took {} [sec] and got performance: {}.\n"
                        "additional info:\n{}".format(cls, runtime, cost, dict_repr(additional_info))
                    )
                    origin: str = additional_info['configuration_origin']
                    current_config: Union[str, dict] = additional_info['configuration']
                    current_budget: float = additional_info['budget']

                    # indicates the finished model is part of autopytorch search space
                    if isinstance(current_config, dict):
                        configuration = Configuration(current_search_space, current_config)  # type: ignore[misc]
                    else:
                        # we assume that it is a traditional model and `pipeline_configuration`
                        # specifies the configuration.
                        configuration = additional_info.pop('pipeline_configuration')

                    run_history.add(config=configuration, cost=cost,
                                    time=runtime, status=status, seed=seed,
                                    starttime=starttime, endtime=starttime + runtime,
                                    origin=origin, additional_info=additional_info)
                    model_identifiers.append((seed, additional_info['num_run'], float(current_budget)))
                else:
                    if additional_info.get('exitcode') == -6:
                        logger.error(
                            "Traditional prediction for {} failed with run state {},\n"
                            "because the provided memory limits were too tight.\n"
                            "Please increase the 'ml_memory_limit' and try again.\n"
                            "If you still get the problem, please open an issue\n"
                            "and paste the additional info.\n"
                            "Additional info:\n{}".format(cls, str(status), dict_repr(additional_info))
                        )
                    else:
                        logger.error(
                            "Traditional prediction for {} failed with run state {}.\nAdditional info:\n{}".format(
                                cls, str(status), dict_repr(additional_info)
                            )
                        )
                    model_identifiers.append(None)
        # In the case of a serial execution, calling submit halts the run for a resource
        # dynamically adjust time in this case
        time_left -= int(time.time() - start_time)

        # Exit if no more time is available for a new classifier
        if time_left < func_eval_time_limit_secs:
            logger.warning("Not enough time to fit all machine learning models."
                           "Please consider increasing the run time to further improve performance.")
            break

    return run_history, model_identifiers
