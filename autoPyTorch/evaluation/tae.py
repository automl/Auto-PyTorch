# -*- encoding: utf-8 -*-
import functools
import json
import logging
import math
import multiprocessing
import time
import traceback
import warnings
from multiprocessing.context import BaseContext
from multiprocessing.queues import Queue
from queue import Empty
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from ConfigSpace import Configuration

import numpy as np

import pynisher

from smac.runhistory.runhistory import RunInfo, RunValue
from smac.stats.stats import Stats
from smac.tae import StatusType, TAEAbortException
from smac.tae.execute_func import AbstractTAFunc

from autoPyTorch.automl_common.common.utils.backend import Backend
from autoPyTorch.evaluation.abstract_evaluator import EvaluatorParams, FixedPipelineParams
from autoPyTorch.evaluation.evaluator import eval_fn
from autoPyTorch.evaluation.utils import (
    DisableFileOutputParameters,
    empty_queue,
    extract_learning_curve,
    read_queue
)
from autoPyTorch.pipeline.components.training.metrics.base import autoPyTorchMetric
from autoPyTorch.utils.common import dict_repr
from autoPyTorch.utils.hyperparameter_search_space_update import HyperparameterSearchSpaceUpdates
from autoPyTorch.utils.logging_ import PicklableClientLogger, get_named_client_logger
from autoPyTorch.utils.parallel import preload_modules


# cost, status, info, additional_run_info
ProcessedResultsType = Tuple[float, StatusType, Optional[List[RunValue]], Dict[str, Any]]
# status, cost, runtime, additional_info
PynisherResultsType = Tuple[StatusType, float, float, Dict[str, Any]]


class PynisherFunctionWrapperLikeType:
    def __init__(self, func: Callable):
        self.func: Callable = func
        self.exit_status: Any = None
        self.exitcode: Optional[str] = None
        self.wall_clock_time: Optional[float] = None
        self.stdout: Optional[str] = None
        self.stderr: Optional[str] = None
        raise RuntimeError("Cannot instantiate `PynisherFuncWrapperType` instances.")

    def __call__(self, *args: Any, **kwargs: Any) -> PynisherResultsType:
        # status, cost, runtime, additional_info
        raise NotImplementedError


# Since PynisherFunctionWrapperLikeType is not the exact type, we added Any...
PynisherFunctionWrapperType = Union[Any, PynisherFunctionWrapperLikeType]


def run_target_algorithm_with_exception_handling(
    ta: Callable,
    queue: Queue,
    cost_for_crash: float,
    **kwargs: Any
) -> None:
    try:
        ta(queue=queue, **kwargs)
    except Exception as e:
        if isinstance(e, (MemoryError, pynisher.TimeoutException)):
            # Re-raise the memory error to let the pynisher handle that correctly
            raise e

        exception_traceback = traceback.format_exc()
        error_message = repr(e)

        # Print also to STDOUT in case of broken handlers
        warnings.warn("Exception handling in `run_target_algorithm_with_exception_handling`: "
                      f"traceback: {exception_traceback} \nerror message: {error_message}")

        queue.put({'loss': cost_for_crash,
                   'additional_run_info': {'traceback': exception_traceback,
                                           'error': error_message},
                   'status': StatusType.CRASHED,
                   'final_queue_element': True}, block=True)
        queue.close()


def _get_eval_fn(cost_for_crash: float, target_algorithm: Optional[Callable] = None) -> Callable:
    if target_algorithm is not None:
        return target_algorithm
    else:
        return functools.partial(
            run_target_algorithm_with_exception_handling,
            ta=eval_fn,
            cost_for_crash=cost_for_crash,
        )


def _encode_exit_status(exit_status: multiprocessing.connection.Connection) -> str:
    try:
        encoded_exit_status: str = json.dumps(exit_status)
        return encoded_exit_status
    except (TypeError, OverflowError):
        return str(exit_status)


def _get_logger(logger_port: Optional[int], logger_name: str) -> Union[logging.Logger, PicklableClientLogger]:
    if logger_port is None:
        logger: Union[logging.Logger, PicklableClientLogger] = logging.getLogger(logger_name)
    else:
        logger = get_named_client_logger(name=logger_name, port=logger_port)

    return logger


def _get_origin(config: Union[int, str, Configuration]) -> str:
    if isinstance(config, int):
        origin = 'DUMMY'
    elif isinstance(config, str):
        origin = 'traditional'
    else:
        origin = getattr(config, 'origin', 'UNKNOWN')

    return origin


def _exception_handling(
    obj: PynisherFunctionWrapperType,
    queue: Queue,
    info_msg: str,
    info_for_empty: Dict[str, Any],
    status: StatusType,
    is_anything_exception: bool,
    worst_possible_result: float
) -> ProcessedResultsType:
    """
    Args:
        obj (PynisherFuncWrapperType):
        queue (multiprocessing.Queue): The run histories
        info_msg (str):
            a message for the `info` key in additional_run_info
        info_for_empty (AdditionalRunInfo):
            the additional_run_info in the case of empty queue
        status (StatusType): status type of the running
        is_anything_exception (bool):
            Exception other than TimeoutException or MemorylimitException

    Returns:
        result (ProcessedResultsType):
            cost, status, info, additional_run_info.
    """
    cost, info = worst_possible_result, None
    additional_run_info: Dict[str, Any] = {}

    try:
        info = read_queue(queue)
    except Empty:  # alternative of queue.empty(), which is not reliable
        return cost, status, info, info_for_empty

    result, status = info[-1]['loss'], info[-1]['status']
    additional_run_info = info[-1]['additional_run_info']

    _success_in_anything_exc = (is_anything_exception and obj.exit_status == 0)
    _success_in_to_or_mle = (status in [StatusType.SUCCESS, StatusType.DONOTADVANCE]
                             and not is_anything_exception)

    if _success_in_anything_exc or _success_in_to_or_mle:
        cost = result
    if not is_anything_exception or not _success_in_anything_exc:
        additional_run_info.update(
            subprocess_stdout=obj.stdout,
            subprocess_stderr=obj.stderr,
            info=info_msg)
    if is_anything_exception and not _success_in_anything_exc:
        status = StatusType.CRASHED
        additional_run_info.update(exit_status=_encode_exit_status(obj.exit_status))

    return cost, status, info, additional_run_info


def _process_exceptions(
    obj: PynisherFunctionWrapperType,
    queue: Queue,
    budget: float,
    worst_possible_result: float
) -> ProcessedResultsType:
    if obj.exit_status is TAEAbortException:
        info, status, cost = None, StatusType.ABORT, worst_possible_result
        additional_run_info = dict(
            error='Your configuration of autoPyTorch did not work',
            exit_status=_encode_exit_status(obj.exit_status),
            subprocess_stdout=obj.stdout,
            subprocess_stderr=obj.stderr
        )
        return cost, status, info, additional_run_info

    info_for_empty: Dict[str, Any] = {}
    if obj.exit_status in (pynisher.TimeoutException, pynisher.MemorylimitException):
        is_timeout = obj.exit_status is pynisher.TimeoutException
        status = StatusType.TIMEOUT if is_timeout else StatusType.MEMOUT
        is_anything_exception = False
        info_msg = f'Run stopped because of {"timeout" if is_timeout else "memout"}.'
        info_for_empty = {'error': 'Timeout' if is_timeout else 'Memout'}
    else:
        status, is_anything_exception = StatusType.CRASHED, True
        info_msg = 'Run treated as crashed because the pynisher exit ' \
                   f'status {str(obj.exit_status)} is unknown.'
        info_for_empty = dict(
            error='Result queue is empty',
            exit_status=_encode_exit_status(obj.exit_status),
            subprocess_stdout=obj.stdout,
            subprocess_stderr=obj.stderr,
            exitcode=obj.exitcode
        )

    cost, status, info, additional_run_info = _exception_handling(
        obj=obj, queue=queue, is_anything_exception=is_anything_exception,
        info_msg=info_msg, info_for_empty=info_for_empty,
        status=status, worst_possible_result=worst_possible_result
    )

    if budget == 0 and status == StatusType.DONOTADVANCE:
        status = StatusType.SUCCESS

    if not isinstance(additional_run_info, dict):
        additional_run_info = {'message': additional_run_info}

    return cost, status, info, additional_run_info


class TargetAlgorithmQuery(AbstractTAFunc):
    """
    Wrapper class that executes the target algorithm with
    queues according to what SMAC expects. This allows us to
    run our target algorithm with different configurations
    in parallel
    """

    def __init__(
        self,
        backend: Backend,
        seed: int,
        metric: autoPyTorchMetric,
        cost_for_crash: float,
        abort_on_first_run_crash: bool,
        pynisher_context: str,
        multi_objectives: List[str],
        pipeline_config: Optional[Dict[str, Any]] = None,
        initial_num_run: int = 1,
        stats: Optional[Stats] = None,
        run_obj: str = 'quality',
        par_factor: int = 1,
        save_y_opt: bool = True,
        include: Optional[Dict[str, Any]] = None,
        exclude: Optional[Dict[str, Any]] = None,
        memory_limit: Optional[int] = None,
        disable_file_output: Optional[List[Union[str, DisableFileOutputParameters]]] = None,
        init_params: Dict[str, Any] = None,
        ta: Optional[Callable] = None,
        logger_port: Optional[int] = None,
        all_supported_metrics: bool = True,
        search_space_updates: Optional[HyperparameterSearchSpaceUpdates] = None
    ):
        dm = backend.load_datamanager()
        self._exist_val_tensor = (dm.val_tensors is not None)
        self._exist_test_tensor = (dm.test_tensors is not None)

        self.worst_possible_result = cost_for_crash

        super().__init__(
            ta=_get_eval_fn(self.worst_possible_result, target_algorithm=ta),
            stats=stats,
            run_obj=run_obj,
            par_factor=par_factor,
            cost_for_crash=self.worst_possible_result,
            abort_on_first_run_crash=abort_on_first_run_crash,
        )

        # TODO: Modify so that we receive fixed_params from outside
        self.fixed_pipeline_params = FixedPipelineParams.with_default_pipeline_config(
            pipeline_config=pipeline_config,
            backend=backend,
            seed=seed,
            metric=metric,
            save_y_opt=save_y_opt,
            include=include,
            exclude=exclude,
            disable_file_output=disable_file_output,
            logger_port=logger_port,
            all_supported_metrics=all_supported_metrics,
            search_space_updates=search_space_updates,
        )
        self.pynisher_context = pynisher_context
        self.initial_num_run = initial_num_run
        self.init_params = init_params
        self.logger = _get_logger(logger_port, 'TAE')
        self.memory_limit = int(math.ceil(memory_limit)) if memory_limit is not None else memory_limit

    @property
    def eval_fn(self) -> Callable:
        # this is a target algorithm defined in AbstractTAFunc during super().__init__(ta)
        return self.ta  # type: ignore

    @property
    def budget_type(self) -> str:
        # budget is defined by epochs by default
        return self.fixed_pipeline_params.budget_type

    def _check_and_get_default_budget(self) -> float:
        budget_type_choices = ('epochs', 'runtime')
        pipeline_config = self.fixed_pipeline_params.pipeline_config
        budget_choices = {
            budget_type: float(pipeline_config.get(budget_type, np.inf))
            for budget_type in budget_type_choices
        }

        if self.budget_type not in budget_type_choices:
            raise ValueError(f"budget type must be in {budget_type_choices}, but got {self.budget_type}")
        else:
            return budget_choices[self.budget_type]

    def run_wrapper(self, run_info: RunInfo) -> Tuple[RunInfo, RunValue]:
        """
        wrapper function for ExecuteTARun.run_wrapper() to cap the target algorithm
        runtime if it would run over the total allowed runtime.

        Args:
            run_info (RunInfo): Object that contains enough information
                to execute a configuration run in isolation.
        Returns:
            RunInfo:
                an object containing the configuration launched
            RunValue:
                Contains information about the status/performance of config
        """
        # SMAC returns non-zero budget for intensification
        # In other words, SMAC returns budget=0 for a simple intensifier (i.e. no intensification)
        is_intensified = (run_info.budget != 0)
        default_budget = self._check_and_get_default_budget()

        if run_info.budget < 0:
            raise ValueError(f'budget must be greater than zero but got {run_info.budget}')

        if not is_intensified:
            # The budget will be provided in train evaluator when budget_type is None
            run_info = run_info._replace(budget=default_budget)

        remaining_time = self.stats.get_remaing_time_budget()

        if remaining_time - 5 < run_info.cutoff:
            run_info = run_info._replace(cutoff=int(remaining_time - 5))

        cutoff = run_info.cutoff
        if cutoff < 1.0:
            return run_info, RunValue(
                status=StatusType.STOP,
                cost=self.worst_possible_result,
                time=0.0,
                additional_info={},
                starttime=time.time(),
                endtime=time.time(),
            )
        elif cutoff != int(np.ceil(cutoff)) and not isinstance(cutoff, int):
            run_info = run_info._replace(cutoff=int(np.ceil(cutoff)))

        self.logger.info(f"Starting to evaluate configuration {run_info.config.config_id}")
        run_info, run_value = super().run_wrapper(run_info=run_info)

        if not is_intensified:  # It is required for the SMAC compatibility
            run_info = run_info._replace(budget=0.0)

        return run_info, run_value

    def _get_pynisher_func_wrapper_and_params(
        self,
        config: Configuration,
        context: BaseContext,
        num_run: int,
        instance: Optional[str] = None,
        cutoff: Optional[float] = None,
        budget: float = 0.0,
        instance_specific: Optional[str] = None,
    ) -> Tuple[PynisherFunctionWrapperType, EvaluatorParams]:

        preload_modules(context)
        if not (instance_specific is None or instance_specific == '0'):
            raise ValueError(instance_specific)

        init_params = {'instance': instance}
        if self.init_params is not None:
            init_params.update(self.init_params)

        pynisher_arguments = dict(
            logger=_get_logger(self.fixed_pipeline_params.logger_port, 'pynisher'),
            # Pynisher expects seconds as a time indicator
            wall_time_in_s=int(cutoff) if cutoff is not None else None,
            mem_in_mb=self.memory_limit,
            capture_output=True,
            context=context,
        )

        search_space_updates = self.fixed_pipeline_params.search_space_updates
        self.logger.debug(f"Search space updates for {num_run}: {search_space_updates}")

        evaluator_params = EvaluatorParams(
            configuration=config,
            num_run=num_run,
            init_params=init_params,
            budget=budget
        )

        return pynisher.enforce_limits(**pynisher_arguments)(self.eval_fn), evaluator_params

    def run(
        self,
        config: Configuration,
        instance: Optional[str] = None,
        cutoff: Optional[float] = None,
        budget: float = 0.0,
        seed: int = 12345,  # required for the compatibility with smac
        instance_specific: Optional[str] = None,
    ) -> PynisherResultsType:

        context = multiprocessing.get_context(self.pynisher_context)
        queue: multiprocessing.queues.Queue = context.Queue()
        budget_type = self.fixed_pipeline_params.budget_type
        budget = self.fixed_pipeline_params.pipeline_config[budget_type] if budget == 0 else budget
        num_run = self.initial_num_run if isinstance(config, (int, str)) else config.config_id + self.initial_num_run

        obj, params = self._get_pynisher_func_wrapper_and_params(
            config=config,
            context=context,
            num_run=num_run,
            instance=instance,
            cutoff=cutoff,
            budget=budget,
            instance_specific=instance_specific
        )

        try:
            obj(queue=queue, evaluator_params=params, fixed_pipeline_params=self.fixed_pipeline_params)
        except Exception as e:
            exception_traceback = traceback.format_exc()
            error_message = repr(e)
            additional_run_info = {
                'traceback': exception_traceback,
                'error': error_message
            }
            return StatusType.CRASHED, self.cost_for_crash, 0.0, additional_run_info

        return self._process_results(obj, config, queue, num_run, budget)

    def _add_learning_curve_info(self, additional_run_info: Dict[str, Any], info: List[RunValue]) -> None:
        """ This method is experimental (The source of information in RunValue might require modifications.) """
        lc_runtime = extract_learning_curve(info, 'duration')
        stored = False
        targets = {'learning_curve': (True, None),
                   'train_learning_curve': (True, 'train_loss'),
                   'validation_learning_curve': (self._exist_val_tensor, 'validation_loss'),
                   'test_learning_curve': (self._exist_test_tensor, 'test_loss')}

        for key, (collect, metric_name) in targets.items():
            if collect:
                lc = extract_learning_curve(info, metric_name)
                if len(lc) >= 1:
                    stored = True
                    additional_run_info[key] = lc

        if stored:
            additional_run_info['learning_curve_runtime'] = lc_runtime

    def _process_results(
        self,
        obj: PynisherFunctionWrapperType,
        config: Configuration,
        queue: Queue,
        num_run: int,
        budget: float
    ) -> PynisherResultsType:

        cost, status, info, additional_run_info = _process_exceptions(obj, queue, budget, self.worst_possible_result)

        if info is not None and status != StatusType.CRASHED:
            self._add_learning_curve_info(additional_run_info, info)

        additional_run_info['configuration_origin'] = _get_origin(config)
        assert obj.wall_clock_time is not None  # mypy check
        runtime = float(obj.wall_clock_time)

        empty_queue(queue)
        self.logger.debug(
            f"Finish function evaluation {num_run}.\n"
            f"Status: {status}, Cost: {cost}, Runtime: {runtime},\n"
            f"Additional information:\n{dict_repr(additional_run_info)}"
        )
        return status, cost, runtime, additional_run_info
