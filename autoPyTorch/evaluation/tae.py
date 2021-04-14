"""
TODO:
    * Put documentation for some functions
    * Add tests for some functions
"""
# -*- encoding: utf-8 -*-
import functools
import json
import logging
import math
import multiprocessing
import time
import traceback
import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from ConfigSpace import Configuration

import numpy as np

import pynisher
from pynisher import TimeoutException, MemorylimitException
from pynisher.enforce_limits import function_wrapper

from smac.runhistory.runhistory import RunInfo, RunValue
from smac.stats.stats import Stats
from smac.tae import StatusType, TAEAbortException
from smac.tae.execute_func import AbstractTAFunc

import autoPyTorch.evaluation.train_evaluator
from autoPyTorch.evaluation.utils import empty_queue, extract_learning_curve, read_queue
from autoPyTorch.pipeline.components.training.metrics.base import autoPyTorchMetric
from autoPyTorch.utils.attr_dict import AttrDict
from autoPyTorch.utils.backend import Backend
from autoPyTorch.utils.hyperparameter_search_space_update import HyperparameterSearchSpaceUpdates
from autoPyTorch.utils.logging_ import PicklableClientLogger, get_named_client_logger


class AdditionalRunInfo(AttrDict):
    traceback: Optional[str]
    error: Optional[str]
    subprocess_stdout: Optional[str]
    subprocess_stderr: Optional[str]
    info: Optional[str]
    exit_status: Optional[Any]
    exit_code: Optional[str]
    learning_curve: Optional[List[List]]
    learning_curve_runtime: Optional[List[List]]
    train_learning_curve: Optional[List[List]]
    validation_learning_curve: Optional[List[List]]
    test_learning_curve: Optional[List[List]]
    configuration_origin: Optional[str]
    duration: Optional[float]
    num_run: Optional[int]
    train_loss: Optional[Dict[str, float]]
    validation_loss: Optional[Dict[str, float]]
    test_loss: Optional[Dict[str, float]]


AdditionalRunInfoType = Union[Dict[str, Any], AdditionalRunInfo]
ExceptionReturnType = Tuple[float, StatusType, Optional[List[RunValue]], AdditionalRunInfoType]


def fit_predict_try_except_decorator(ta: Callable, queue: multiprocessing.Queue,
                                     cost_for_crash: float, **kwargs: Any) -> None:
    try:
        return ta(queue=queue, **kwargs)
    except Exception as e:
        if isinstance(e, (MemoryError, TimeoutException)):
            # Re-raise the memory error to let the pynisher handle that correctly
            raise e

        exception_traceback = traceback.format_exc()
        error_message = repr(e)

        # Print also to STDOUT in case of broken handlers
        warnings.warn("Exception handling in `fit_predict_try_except_decorator`: "
                      "traceback: %s \nerror message: %s" % (exception_traceback, error_message))

        queue.put({'loss': cost_for_crash,
                   'additional_run_info': {'traceback': exception_traceback,
                                           'error': error_message},
                   'status': StatusType.CRASHED,
                   'final_queue_element': True}, block=True)
        queue.close()


def get_cost_of_crash(metric: autoPyTorchMetric) -> float:
    # The metric must always be defined to extract optimum/worst
    if not isinstance(metric, autoPyTorchMetric):
        raise ValueError("The metric must be strictly be an instance of autoPyTorchMetric")

    # Autopytorch optimizes the err. This function translates
    # worst_possible_result to be a minimization problem.
    # For metrics like accuracy that are bounded to [0,1]
    # metric.optimum==1 is the worst cost.
    # A simple guide is to use greater_is_better embedded as sign
    if metric._sign < 0:
        worst_possible_result = metric._worst_possible_result
    else:
        worst_possible_result = metric._optimum - metric._worst_possible_result

    return worst_possible_result


def _encode_exit_status(exit_status: multiprocessing.connection.Connection
                        ) -> str:
    try:
        encoded_exit_status: str = json.dumps(exit_status)
        return encoded_exit_status
    except (TypeError, OverflowError):
        return str(exit_status)


class ExecuteTaFuncWithQueue(AbstractTAFunc):
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
            pipeline_config: Optional[Dict[str, Any]] = None,
            initial_num_run: int = 1,
            stats: Optional[Stats] = None,
            run_obj: str = 'quality',
            par_factor: int = 1,
            output_y_hat_optimization: bool = True,
            include: Optional[Dict[str, Any]] = None,
            exclude: Optional[Dict[str, Any]] = None,
            memory_limit: Optional[int] = None,
            disable_file_output: bool = False,
            init_params: Dict[str, Any] = None,
            budget_type: str = None,
            ta: Optional[Callable] = None,
            logger_port: int = None,
            all_supported_metrics: bool = True,
            pynisher_context: str = 'spawn',
            search_space_updates: Optional[HyperparameterSearchSpaceUpdates] = None
    ):

        eval_function = autoPyTorch.evaluation.train_evaluator.eval_function

        self.worst_possible_result = cost_for_crash

        eval_function = functools.partial(
            fit_predict_try_except_decorator,
            ta=eval_function,
            cost_for_crash=self.worst_possible_result,
        )

        super().__init__(
            ta=ta if ta is not None else eval_function,
            stats=stats,
            run_obj=run_obj,
            par_factor=par_factor,
            cost_for_crash=self.worst_possible_result,
            abort_on_first_run_crash=abort_on_first_run_crash,
        )

        self.backend = backend
        self.pynisher_context = pynisher_context
        self.seed = seed
        self.initial_num_run = initial_num_run
        self.metric = metric
        self.output_y_hat_optimization = output_y_hat_optimization
        self.include = include
        self.exclude = exclude
        self.disable_file_output = disable_file_output
        self.init_params = init_params
        self.pipeline_config = pipeline_config
        self.budget_type = pipeline_config['budget_type'] if pipeline_config is not None else budget_type
        self.logger_port = logger_port
        if self.logger_port is None:
            self.logger: Union[logging.Logger, PicklableClientLogger] = logging.getLogger("TAE")
        else:
            self.logger = get_named_client_logger(
                name="TAE",
                port=self.logger_port,
            )
        self.all_supported_metrics = all_supported_metrics

        if memory_limit is not None:
            memory_limit = int(math.ceil(memory_limit))
        self.memory_limit = memory_limit

        dm = self.backend.load_datamanager()
        if dm.val_tensors is not None:
            self._get_validation_loss = True
        else:
            self._get_validation_loss = False
        if dm.test_tensors is not None:
            self._get_test_loss = True
        else:
            self._get_test_loss = False

        self.resampling_strategy = dm.resampling_strategy
        self.resampling_strategy_args = dm.resampling_strategy_args

        self.search_space_updates = search_space_updates

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
        if self.budget_type is None:
            if run_info.budget != 0:
                raise ValueError(
                    f'If budget_type is None, budget must be.0, but is {run_info.budget}'
                )
        else:
            if run_info.budget == 0:
                run_info = run_info._replace(budget=100.0)
            elif run_info.budget <= 0 or run_info.budget > 100:
                raise ValueError(f'Invalid value for budget, must be in (0, 100], but got {run_info.budget}')
            if self.budget_type not in ('epochs', 'runtime'):
                raise ValueError("Invalid value for budget type, must be one of "
                                 f"('epochs', 'runtime'), but got : {self.budget_type}")

        remaining_time = self.stats.get_remaing_time_budget()

        if remaining_time - 5 < run_info.cutoff:
            run_info = run_info._replace(cutoff=int(remaining_time - 5))

        if run_info.cutoff < 1.0:
            return run_info, RunValue(
                status=StatusType.STOP,
                cost=self.worst_possible_result,
                time=0.0,
                additional_info={},
                starttime=time.time(),
                endtime=time.time(),
            )
        elif (
                run_info.cutoff != int(np.ceil(run_info.cutoff))
                and not isinstance(run_info.cutoff, int)
        ):
            run_info = run_info._replace(cutoff=int(np.ceil(run_info.cutoff)))

        self.logger.info("Starting to evaluate configuration %s" % run_info.config.config_id)
        return super().run_wrapper(run_info=run_info)

    def _exception_processor(self, obj: function_wrapper, queue: multiprocessing.Queue,
                             info_msg: str, info_for_empty: AdditionalRunInfoType,
                             status: StatusType, is_anything_exception: bool
                             ) -> ExceptionReturnType:

        cost, info = self.worst_possible_result, None
        additional_run_info: AdditionalRunInfoType = {}

        if queue.empty():
            additional_run_info.update(info_for_empty)
        else:
            info = read_queue(queue)
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

    def _process_exceptions(self, obj: function_wrapper, queue: multiprocessing.Queue, budget: float
                            ) -> ExceptionReturnType:
        additional_run_info: AdditionalRunInfoType = {}

        if obj.exit_status is TimeoutException or obj.exit_status is MemorylimitException:
            is_timeout = obj.exit_status is TimeoutException
            cost, status, info, additional_run_info = self._exception_processor(
                obj=obj, queue=queue, is_anything_exception=False,
                info_msg=f'Run stopped because of {"timeout" if is_timeout else "memout"}.',
                info_for_empty={'error': 'Timeout' if is_timeout else 'Memout'},
                status=StatusType.TIMEOUT if is_timeout else StatusType.MEMOUT
            )
        elif obj.exit_status is TAEAbortException:
            info, status, cost = None, StatusType.ABORT, self.worst_possible_result
            additional_run_info.update(
                error='Your configuration of autoPyTorch does not work!',
                exit_status=_encode_exit_status(obj.exit_status),
                subprocess_stdout=obj.stdout,
                subprocess_stderr=obj.stderr
            )
        else:
            info_msg = 'Run treated as crashed because the pynisher exit ' \
                       f'status {str(obj.exit_status)} is unknown.'
            info_for_empty: AdditionalRunInfoType = {}
            info_for_empty.update(
                error='Result queue is empty',
                exit_status=_encode_exit_status(obj.exit_status),
                subprocess_stdout=obj.stdout,
                subprocess_stderr=obj.stderr,
                exitcode=obj.exitcode
            )

            cost, status, info, additional_run_info = self._exception_processor(
                obj=obj, queue=queue, is_anything_exception=True,
                info_msg=info_msg, info_for_empty=info_for_empty,
                status=StatusType.CRASHED
            )

        if ((self.budget_type is None or budget == 0) and status == StatusType.DONOTADVANCE):
            status = StatusType.SUCCESS

        if not isinstance(additional_run_info, dict):
            additional_run_info = {'message': additional_run_info}

        return cost, status, info, additional_run_info

    def _add_learning_curve_info(self, info: Optional[List[RunValue]],
                                 additional_run_info: AdditionalRunInfoType,
                                 ) -> AdditionalRunInfoType:
        lc_runtime = extract_learning_curve(info, 'duration')
        stored = False
        targets = {'learning_curve': True,
                   'train_learning_curve': True,
                   'validation_learning_curve': self._get_validation_loss,
                   'test_learning_curve': self._get_test_loss}

        for key, collect in targets.items():
            if collect:
                lc = extract_learning_curve(info, key)
                if len(lc) > 1:
                    stored = True
                    additional_run_info[key] = lc

        if stored:
            additional_run_info['learning_curve_runtime'] = lc_runtime

        return additional_run_info

    def _get_init_params(self, instance: Optional[str]) -> Dict[str, Any]:
        init_params = {'instance': instance}
        if self.init_params is not None:
            init_params.update(self.init_params)
        return init_params

    def _get_pynisher_kwargs(self, cutoff: Optional[float] = None) -> Dict[str, Any]:

        context = multiprocessing.get_context(self.pynisher_context)
        if self.logger_port is None:
            logger: Union[logging.Logger, PicklableClientLogger] = logging.getLogger("pynisher")
        else:
            logger = get_named_client_logger(name="pynisher", port=self.logger_port)

        return dict(
            logger=logger, mem_in_mb=self.memory_limit,
            capture_output=True, context=context,
            wall_time_in_s=int(cutoff) if cutoff is not None else None,
        )

    def _get_num_run(self, config: Any) -> int:
        if isinstance(config, (int, str)):
            num_run = self.initial_num_run
        else:
            num_run = config.config_id + self.initial_num_run

        self.logger.debug(f"Search space updates for {num_run}: {self.search_space_updates}")

        return num_run

    def _get_configuration_origin(self, config: Any) -> str:
        if isinstance(config, int):
            origin = 'DUMMY'
        elif isinstance(config, str):
            origin = 'traditional'
        else:
            origin = getattr(config, 'origin', 'UNKNOWN')

        return origin

    def _collect_obj_kwargs(self) -> Dict[str, Any]:
        return dict(
            backend=self.backend,
            metric=self.metric,
            seed=self.seed,
            output_y_hat_optimization=self.output_y_hat_optimization,
            include=self.include,
            exclude=self.exclude,
            disable_file_output=self.disable_file_output,
            budget_type=self.budget_type,
            pipeline_config=self.pipeline_config,
            logger_port=self.logger_port,
            all_supported_metrics=self.all_supported_metrics,
            search_space_updates=self.search_space_updates
        )

    def _run_objective(self, obj: function_wrapper,
                       cutoff: Optional[float],
                       obj_kwargs: Dict[str, Any]) -> Tuple[bool, AdditionalRunInfoType]:
        additional_run_info: AdditionalRunInfoType = {}

        try:
            obj(**obj_kwargs)
            return True, additional_run_info
        except Exception as e:
            exception_traceback = traceback.format_exc()
            error_message = repr(e)
            additional_run_info = {
                'traceback': exception_traceback,
                'error': error_message
            }
            return False, additional_run_info

    def run(self, config: Configuration,
            instance: Optional[str] = None,
            cutoff: Optional[float] = None,
            seed: int = 12345, budget: float = 0.0,
            instance_specific: Optional[str] = None,
            ) -> Tuple[StatusType, float, float, AdditionalRunInfoType]:
        """
        Args:
            cutoff (float):
                The cutoff threshold (seconds)

        Returns:

        """

        context = multiprocessing.get_context(self.pynisher_context)
        queue: multiprocessing.queues.Queue = context.Queue()

        if not (instance_specific is None or instance_specific == '0'):
            raise ValueError(instance_specific)

        num_run = self._get_num_run(config)

        obj_kwargs = self._collect_obj_kwargs()
        obj_kwargs.update(
            queue=queue, config=config, budget=budget,
            num_run=num_run, instance=instance,
            init_params=self._get_init_params(instance)
        )
        obj = pynisher.enforce_limits(**self._get_pynisher_args(cutoff=cutoff))(self.ta)

        _success, additional_run_info = self._run_objective(obj=obj, cutoff=cutoff, obj_kwargs=obj_kwargs)

        if not _success:
            return StatusType.CRASHED, self.cost_for_crash, 0.0, additional_run_info

        cost, status, info, additional_run_info = self._process_exceptions(obj=obj, queue=queue, budget=budget)

        """TODO: Check how is_iterative_fit worked in auto-sklearn"""
        if (info is not None and status != StatusType.CRASHED):
            self._add_learning_curve_info(info=info, additional_run_info=additional_run_info)

        additional_run_info['configuration_origin'] = self._get_configuration_origin(config)
        runtime = float(obj.wall_clock_time)

        empty_queue(queue)
        self.logger.debug(f'Finished function evaluation {str(num_run)}. Status: {status}, '
                          f'Cost: {cost}, Runtime: {runtime}, Additional {additional_run_info}')
        return status, cost, runtime, additional_run_info
