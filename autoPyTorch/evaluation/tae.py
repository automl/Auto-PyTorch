"""
Target algorithm execution package.

Most notations comply with the SMAC API.
You can find the information below:
https://automl.github.io/SMAC3/master/apidoc/smac.tae.execute_func.html

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
    """ The information obtained from the training """
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


class PynisherFuncWrapperType(object):
    """ Typing class for function wrapped by pynisher """
    def __init__(self, *args: List[Any], **kwargs: Dict[str, Any]):
        self.exit_status: Any = None
        self.exitcode: Optional[str] = None
        self.wall_clock_time: Optional[float] = None
        self.stdout: Optional[str] = None
        self.stderr: Optional[str] = None
        raise TypeError("Cannot instantiate `PynisherFuncWrapperType` instances.")

    def __call__(self,  *args: List[Any], **kwargs: Dict[str, Any]) -> None:
        raise NotImplementedError


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

    def _exception_processor(self, obj: PynisherFuncWrapperType, queue: multiprocessing.Queue,
                             info_msg: str, info_for_empty: AdditionalRunInfoType,
                             status: StatusType, is_anything_exception: bool
                             ) -> ExceptionReturnType:
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
            cost (float): The metric obtained from the training
            status (StatusType)
            info: (Optional[List[RunValue]]): The training results at each time step
            additional_run_info (AdditionalRunInfoType)
        """

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

    def _process_exceptions(self, obj: PynisherFuncWrapperType, queue: multiprocessing.Queue, budget: float
                            ) -> ExceptionReturnType:
        """ the conditional branch for _exception_processor() func """

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
        targets = {'learning_curve': (True, None),
                   'train_learning_curve': (True, 'train_loss'),
                   'validation_learning_curve': (self._get_validation_loss, 'validation_loss'),
                   'test_learning_curve': (self._get_test_loss, 'test_loss')}

        for key, (collect, metric) in targets.items():
            if collect:
                lc = extract_learning_curve(info, metric)
                if len(lc) > 1:
                    stored = True
                    additional_run_info[key] = lc

        if stored:
            additional_run_info['learning_curve_runtime'] = lc_runtime

        return additional_run_info

    def _get_init_params(self, instance: Optional[str]) -> Dict[str, Any]:
        """ The initialization parameters for the objective function """
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

    def _get_num_run(self, config: Union[int, str, Configuration]) -> int:
        if isinstance(config, (int, str)):
            num_run = self.initial_num_run
        else:
            num_run = config.config_id + self.initial_num_run

        self.logger.debug(f"Search space updates for {num_run}: {self.search_space_updates}")

        return num_run

    def _get_configuration_origin(self, config: Union[int, str, Configuration]) -> str:
        """ Get what the configuration is for """
        if isinstance(config, int):
            origin = 'DUMMY'
        elif isinstance(config, str):
            origin = 'traditional'
        else:
            origin = getattr(config, 'origin', 'UNKNOWN')

        return origin

    def _collect_obj_kwargs(self) -> Dict[str, Any]:
        """ collect obj_kwargs from the member variables """
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

    def _run_objective(self, obj: PynisherFuncWrapperType,
                       cutoff: Optional[float],
                       obj_kwargs: Dict[str, Any]) -> Tuple[bool, AdditionalRunInfoType]:
        """
        Run the objective function and return errors if needed

        Args:
            obj (PynisherFuncWrapperType):
                The target algorithm wrapped by pynisher.enforce_limits
            cutoff (Optional[float]):
                The cutoff of the training (cutoff seconds)
            obj_kwargs (Dict[str, Any]):
                The arguments for the objective function

        Returns:
            is_success (bool):
                if we could finish the objective function successfully.
            additional_run_info (AdditionalRunInfoType):
                The information obtained from the training.

        Note:
            By default, we run `fit_predict_try_except_decorator`
            taking `autoPyTorch.evaluation.train_evaluator.eval_function`
            as `self.ta`.
            When calling `obj`, we first instantiate
            a model/models based on given `config`
            and train and measure the performance.
        """
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

    def run(self, config: Union[int, str, Configuration],
            instance: Optional[str] = None,
            cutoff: Optional[float] = None,
            seed: int = 12345, budget: float = 0.0,
            instance_specific: Optional[str] = None,
            ) -> Tuple[StatusType, float, float, AdditionalRunInfoType]:
        """
        Args:
            cutoff (float):
                The cutoff threshold (seconds)
            config (Union[int, str, Configuration]):
                `int` for dummy,
                `str` for traditional machine learning,
                `Configuration` for AutoPytorch pipeline
            seed (int): a seed for random numbers
            budget (float): a time budget for running
            instance (Optional[str]): Problem instance
            instance_specific (Optional[str]):
                Instance specific information
                (e.g., domain file or solution)

        Returns:
            cost (float): The metric obtained from the training
            status (StatusType)
            info: (Optional[List[RunValue]]): The training results at each time step
            additional_run_info (AdditionalRunInfoType)
        """

        context = multiprocessing.get_context(self.pynisher_context)
        queue: multiprocessing.queues.Queue = context.Queue()

        if instance_specific is not None and instance_specific != '0':
            raise ValueError('Instance specific feature has not been supported yet.'
                             ' Do not set instance_specific.')

        num_run = self._get_num_run(config)

        obj_kwargs = self._collect_obj_kwargs()
        obj_kwargs.update(
            queue=queue, config=config, budget=budget,
            num_run=num_run, instance=instance,
            init_params=self._get_init_params(instance)
        )

        """For more details of `obj`, see the Note in _run_objective."""
        obj = pynisher.enforce_limits(**self._get_pynisher_kwargs(cutoff=cutoff))(self.ta)
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
