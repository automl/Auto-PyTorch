# -*- encoding: utf-8 -*-
import functools
import json
import logging
import math
import multiprocessing
import os
import time
import traceback
import warnings
from queue import Empty
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from ConfigSpace import Configuration

import numpy as np

import pynisher

from smac.runhistory.runhistory import RunInfo, RunValue
from smac.stats.stats import Stats
from smac.tae import StatusType, TAEAbortException
from smac.tae.execute_func import AbstractTAFunc

import autoPyTorch.evaluation.train_evaluator
from autoPyTorch.automl_common.common.utils.backend import Backend
from autoPyTorch.evaluation.utils import empty_queue, extract_learning_curve, read_queue
from autoPyTorch.pipeline.components.training.metrics.base import autoPyTorchMetric
from autoPyTorch.utils.common import dict_repr, replace_string_bool_to_bool
from autoPyTorch.utils.hyperparameter_search_space_update import HyperparameterSearchSpaceUpdates
from autoPyTorch.utils.logging_ import PicklableClientLogger, get_named_client_logger
from autoPyTorch.utils.parallel import preload_modules


def fit_predict_try_except_decorator(
        ta: Callable,
        queue: multiprocessing.Queue, cost_for_crash: float, **kwargs: Any) -> None:
    try:
        ta(queue=queue, **kwargs)
    except Exception as e:
        if isinstance(e, (MemoryError, pynisher.TimeoutException)):
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
        pynisher_context: str,
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

        self.budget_type = pipeline_config['budget_type'] if pipeline_config is not None else budget_type

        self.pipeline_config: Dict[str, Union[int, str, float]] = dict()
        if pipeline_config is None:
            pipeline_config = replace_string_bool_to_bool(json.load(open(
                os.path.join(os.path.dirname(__file__), '../configs/default_pipeline_options.json'))))
        self.pipeline_config.update(pipeline_config)

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

    def run_wrapper(
        self,
        run_info: RunInfo,
    ) -> Tuple[RunInfo, RunValue]:
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
                    'If budget_type is None, budget must be.0, but is %f' % run_info.budget
                )
        else:
            if run_info.budget == 0:
                # SMAC can return budget zero for intensifiers that don't have a concept
                # of budget, for example a simple bayesian optimization intensifier.
                # Budget determines how our pipeline trains, which can be via runtime or epochs
                epochs_budget = self.pipeline_config.get('epochs', np.inf)
                runtime_budget = self.pipeline_config.get('runtime', np.inf)
                run_info = run_info._replace(budget=min(epochs_budget, runtime_budget))
            elif run_info.budget <= 0:
                raise ValueError('Illegal value for budget, must be greater than zero but is %f' %
                                 run_info.budget)
            if self.budget_type not in ('epochs', 'runtime'):
                raise ValueError("Illegal value for budget type, must be one of "
                                 "('epochs', 'runtime'), but is : %s" %
                                 self.budget_type)

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
        run_info, run_value = super().run_wrapper(run_info=run_info)
        return run_info, run_value

    def run(
        self,
        config: Configuration,
        instance: Optional[str] = None,
        cutoff: Optional[float] = None,
        seed: int = 12345,
        budget: float = 0.0,
        instance_specific: Optional[str] = None,
    ) -> Tuple[StatusType, float, float, Dict[str, Any]]:

        context = multiprocessing.get_context(self.pynisher_context)
        preload_modules(context)
        queue: multiprocessing.queues.Queue = context.Queue()

        if not (instance_specific is None or instance_specific == '0'):
            raise ValueError(instance_specific)
        init_params = {'instance': instance}
        if self.init_params is not None:
            init_params.update(self.init_params)

        if self.logger_port is None:
            logger: Union[logging.Logger, PicklableClientLogger] = logging.getLogger("pynisher")
        else:
            logger = get_named_client_logger(
                name="pynisher",
                port=self.logger_port,
            )

        pynisher_arguments = dict(
            logger=logger,
            # Pynisher expects seconds as a time indicator
            wall_time_in_s=int(cutoff) if cutoff is not None else None,
            mem_in_mb=self.memory_limit,
            capture_output=True,
            context=context,
        )

        if isinstance(config, (int, str)):
            num_run = self.initial_num_run
        else:
            num_run = config.config_id + self.initial_num_run

        self.logger.debug("Search space updates for {}: {}".format(num_run,
                                                                   self.search_space_updates))
        obj_kwargs = dict(
            queue=queue,
            config=config,
            backend=self.backend,
            metric=self.metric,
            seed=self.seed,
            num_run=num_run,
            output_y_hat_optimization=self.output_y_hat_optimization,
            include=self.include,
            exclude=self.exclude,
            disable_file_output=self.disable_file_output,
            instance=instance,
            init_params=init_params,
            budget=budget,
            budget_type=self.budget_type,
            pipeline_config=self.pipeline_config,
            logger_port=self.logger_port,
            all_supported_metrics=self.all_supported_metrics,
            search_space_updates=self.search_space_updates
        )

        info: Optional[List[RunValue]]
        additional_run_info: Dict[str, Any]
        try:
            obj = pynisher.enforce_limits(**pynisher_arguments)(self.ta)
            obj(**obj_kwargs)
        except Exception as e:
            exception_traceback = traceback.format_exc()
            error_message = repr(e)
            additional_run_info = {
                'traceback': exception_traceback,
                'error': error_message
            }
            return StatusType.CRASHED, self.cost_for_crash, 0.0, additional_run_info

        if obj.exit_status in (pynisher.TimeoutException, pynisher.MemorylimitException):
            # Even if the pynisher thinks that a timeout or memout occured,
            # it can be that the target algorithm wrote something into the queue
            #  - then we treat it as a successful run
            try:
                info = read_queue(queue)  # type: ignore
                result = info[-1]['loss']  # type: ignore
                status = info[-1]['status']  # type: ignore
                additional_run_info = info[-1]['additional_run_info']  # type: ignore

                if obj.stdout:
                    additional_run_info['subprocess_stdout'] = obj.stdout
                if obj.stderr:
                    additional_run_info['subprocess_stderr'] = obj.stderr

                if obj.exit_status is pynisher.TimeoutException:
                    additional_run_info['info'] = 'Run stopped because of timeout.'
                elif obj.exit_status is pynisher.MemorylimitException:
                    additional_run_info['info'] = 'Run stopped because of memout.'

                if status in [StatusType.SUCCESS, StatusType.DONOTADVANCE]:
                    cost = result
                else:
                    cost = self.worst_possible_result

            except Empty:
                info = None
                if obj.exit_status is pynisher.TimeoutException:
                    status = StatusType.TIMEOUT
                    additional_run_info = {'error': 'Timeout'}
                elif obj.exit_status is pynisher.MemorylimitException:
                    status = StatusType.MEMOUT
                    additional_run_info = {
                        'error': 'Memout (used more than {} MB).'.format(self.memory_limit)
                    }
                else:
                    raise ValueError(obj.exit_status)
                cost = self.worst_possible_result

        elif obj.exit_status is TAEAbortException:
            info = None
            status = StatusType.ABORT
            cost = self.worst_possible_result
            additional_run_info = {'error': 'Your configuration of '
                                            'autoPyTorch does not work!',
                                   'exit_status': _encode_exit_status(obj.exit_status),
                                   'subprocess_stdout': obj.stdout,
                                   'subprocess_stderr': obj.stderr,
                                   }

        else:
            try:
                info = read_queue(queue)  # type: ignore
                result = info[-1]['loss']  # type: ignore
                status = info[-1]['status']  # type: ignore
                additional_run_info = info[-1]['additional_run_info']  # type: ignore

                if obj.exit_status == 0:
                    cost = result
                else:
                    status = StatusType.CRASHED
                    cost = self.worst_possible_result
                    additional_run_info['info'] = 'Run treated as crashed ' \
                                                  'because the pynisher exit ' \
                                                  'status %s is unknown.' % \
                                                  str(obj.exit_status)
                    additional_run_info['exit_status'] = _encode_exit_status(obj.exit_status)
                    additional_run_info['subprocess_stdout'] = obj.stdout
                    additional_run_info['subprocess_stderr'] = obj.stderr
            except Empty:
                info = None
                additional_run_info = {
                    'error': 'Result queue is empty',
                    'exit_status': _encode_exit_status(obj.exit_status),
                    'subprocess_stdout': obj.stdout,
                    'subprocess_stderr': obj.stderr,
                    'exitcode': obj.exitcode
                }
                status = StatusType.CRASHED
                cost = self.worst_possible_result

        if (
                (self.budget_type is None or budget == 0)
                and status == StatusType.DONOTADVANCE
        ):
            status = StatusType.SUCCESS

        if not isinstance(additional_run_info, dict):
            additional_run_info = {'message': additional_run_info}

        if (
                info is not None
                and self.resampling_strategy in ['holdout-iterative-fit', 'cv-iterative-fit']
                and status != StatusType.CRASHED
        ):
            learning_curve = extract_learning_curve(info)
            learning_curve_runtime = extract_learning_curve(info, 'duration')
            if len(learning_curve) > 1:
                additional_run_info['learning_curve'] = learning_curve
                additional_run_info['learning_curve_runtime'] = learning_curve_runtime

            train_learning_curve = extract_learning_curve(info, 'train_loss')
            if len(train_learning_curve) > 1:
                additional_run_info['train_learning_curve'] = train_learning_curve
                additional_run_info['learning_curve_runtime'] = learning_curve_runtime

            if self._get_validation_loss:
                validation_learning_curve = extract_learning_curve(info, 'validation_loss')
                if len(validation_learning_curve) > 1:
                    additional_run_info['validation_learning_curve'] = \
                        validation_learning_curve
                    additional_run_info[
                        'learning_curve_runtime'] = learning_curve_runtime

            if self._get_test_loss:
                test_learning_curve = extract_learning_curve(info, 'test_loss')
                if len(test_learning_curve) > 1:
                    additional_run_info['test_learning_curve'] = test_learning_curve
                    additional_run_info[
                        'learning_curve_runtime'] = learning_curve_runtime

        if isinstance(config, int):
            origin = 'DUMMY'
        elif isinstance(config, str):
            origin = 'traditional'
        else:
            origin = getattr(config, 'origin', 'UNKNOWN')
        additional_run_info['configuration_origin'] = origin

        runtime = float(obj.wall_clock_time)

        empty_queue(queue)
        self.logger.debug(
            "Finish function evaluation {}.\n"
            "Status: {}, Cost: {}, Runtime: {},\n"
            "Additional information:\n{}".format(
                str(num_run),
                status,
                cost,
                runtime,
                dict_repr(additional_run_info)
            )
        )
        return status, cost, runtime, additional_run_info
