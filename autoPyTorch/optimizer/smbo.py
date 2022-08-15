import copy
import json
import logging.handlers
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import ConfigSpace
from ConfigSpace.configuration_space import Configuration

import dask.distributed

from smac.facade.smac_ac_facade import SMAC4AC
from smac.intensification.hyperband import Hyperband
from smac.intensification.intensification import Intensifier
from smac.runhistory.runhistory import RunHistory
from smac.runhistory.runhistory2epm import RunHistory2EPM4LogCost
from smac.scenario.scenario import Scenario
from smac.tae.dask_runner import DaskParallelRunner
from smac.tae.serial_runner import SerialRunner
from smac.utils.io.traj_logging import TrajEntry

from autoPyTorch.automl_common.common.utils.backend import Backend
from autoPyTorch.constants import (
    FORECASTING_BUDGET_TYPE,
    STRING_TO_TASK_TYPES,
    TIMESERIES_FORECASTING
)
from autoPyTorch.datasets.base_dataset import BaseDataset
from autoPyTorch.datasets.resampling_strategy import (
    CrossValTypes,
    DEFAULT_RESAMPLING_PARAMETERS,
    HoldoutValTypes,
    NoResamplingStrategyTypes
)
from autoPyTorch.ensemble.ensemble_builder import EnsembleBuilderManager
from autoPyTorch.evaluation.tae import ExecuteTaFuncWithQueue, get_cost_of_crash
from autoPyTorch.optimizer.utils import read_forecasting_init_configurations, read_return_initial_configurations
from autoPyTorch.pipeline.components.training.metrics.base import autoPyTorchMetric
from autoPyTorch.utils.hyperparameter_search_space_update import HyperparameterSearchSpaceUpdates
from autoPyTorch.utils.logging_ import get_named_client_logger
from autoPyTorch.utils.stopwatch import StopWatch


def get_smac_object(
    scenario_dict: Dict[str, Any],
    seed: int,
    ta: Callable,
    ta_kwargs: Dict[str, Any],
    n_jobs: int,
    initial_budget: Union[int, float],
    max_budget: Union[int, float],
    dask_client: Optional[dask.distributed.Client],
    initial_configurations: Optional[List[Configuration]] = None,
) -> SMAC4AC:
    """
    This function returns an SMAC object that is gonna be used as
    optimizer of pipelines

    Args:
        scenario_dict (Dict[str, Any]): constrain on how to run
            the jobs
        seed (int): to make the job deterministic
        ta (Callable): the function to be intensifier by smac
        ta_kwargs (Dict[str, Any]): Arguments to the above ta
        n_jobs (int): Amount of cores to use for this task
        initial_budget (int): the minimal budget to be allocated to the target algorithm
        max_budget (int): the max budget to be allocated to the target algorithm
        dask_client (dask.distributed.Client): User provided scheduler
        initial_configurations (List[Configuration]): List of initial
            configurations which smac will run before starting the search process

    Returns:
        (SMAC4HPO): sequential model algorithm configuration object

    """
    if initial_budget == max_budget:
        # This allows vanilla BO optimization
        intensifier = Intensifier
        intensifier_kwargs: Dict[str, Any] = {'deterministic': True, }

    else:
        intensifier = Hyperband
        intensifier_kwargs = {'initial_budget': initial_budget, 'max_budget': max_budget,
                              'eta': 3, 'min_chall': 1, 'instance_order': 'shuffle_once'}
    rh2EPM = RunHistory2EPM4LogCost

    return SMAC4AC(
        scenario=Scenario(scenario_dict),
        rng=seed,
        runhistory2epm=rh2EPM,
        tae_runner=ta,
        tae_runner_kwargs=ta_kwargs,
        initial_configurations=initial_configurations,
        initial_design=None,
        run_id=seed,
        intensifier=intensifier,
        intensifier_kwargs=intensifier_kwargs,
        dask_client=dask_client,
        n_jobs=n_jobs,
    )


class AutoMLSMBO(object):
    def __init__(self,
                 config_space: ConfigSpace.ConfigurationSpace,
                 dataset_name: str,
                 backend: Backend,
                 total_walltime_limit: float,
                 func_eval_time_limit_secs: float,
                 memory_limit: Optional[int],
                 metric: autoPyTorchMetric,
                 watcher: StopWatch,
                 n_jobs: int,
                 dask_client: Optional[dask.distributed.Client],
                 pipeline_options: Dict[str, Any],
                 start_num_run: int = 1,
                 seed: int = 1,
                 resampling_strategy: Union[HoldoutValTypes,
                                            CrossValTypes,
                                            NoResamplingStrategyTypes] = HoldoutValTypes.holdout_validation,
                 resampling_strategy_args: Optional[Dict[str, Any]] = None,
                 include: Optional[Dict[str, Any]] = None,
                 exclude: Optional[Dict[str, Any]] = None,
                 disable_file_output: List = [],
                 smac_scenario_args: Optional[Dict[str, Any]] = None,
                 get_smac_object_callback: Optional[Callable] = None,
                 all_supported_metrics: bool = True,
                 ensemble_callback: Optional[EnsembleBuilderManager] = None,
                 logger_port: Optional[int] = None,
                 search_space_updates: Optional[HyperparameterSearchSpaceUpdates] = None,
                 portfolio_selection: Optional[str] = None,
                 pynisher_context: str = 'spawn',
                 min_budget: Union[int, float] = 5,
                 max_budget: Union[int, float] = 50,
                 task_type: str = "",
                 **kwargs: Dict[str, Any]
                 ):
        """
        Interface to SMAC. This method calls the SMAC optimize method, and allows
        to pass a callback (ensemble_callback) to make launch task at the end of each
        optimize() algorithm. The later is needed due to the nature of blocking long running
        tasks in Dask.

        Args:
            config_space (ConfigSpace.ConfigurationSpac):
                The configuration space of the whole process
            dataset_name (str):
                The name of the dataset, used to identify the current job
            backend (Backend):
                An interface with disk
            total_walltime_limit (float):
                The maximum allowed time for this job
            func_eval_time_limit_secs (float):
                How much each individual task is allowed to last
            memory_limit (Optional[int]):
                Maximum allowed CPU memory this task can use
            metric (autoPyTorchMetric):
                An scorer object to evaluate the performance of each jon
            watcher (StopWatch):
                A stopwatch object to debug time consumption
            n_jobs (int):
                How many workers are allowed in each task
            dask_client (Optional[dask.distributed.Client]):
                An user provided scheduler. Else smac will create its own.
            start_num_run (int):
                The ID index to start runs
            seed (int):
                To make the run deterministic
            resampling_strategy (str):
                What strategy to use for performance validation
            resampling_strategy_args (Optional[Dict[str, Any]]):
                Arguments to the resampling strategy -- like number of folds
            include (Optional[Dict[str, Any]] = None):
                Optimal Configuration space modifiers
            exclude (Optional[Dict[str, Any]] = None):
                Optimal Configuration space modifiers
            disable_file_output List:
                Support to disable file output to disk -- to reduce space
            smac_scenario_args (Optional[Dict[str, Any]]):
                Additional arguments to the smac scenario
            get_smac_object_callback (Optional[Callable]):
                Allows to create a user specified SMAC object
            pynisher_context (str):
                A string indicating the multiprocessing context to use
            ensemble_callback (Optional[EnsembleBuilderManager]):
                A callback used in this scenario to start ensemble building subtasks
            portfolio_selection (Optional[str]):
                This argument controls the initial configurations that
                AutoPyTorch uses to warm start SMAC for hyperparameter
                optimization. By default, no warm-starting happens.
                The user can provide a path to a json file containing
                configurations, similar to (autoPyTorch/configs/greedy_portfolio.json).
                Additionally, the keyword 'greedy' is supported,
                which would use the default portfolio from
                `AutoPyTorch Tabular <https://arxiv.org/abs/2006.13799>_`
            min_budget (int):
                Auto-PyTorch uses `Hyperband <https://arxiv.org/abs/1603.06560>_` to
                trade-off resources between running many pipelines at min_budget and
                running the top performing pipelines on max_budget.
                min_budget states the minimum resource allocation a pipeline should have
                so that we can compare and quickly discard bad performing models.
                For example, if the budget_type is epochs, and min_budget=5, then we will
                run every pipeline to a minimum of 5 epochs before performance comparison.
            max_budget (int):
                Auto-PyTorch uses `Hyperband <https://arxiv.org/abs/1603.06560>_` to
                trade-off resources between running many pipelines at min_budget and
                running the top performing pipelines on max_budget.
                max_budget states the maximum resource allocation a pipeline is going to
                be ran. For example, if the budget_type is epochs, and max_budget=50,
                then the pipeline training will be terminated after 50 epochs.
            task_type (str):
                task type. Forecasting tasks require special process
            kwargs (Any):
                additional arguments that are customed by some specific task.
                For instance, forecasting tasks require:
                    min_num_test_instances (int):  minimal number of instances used to initialize a proxy validation set
                    suggested_init_models (List[str]):  A set of initial models suggested by the users. Their
                        hyperparameters are determined by the default configurations
                    custom_init_setting_path (str): The path to the initial hyperparameter configurations set by
                    the users

        """
        super(AutoMLSMBO, self).__init__()
        # data related
        self.dataset_name = dataset_name
        self.metric = metric

        self.backend = backend
        self.all_supported_metrics = all_supported_metrics

        self.pipeline_options = pipeline_options
        # the configuration space
        self.config_space = config_space

        # the number of parallel workers/jobs
        self.n_jobs = n_jobs
        self.dask_client = dask_client

        # Evaluation
        self.resampling_strategy = resampling_strategy
        if resampling_strategy_args is None:
            resampling_strategy_args = DEFAULT_RESAMPLING_PARAMETERS[resampling_strategy]
        self.resampling_strategy_args = resampling_strategy_args

        # and a bunch of useful limits
        self.worst_possible_result = get_cost_of_crash(self.metric)
        self.total_walltime_limit = int(total_walltime_limit)
        self.func_eval_time_limit_secs = int(func_eval_time_limit_secs)
        self.memory_limit = memory_limit
        self.watcher = watcher
        self.seed = seed
        self.start_num_run = start_num_run
        self.include = include
        self.exclude = exclude
        self.disable_file_output = disable_file_output
        self.smac_scenario_args = smac_scenario_args
        self.get_smac_object_callback = get_smac_object_callback
        self.pynisher_context = pynisher_context
        self.min_budget = min_budget
        self.max_budget = max_budget

        self.ensemble_callback = ensemble_callback

        self.search_space_updates = search_space_updates

        self.task_type = task_type

        if logger_port is None:
            self.logger_port = logging.handlers.DEFAULT_TCP_LOGGING_PORT
        else:
            self.logger_port = logger_port
        logger_name = '%s(%d):%s' % (self.__class__.__name__, self.seed, ":" + self.dataset_name)
        self.logger = get_named_client_logger(name=logger_name,
                                              port=self.logger_port)
        self.logger.info("initialised {}".format(self.__class__.__name__))

        initial_configurations = []

        if STRING_TO_TASK_TYPES.get(self.task_type, -1) == TIMESERIES_FORECASTING:
            initial_configurations = self.get_init_configs_for_forecasting(config_space, kwargs)
            # proxy-validation sets
            self.min_num_test_instances: Optional[int] = kwargs.get('min_num_test_instances',  # type:ignore[assignment]
                                                                    None)
        else:
            if portfolio_selection is not None:
                initial_configurations = read_return_initial_configurations(config_space=config_space,
                                                                            portfolio_selection=portfolio_selection)

        self.initial_configurations = initial_configurations if len(initial_configurations) > 0 else None

    def run_smbo(self, func: Optional[Callable] = None
                 ) -> Tuple[RunHistory, List[TrajEntry], str]:

        self.watcher.start_task('SMBO')
        self.logger.info("Started run of SMBO")

        # == Initialize non-SMBO stuff
        # first create a scenario
        seed = self.seed
        self.config_space.seed(seed)
        # allocate a run history
        num_run = self.start_num_run

        # Initialize some SMAC dependencies

        if isinstance(self.resampling_strategy, CrossValTypes):
            num_splits = self.resampling_strategy_args['num_splits']
            instances = [[json.dumps({'task_id': self.dataset_name,
                                      'fold': fold_number})]
                         for fold_number in range(num_splits)]
        else:
            instances = [[json.dumps({'task_id': self.dataset_name})]]

        # TODO rebuild target algorithm to be it's own target algorithm
        # evaluator, which takes into account that a run can be killed prior
        # to the model being fully fitted; thus putting intermediate results
        # into a queue and querying them once the time is over
        ta_kwargs = dict(
            backend=copy.deepcopy(self.backend),
            seed=seed,
            initial_num_run=num_run,
            include=self.include if self.include is not None else dict(),
            exclude=self.exclude if self.exclude is not None else dict(),
            metric=self.metric,
            memory_limit=self.memory_limit,
            disable_file_output=self.disable_file_output,
            ta=func,
            logger_port=self.logger_port,
            all_supported_metrics=self.all_supported_metrics,
            pipeline_options=self.pipeline_options,
            search_space_updates=self.search_space_updates,
            pynisher_context=self.pynisher_context,
        )
        ta = ExecuteTaFuncWithQueue
        self.logger.info("Finish creating Target Algorithm (TA) function")

        startup_time = self.watcher.wall_elapsed(self.dataset_name)
        total_walltime_limit = self.total_walltime_limit - startup_time - 5
        scenario_dict = {
            'abort_on_first_run_crash': False,
            'cs': self.config_space,
            'cutoff_time': self.func_eval_time_limit_secs,
            'deterministic': 'true',
            'instances': instances,
            'memory_limit': self.memory_limit,
            'output-dir': self.backend.get_smac_output_directory(),
            'run_obj': 'quality',
            'wallclock_limit': total_walltime_limit,
            'cost_for_crash': self.worst_possible_result,
        }
        if self.smac_scenario_args is not None:
            for arg in [
                'abort_on_first_run_crash',
                'cs',
                'deterministic',
                'instances',
                'output-dir',
                'run_obj',
                'shared-model',
                'cost_for_crash',
            ]:
                if arg in self.smac_scenario_args:
                    self.logger.warning('Cannot override scenario argument %s, '
                                        'will ignore this.', arg)
                    del self.smac_scenario_args[arg]
            for arg in [
                'cutoff_time',
                'memory_limit',
                'wallclock_limit',
            ]:
                if arg in self.smac_scenario_args:
                    self.logger.warning(
                        'Overriding scenario argument %s: %s with value %s',
                        arg,
                        scenario_dict[arg],
                        self.smac_scenario_args[arg]
                    )
            scenario_dict.update(self.smac_scenario_args)

        budget_type = self.pipeline_options['budget_type']
        if budget_type in FORECASTING_BUDGET_TYPE:
            if STRING_TO_TASK_TYPES.get(self.task_type, -1) != TIMESERIES_FORECASTING:
                raise ValueError('Forecasting Budget type is only available for forecasting task!')
            if self.min_budget > 1. or self.max_budget > 1.:
                self.min_budget = float(self.min_budget) / float(self.max_budget)
                self.max_budget = 1.0
            ta_kwargs['min_num_test_instances'] = self.min_num_test_instances

        if self.get_smac_object_callback is not None:
            smac = self.get_smac_object_callback(scenario_dict=scenario_dict,
                                                 seed=seed,
                                                 ta=ta,
                                                 ta_kwargs=ta_kwargs,
                                                 n_jobs=self.n_jobs,
                                                 initial_budget=self.min_budget,
                                                 max_budget=self.max_budget,
                                                 dask_client=self.dask_client,
                                                 initial_configurations=self.initial_configurations)
        else:
            smac = get_smac_object(scenario_dict=scenario_dict,
                                   seed=seed,
                                   ta=ta,
                                   ta_kwargs=ta_kwargs,
                                   n_jobs=self.n_jobs,
                                   initial_budget=self.min_budget,
                                   max_budget=self.max_budget,
                                   dask_client=self.dask_client,
                                   initial_configurations=self.initial_configurations)

        if self.ensemble_callback is not None:
            smac.register_callback(self.ensemble_callback)

        self.logger.info("initialised SMBO, running SMBO.optimize()")

        smac.optimize()

        self.logger.info("finished SMBO.optimize()")

        self.runhistory = smac.solver.runhistory
        self.trajectory = smac.solver.intensifier.traj_logger.trajectory
        if isinstance(smac.solver.tae_runner, DaskParallelRunner):
            self._budget_type = smac.solver.tae_runner.single_worker.budget_type
        elif isinstance(smac.solver.tae_runner, SerialRunner):
            self._budget_type = smac.solver.tae_runner.budget_type
        else:
            raise NotImplementedError(type(smac.solver.tae_runner))

        return self.runhistory, self.trajectory, self._budget_type

    def get_init_configs_for_forecasting(self, config_space: ConfigSpace, kwargs: Dict) -> List[Configuration]:
        """get initial configurations for forecasting tasks"""
        suggested_init_models: Optional[List[str]] = kwargs.get('suggested_init_models',  # type:ignore[assignment]
                                                                None)
        custom_init_setting_path: Optional[str] = kwargs.get('custom_init_setting_path',  # type:ignore[assignment]
                                                             None)
        # if suggested_init_models is an empty list, and  custom_init_setting_path is not provided, we
        # do not provide any initial configurations
        if suggested_init_models is None or suggested_init_models or custom_init_setting_path is not None:
            datamanager: BaseDataset = self.backend.load_datamanager()
            dataset_properties = datamanager.get_dataset_properties([])
            initial_configurations = read_forecasting_init_configurations(
                config_space=config_space,
                suggested_init_models=suggested_init_models,
                custom_init_setting_path=custom_init_setting_path,
                dataset_properties=dataset_properties
            )
            return initial_configurations
        return []
