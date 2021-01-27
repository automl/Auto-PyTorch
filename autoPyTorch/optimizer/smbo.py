import copy
import json
import logging.handlers
import typing

import ConfigSpace

import dask.distributed

from smac.facade.smac_ac_facade import SMAC4AC
from smac.intensification.hyperband import Hyperband
from smac.runhistory.runhistory import RunHistory
from smac.runhistory.runhistory2epm import RunHistory2EPM4LogCost
from smac.scenario.scenario import Scenario
from smac.tae.dask_runner import DaskParallelRunner
from smac.tae.serial_runner import SerialRunner
from smac.utils.io.traj_logging import TrajEntry

from autoPyTorch.ensemble.ensemble_builder import EnsembleBuilderManager
from autoPyTorch.datasets.base_dataset import BaseDataset
from autoPyTorch.datasets.resampling_strategy import (
    CrossValTypes,
    DEFAULT_RESAMPLING_PARAMETERS,
    HoldoutValTypes,
)
from autoPyTorch.evaluation.tae import ExecuteTaFuncWithQueue, get_cost_of_crash
from autoPyTorch.pipeline.components.training.metrics.base import autoPyTorchMetric
from autoPyTorch.utils.backend import Backend
from autoPyTorch.utils.hyperparameter_search_space_update import HyperparameterSearchSpaceUpdates
from autoPyTorch.utils.logging_ import get_named_client_logger
from autoPyTorch.utils.stopwatch import StopWatch


def get_smac_object(
    scenario_dict: typing.Dict[str, typing.Any],
    seed: int,
    ta: typing.Callable,
    ta_kwargs: typing.Dict[str, typing.Any],
    n_jobs: int,
    initial_budget: int,
    max_budget: int,
    dask_client: typing.Optional[dask.distributed.Client],
) -> SMAC4AC:
    """
    This function returns an SMAC object that is gonna be used as
    optimizer of pipelines

    Args:
        scenario_dict (typing.Dict[str, typing.Any]): constrain on how to run
            the jobs
        seed (int): to make the job deterministic
        ta (typing.Callable): the function to be intensifier by smac
        ta_kwargs (typing.Dict[str, typing.Any]): Arguments to the above ta
        n_jobs (int): Amount of cores to use for this task
        dask_client (dask.distributed.Client): User provided scheduler

    Returns:
        (SMAC4AC): sequential model algorithm configuration object

    """
    intensifier = Hyperband

    rh2EPM = RunHistory2EPM4LogCost
    return SMAC4AC(
        scenario=Scenario(scenario_dict),
        rng=seed,
        runhistory2epm=rh2EPM,
        tae_runner=ta,
        tae_runner_kwargs=ta_kwargs,
        initial_configurations=None,
        run_id=seed,
        intensifier=intensifier,
        intensifier_kwargs={'initial_budget': initial_budget, 'max_budget': max_budget,
                            'eta': 3, 'min_chall': 1, 'instance_order': 'shuffle_once'},
        dask_client=dask_client,
        n_jobs=n_jobs,
    )


class AutoMLSMBO(object):

    def __init__(self,
                 config_space: ConfigSpace.ConfigurationSpace,
                 dataset_name: str,
                 backend: Backend,
                 total_walltime_limit: float,
                 func_eval_time_limit: float,
                 memory_limit: typing.Optional[int],
                 metric: autoPyTorchMetric,
                 watcher: StopWatch,
                 n_jobs: int,
                 dask_client: typing.Optional[dask.distributed.Client],
                 pipeline_config: typing.Dict[str, typing.Any],
                 start_num_run: int = 1,
                 seed: int = 1,
                 resampling_strategy: typing.Union[HoldoutValTypes, CrossValTypes] = HoldoutValTypes.holdout_validation,
                 resampling_strategy_args: typing.Optional[typing.Dict[str, typing.Any]] = None,
                 include: typing.Optional[typing.Dict[str, typing.Any]] = None,
                 exclude: typing.Optional[typing.Dict[str, typing.Any]] = None,
                 disable_file_output: typing.List = [],
                 smac_scenario_args: typing.Optional[typing.Dict[str, typing.Any]] = None,
                 get_smac_object_callback: typing.Optional[typing.Callable] = None,
                 all_supported_metrics: bool = True,
                 ensemble_callback: typing.Optional[EnsembleBuilderManager] = None,
                 logger_port: typing.Optional[int] = None,
                 search_space_updates: typing.Optional[HyperparameterSearchSpaceUpdates] = None
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
            func_eval_time_limit (float):
                How much each individual task is allowed to last
            memory_limit (typing.Optional[int]):
                Maximum allowed CPU memory this task can use
            metric (autoPyTorchMetric):
                An scorer object to evaluate the performance of each jon
            watcher (StopWatch):
                A stopwatch object to debug time consumption
            n_jobs (int):
                How many workers are allowed in each task
            dask_client (typing.Optional[dask.distributed.Client]):
                An user provided scheduler. Else smac will create its own.
            start_num_run (int):
                The ID index to start runs
            seed (int):
                To make the run deterministic
            resampling_strategy (str):
                What strategy to use for performance validation
            resampling_strategy_args (typing.Optional[typing.Dict[str, typing.Any]]):
                Arguments to the resampling strategy -- like number of folds
            include (typing.Optional[typing.Dict[str, typing.Any]] = None):
                Optimal Configuration space modifiers
            exclude (typing.Optional[typing.Dict[str, typing.Any]] = None):
                Optimal Configuration space modifiers
            disable_file_output List:
                Support to disable file output to disk -- to reduce space
            smac_scenario_args (typing.Optional[typing.Dict[str, typing.Any]]):
                Additional arguments to the smac scenario
            get_smac_object_callback (typing.Optional[typing.Callable]):
                Allows to create a user specified SMAC object
            ensemble_callback (typing.Optional[EnsembleBuilderManager]):
                A callback used in this scenario to start ensemble building subtasks

        """
        super(AutoMLSMBO, self).__init__()
        # data related
        self.dataset_name = dataset_name
        self.datamanager: typing.Optional[BaseDataset] = None
        self.metric = metric
        self.task: typing.Optional[str] = None
        self.backend = backend
        self.all_supported_metrics = all_supported_metrics

        self.pipeline_config = pipeline_config
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
        self.func_eval_time_limit = int(func_eval_time_limit)
        self.memory_limit = memory_limit
        self.watcher = watcher
        self.seed = seed
        self.start_num_run = start_num_run
        self.include = include
        self.exclude = exclude
        self.disable_file_output = disable_file_output
        self.smac_scenario_args = smac_scenario_args
        self.get_smac_object_callback = get_smac_object_callback

        self.ensemble_callback = ensemble_callback

        self.search_space_updates = search_space_updates

        dataset_name_ = "" if dataset_name is None else dataset_name
        if logger_port is None:
            self.logger_port = logging.handlers.DEFAULT_TCP_LOGGING_PORT
        else:
            self.logger_port = logger_port
        logger_name = '%s(%d):%s' % (self.__class__.__name__, self.seed, ":" + dataset_name_)
        self.logger = get_named_client_logger(name=logger_name,
                                              port=self.logger_port)
        self.logger.info("initialised {}".format(self.__class__.__name__))

    def reset_data_manager(self) -> None:
        if self.datamanager is not None:
            del self.datamanager
        self.datamanager = self.backend.load_datamanager()

        if self.datamanager is not None and self.datamanager.task_type is not None:
            self.task = self.datamanager.task_type

    def run_smbo(self, func: typing.Optional[typing.Callable] = None
                 ) -> typing.Tuple[RunHistory, typing.List[TrajEntry], str]:

        self.watcher.start_task('SMBO')
        self.logger.info("Started run of SMBO")
        # == first things first: load the datamanager
        self.reset_data_manager()

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
            logger=self.logger,
            include=self.include if self.include is not None else dict(),
            exclude=self.exclude if self.exclude is not None else dict(),
            metric=self.metric,
            memory_limit=self.memory_limit,
            disable_file_output=self.disable_file_output,
            ta=func,
            logger_port=self.logger_port,
            all_supported_metrics=self.all_supported_metrics,
            pipeline_config=self.pipeline_config,
            search_space_updates=self.search_space_updates
        )
        ta = ExecuteTaFuncWithQueue
        self.logger.info("Created TA")

        startup_time = self.watcher.wall_elapsed(self.dataset_name)
        total_walltime_limit = self.total_walltime_limit - startup_time - 5
        scenario_dict = {
            'abort_on_first_run_crash': False,
            'cs': self.config_space,
            'cutoff_time': self.func_eval_time_limit,
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

        initial_budget = self.pipeline_config['min_epochs']
        max_budget = self.pipeline_config['epochs']

        if self.get_smac_object_callback is not None:
            smac = self.get_smac_object_callback(scenario_dict=scenario_dict,
                                                 seed=seed,
                                                 ta=ta,
                                                 ta_kwargs=ta_kwargs,
                                                 n_jobs=self.n_jobs,
                                                 initial_budget=initial_budget,
                                                 max_budget=max_budget,
                                                 dask_client=self.dask_client)
        else:
            smac = get_smac_object(scenario_dict=scenario_dict,
                                   seed=seed,
                                   ta=ta,
                                   ta_kwargs=ta_kwargs,
                                   n_jobs=self.n_jobs,
                                   initial_budget=initial_budget,
                                   max_budget=max_budget,
                                   dask_client=self.dask_client)

        if self.ensemble_callback is not None:
            smac.register_callback(self.ensemble_callback)

        self.logger.info("initialised smac, running optimise")

        smac.optimize()

        self.runhistory = smac.solver.runhistory
        self.trajectory = smac.solver.intensifier.traj_logger.trajectory
        if isinstance(smac.solver.tae_runner, DaskParallelRunner):
            self._budget_type = smac.solver.tae_runner.single_worker.budget_type
        elif isinstance(smac.solver.tae_runner, SerialRunner):
            self._budget_type = smac.solver.tae_runner.budget_type
        else:
            raise NotImplementedError(type(smac.solver.tae_runner))

        return self.runhistory, self.trajectory, self._budget_type
