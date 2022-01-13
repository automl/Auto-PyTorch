from json import dump, load
import json
import logging
import os
import pickle
import re
import time
import traceback
from typing import List, Union, Dict, Tuple, Optional

import dask.distributed
from distributed.utils import Any
from numpy.random.mtrand import seed

from smac.optimizer.smbo import SMBO
from smac.runhistory.runhistory import RunInfo, RunKey
from torch.utils import data
from autoPyTorch.datasets.resampling_strategy import CrossValTypes, HoldoutValTypes, NoResamplingStrategyTypes

from autoPyTorch.optimizer.utils import AdjustRunHistoryCallback
from autoPyTorch.automl_common.common.utils.backend import Backend
from autoPyTorch.automl_common.common.utils.logging_ import get_named_client_logger


MODEL_FN_RE = r'_([0-9]*)_([0-9]*)_([0-9]+\.*[0-9]*)\.npy'

class RunHistoryUpdaterManager(AdjustRunHistoryCallback):
    def __init__(
        self,
        backend: Backend,
        dataset_name: str,
        resampling_strategy: Union[
            HoldoutValTypes, CrossValTypes, NoResamplingStrategyTypes
        ],
        resampling_strategy_args: Dict[str, Any], 
        logger_port: int = logging.handlers.DEFAULT_TCP_LOGGING_PORT,
    ):
        """ 
        SMAC callback to update run history
        Args:
            backend: util.backend.Backend
                backend to write and read files
            logger_port: int
                port in where to publish a msg

        Returns:
            List[Tuple[int, float, float, float]]:
                A list with the performance history of this ensemble, of the form
                [[pandas_timestamp, train_performance, val_performance, test_performance], ...]
        """

        self.backend = backend
        
        self.logger_port = logger_port

        # We only submit new ensembles when there is not an active ensemble job
        self.futures: List[dask.Future] = []

        # The last criteria is the number of iterations
        self.iteration = 0

        # Keep track of when we started to know when we need to finish!
        self.start_time = time.time()
        self.dataset_name = dataset_name
        self.resampling_strategy = resampling_strategy
        self.resampling_strategy_args = resampling_strategy_args

    def __call__(
        self,
        smbo: 'SMBO',
    ) -> None:
        self.adjust_run_history(smbo.tae_runner.client)

    def adjust_run_history(
        self,
        dask_client: dask.distributed.Client,
        unit_test: bool = False
    ) -> None:

        # The second criteria is elapsed time
        elapsed_time = time.time() - self.start_time

        logger = get_named_client_logger(
            name='EnsembleBuilder',
            port=self.logger_port,
        )

        if len(self.futures) != 0:
            if self.futures[0].done():
                result = self.futures.pop().result()
                if result:
                    response = result
                    logger.debug("iteration={} @ elapsed_time={} has response={}".format(
                        self.iteration,
                        elapsed_time,
                        response,
                    ))

        # Only submit new jobs if the previous ensemble job finished
        if len(self.futures) == 0:

            # Add the result of the run
            # On the next while iteration, no references to
            # ensemble builder object, so it should be garbage collected to
            # save memory while waiting for resources
            # Also, notice how ensemble nbest is returned, so we don't waste
            # iterations testing if the deterministic predictions size can
            # be fitted in memory
            try:
                # Submit a Dask job from this job, to properly
                # see it in the dask diagnostic dashboard
                # Notice that the forked ensemble_builder_process will
                # wait for the below function to be done
                self.futures.append(
                    dask_client.submit(
                        return_run_info_cost,
                        backend=self.backend,
                        dataset_name=self.dataset_name,
                        iteration=self.iteration,
                        resampling_strategy=self.resampling_strategy,
                        resampling_strategy_args=self.resampling_strategy_args,
                        logger_port=self.logger_port,
                        priority=100
                        )
                )

                logger.info(
                    "{}/{} Started Ensemble builder job at {} for iteration {}.".format(
                        # Log the client to make sure we
                        # remain connected to the scheduler
                        self.futures[0],
                        dask_client,
                        time.strftime("%Y.%m.%d-%H.%M.%S"),
                        self.iteration,
                    ),
                )
                self.iteration += 1
            except Exception as e:
                exception_traceback = traceback.format_exc()
                error_message = repr(e)
                logger.critical(exception_traceback)
                logger.critical(error_message)


def return_run_info_cost(
    backend: Backend,
    dataset_name: str,
    resampling_strategy: Union[
        HoldoutValTypes, CrossValTypes, NoResamplingStrategyTypes
    ],
    resampling_strategy_args: Dict[str, Any],
    iteration: int,
    logger_port: int = logging.handlers.DEFAULT_TCP_LOGGING_PORT,
) -> Optional[List[Tuple[RunKey, float]]]:
    """
    A short function to fit and create an ensemble. It is just a wrapper to easily send
    a request to dask to create an ensemble and clean the memory when finished
    Parameters
    ----------
        backend: util.backend.Backend
            backend to write and read files
        dataset_name: str
            name of dataset
        metrics: List[autoPyTorchMetric],
            A set of metrics that will be used to get performance estimates
        opt_metric:
            Name of the metric to optimize
        task_type: int
            type of output expected in the ground truth
        ensemble_size: int
            maximal size of ensemble (passed to ensemble.ensemble_selection)
        ensemble_nbest: int/float
            if int: consider only the n best prediction
            if float: consider only this fraction of the best models
            Both wrt to validation predictions
            If performance_range_threshold > 0, might return less models
        max_models_on_disc: int
           Defines the maximum number of models that are kept in the disc.
           If int, it must be greater or equal than 1, and dictates the max number of
           models to keep.
           If float, it will be interpreted as the max megabytes allowed of disc space. That
           is, if the number of ensemble candidates require more disc space than this float
           value, the worst models will be deleted to keep within this budget.
           Models and predictions of the worst-performing models will be deleted then.
           If None, the feature is disabled.
           It defines an upper bound on the models that can be used in the ensemble.
        seed: int
            random seed
        precision (int): [16,32,64,128]
            precision of floats to read the predictions
        memory_limit: Optional[int]
            memory limit in mb. If ``None``, no memory limit is enforced.
        read_at_most: int
            read at most n new prediction files in each iteration
        end_at: float
            At what time the job must finish. Needs to be the endtime and not the time left
            because we do not know when dask schedules the job.
        iteration: int
            The current iteration
        pynisher_context: str
            Context to use for multiprocessing, can be either fork, spawn or forkserver.
        logger_port: int
            The port where the logging server is listening to.
        unit_test: bool
            Turn on unit testing mode. This currently makes fit_ensemble raise a MemoryError.
            Having this is very bad coding style, but I did not find a way to make
            unittest.mock work through the pynisher with all spawn contexts. If you know a
            better solution, please let us know by opening an issue.
    Returns
    -------
        List[Tuple[int, float, float, float]]
            A list with the performance history of this ensemble, of the form
            [[pandas_timestamp, train_performance, val_performance, test_performance], ...]
    """
    result = RunHistoryUpdater(
        backend=backend,
        dataset_name=dataset_name,
        resampling_strategy=resampling_strategy,
        resampling_strategy_args=resampling_strategy_args,
        logger_port=logger_port,
    ).run(
        iteration=iteration,
    )
    return result


class RunHistoryUpdater:
    def __init__(
        self,
        backend: Backend,
        dataset_name: str,
        resampling_strategy: Union[
            HoldoutValTypes, CrossValTypes, NoResamplingStrategyTypes
        ],
        resampling_strategy_args: Dict[str, Any], 
        logger_port: int = logging.handlers.DEFAULT_TCP_LOGGING_PORT,
    ):
        """ 
        SMAC callback to update run history
        Args:
            backend: util.backend.Backend
                backend to write and read files
            logger_port: int
                port in where to publish a msg

        Returns:
            List[Tuple[int, float, float, float]]:
                A list with the performance history of this ensemble, of the form
                [[pandas_timestamp, train_performance, val_performance, test_performance], ...]
        """

        self.model_fn_re = re.compile(MODEL_FN_RE)
        self.logger_port = logger_port
        self.logger = get_named_client_logger(
            name='RunHistoryUpdater',
            port=self.logger_port,
        )
        self.ensemble_loss_file = os.path.join(backend.internals_directory, 'ensemble_read_losses.pkl')
        if isinstance(resampling_strategy, CrossValTypes):
            num_splits = resampling_strategy_args['num_splits']
            self.instances = [[json.dumps({'task_id': dataset_name,
                                      'fold': fold_number})]
                         for fold_number in range(num_splits)]
        else:
            self.instances = [[json.dumps({'task_id': dataset_name})]]

    def run(self, iteration: int) -> Optional[List[Tuple[RunKey, float]]]:
        self.logger.info(f"Starting iteration {iteration} of run history updater")
        results: List[Tuple[RunInfo, float]] = []
        if os.path.exists(self.ensemble_loss_file):
            try:
                with (open(self.ensemble_loss_file, "rb")) as memory:
                    read_losses = pickle.load(memory)
            except Exception as e:
                self.logger.debug(f"Could not read losses at iteration: {iteration} with exception {e}")
                return
            else:
                for k in read_losses.keys():
                    match = self.model_fn_re.search(k)
                    if match is None or read_losses[k]["loaded"] != 1:
                        continue
                    else:
                        _num_run = int(match.group(2))
                        _budget = float(match.group(3))
                        run_key = RunKey(
                            seed=0,  # 0 is hardcoded for the runhistory coming from smac
                            config_id=_num_run,
                            budget=_budget,
                            instance_id=self.instances[-1]
                        )
                        results.append((run_key, read_losses[k]["ens_loss"]))
        return results
