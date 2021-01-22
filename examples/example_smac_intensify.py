import multiprocessing
import tempfile
import time
import typing

import dask
import dask.distributed

import sklearn.datasets
import sklearn.model_selection

from autoPyTorch.datasets.resampling_strategy import CrossValTypes
from autoPyTorch.datasets.tabular_dataset import TabularDataset
from autoPyTorch.optimizer.smbo import AutoMLSMBO
from autoPyTorch.pipeline.components.training.metrics.utils import get_metrics
from autoPyTorch.utils.backend import create
from autoPyTorch.utils.logging_ import setup_logger, start_log_server
from autoPyTorch.utils.pipeline import get_configuration_space
from autoPyTorch.utils.stopwatch import StopWatch


def _start_logger(name, logging_config, backend):
    logger_name = 'AutoML :%s' % (name)
    setup_logger(
        filename='%s.log' % str(logger_name),
        logging_config=logging_config,
        output_dir=backend.temporary_directory,
    )

    # As Auto-sklearn works with distributed process,
    # we implement a logger server that can receive tcp
    # pickled messages. They are unpickled and processed locally
    # under the above logging configuration setting
    # We need to specify the logger_name so that received records
    # are treated under the logger_name ROOT logger setting
    context = multiprocessing.get_context('spawn')
    stop_logging_server = context.Event()
    port = context.Value('l')  # be safe by using a long
    port.value = -1

    logging_server = context.Process(
        target=start_log_server,
        kwargs=dict(
            host='localhost',
            logname=logger_name,
            event=stop_logging_server,
            port=port,
            filename='%s.log' % str(logger_name),
            logging_config=logging_config,
            output_dir=backend.temporary_directory,
        ),
    )

    logging_server.start()

    while True:
        with port.get_lock():
            if port.value == -1:
                time.sleep(0.01)
            else:
                break

    return int(port.value), stop_logging_server


def get_data_to_train() -> typing.Tuple[typing.Any, typing.Any, typing.Any, typing.Any]:
    """
    This function returns a fit dictionary that within itself, contains all
    the information to fit a pipeline
    """

    # Get the training data for tabular classification
    # Move to Australian to showcase numerical vs categorical
    X, y = sklearn.datasets.fetch_openml(data_id=40981, return_X_y=True, as_frame=True)
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
        X,
        y,
        random_state=1,
        test_size=0.2,
    )

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    # Get data to train
    X_train, X_test, y_train, y_test = get_data_to_train()

    # Build a repository with random fitted models
    backend = create(temporary_directory='./tmp/autoPyTorch_smac_test_tmp',
                     output_directory='./tmp/autoPyTorch_smac_test_out',
                     delete_tmp_folder_after_terminate=False)
    # Create the directory structure
    backend._make_internals_directory()

    # Create a datamanager for this toy problem
    datamanager = TabularDataset(
        X=X_train, Y=y_train,
        X_test=X_test, Y_test=y_test,
        resampling_strategy=CrossValTypes.k_fold_cross_validation)
    backend.save_datamanager(datamanager)

    # Build a ensemble from the above components
    # Use dak client here to make sure this is proper working,
    # as with smac we will have to use a client
    dask.config.set({'distributed.worker.daemon': False})
    dask_client = dask.distributed.Client(
        dask.distributed.LocalCluster(
            n_workers=2,
            processes=True,
            threads_per_worker=1,
            # We use the temporal directory to save the
            # dask workers, because deleting workers
            # more time than deleting backend directories
            # This prevent an error saying that the worker
            # file was deleted, so the client could not close
            # the worker properly
            local_directory=tempfile.gettempdir(),
        )
    )
    port, stop_logging_server = _start_logger("trial_australian", logging_config=None, backend=backend)

    info = {'task_type': datamanager.task_type,
            'output_type': datamanager.output_type,
            'categorical_columns': datamanager.categorical_columns,
            'numerical_columns': datamanager.numerical_columns}
    config_space = get_configuration_space(info)
    # Make the optimizer
    smbo = AutoMLSMBO(
        config_space=config_space,
        dataset_name='Australian',
        backend=backend,
        total_walltime_limit=120,
        dask_client=dask_client,
        func_eval_time_limit=60,
        memory_limit=4096,
        metric=get_metrics(dataset_properties=dict({'task_type': datamanager.task_type,
                                                    'output_type': datamanager.output_type}))[0],
        watcher=StopWatch(),
        n_jobs=2,
        ensemble_callback=None,
        logger_port=port
    )

    # Then run the optimization
    run_history, trajectory, budget = smbo.run_smbo()

    for k, v in run_history.data.items():
        print(f"{k}->{v}")
    if not stop_logging_server.is_set():
        stop_logging_server.set()
