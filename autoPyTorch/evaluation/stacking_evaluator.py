from multiprocessing.queues import Queue
import os
import time
from typing import Any, Dict, List, Optional, Tuple, Union

from ConfigSpace.configuration_space import Configuration

import numpy as np

from sklearn.base import BaseEstimator

from smac.tae import StatusType

from autoPyTorch.automl_common.common.utils.backend import Backend
from autoPyTorch.constants import (
    CLASSIFICATION_TASKS,
    MULTICLASSMULTIOUTPUT,
)
from autoPyTorch.datasets.resampling_strategy import CrossValTypes, HoldoutValTypes
from autoPyTorch.evaluation.abstract_evaluator import (
    AbstractEvaluator,
    fit_and_suppress_warnings
)
from autoPyTorch.ensemble.stacking_ensemble import StackingEnsemble
from autoPyTorch.pipeline.components.training.metrics.base import autoPyTorchMetric
from autoPyTorch.utils.common import dict_repr, subsampler
from autoPyTorch.utils.hyperparameter_search_space_update import HyperparameterSearchSpaceUpdates

__all__ = ['StackingEvaluator', 'eval_function']


def _get_y_array(y: np.ndarray, task_type: int) -> np.ndarray:
    if task_type in CLASSIFICATION_TASKS and task_type != \
            MULTICLASSMULTIOUTPUT:
        return y.ravel()
    else:
        return y


class StackingEvaluator(AbstractEvaluator):
    """
    This class builds a pipeline using the provided configuration.
    A pipeline implementing the provided configuration is fitted
    using the datamanager object retrieved from disc, via the backend.
    After the pipeline is fitted, it is save to disc and the performance estimate
    is communicated to the main process via a Queue.

    Attributes:
        backend (Backend):
            An object to interface with the disk storage. In particular, allows to
            access the train and test datasets
        queue (Queue):
            Each worker available will instantiate an evaluator, and after completion,
            it will return the evaluation result via a multiprocessing queue
        metric (autoPyTorchMetric):
            A scorer object that is able to evaluate how good a pipeline was fit. It
            is a wrapper on top of the actual score method (a wrapper on top of scikit
            lean accuracy for example) that formats the predictions accordingly.
        budget: (float):
            The amount of epochs/time a configuration is allowed to run.
        budget_type  (str):
            The budget type, which can be epochs or time
        pipeline_config (Optional[Dict[str, Any]]):
            Defines the content of the pipeline being evaluated. For example, it
            contains pipeline specific settings like logging name, or whether or not
            to use tensorboard.
        configuration (Union[int, str, Configuration]):
            Determines the pipeline to be constructed. A dummy estimator is created for
            integer configurations, a traditional machine learning pipeline is created
            for string based configuration, and NAS is performed when a configuration
            object is passed.
        seed (int):
            A integer that allows for reproducibility of results
        output_y_hat_optimization (bool):
            Whether this worker should output the target predictions, so that they are
            stored on disk. Fundamentally, the resampling strategy might shuffle the
            Y_train targets, so we store the split in order to re-use them for ensemble
            selection.
        num_run (Optional[int]):
            An identifier of the current configuration being fit. This number is unique per
            configuration.
        include (Optional[Dict[str, Any]]):
            An optional dictionary to include components of the pipeline steps.
        exclude (Optional[Dict[str, Any]]):
            An optional dictionary to exclude components of the pipeline steps.
        disable_file_output (Union[bool, List[str]]):
            By default, the model, it's predictions and other metadata is stored on disk
            for each finished configuration. This argument allows the user to skip
            saving certain file type, for example the model, from being written to disk.
        init_params (Optional[Dict[str, Any]]):
            Optional argument that is passed to each pipeline step. It is the equivalent of
            kwargs for the pipeline steps.
        logger_port (Optional[int]):
            Logging is performed using a socket-server scheme to be robust against many
            parallel entities that want to write to the same file. This integer states the
            socket port for the communication channel. If None is provided, a traditional
            logger is used.
        all_supported_metrics  (bool):
            Whether all supported metric should be calculated for every configuration.
        search_space_updates (Optional[HyperparameterSearchSpaceUpdates]):
            An object used to fine tune the hyperparameter search space of the pipeline
    """
    def __init__(self, backend: Backend, queue: Queue,
                 metric: autoPyTorchMetric,
                 budget: float,
                 configuration: Union[int, str, Configuration],
                 budget_type: str = None,
                 pipeline_config: Optional[Dict[str, Any]] = None,
                 seed: int = 1,
                 output_y_hat_optimization: bool = True,
                 num_run: Optional[int] = None,
                 include: Optional[Dict[str, Any]] = None,
                 exclude: Optional[Dict[str, Any]] = None,
                 disable_file_output: Union[bool, List] = False,
                 init_params: Optional[Dict[str, Any]] = None,
                 logger_port: Optional[int] = None,
                 all_supported_metrics: bool = True,
                 search_space_updates: Optional[HyperparameterSearchSpaceUpdates] = None) -> None:
        super().__init__(
            backend=backend,
            queue=queue,
            configuration=configuration,
            metric=metric,
            seed=seed,
            output_y_hat_optimization=output_y_hat_optimization,
            num_run=num_run,
            include=include,
            exclude=exclude,
            disable_file_output=disable_file_output,
            init_params=init_params,
            budget=budget,
            budget_type=budget_type,
            logger_port=logger_port,
            all_supported_metrics=all_supported_metrics,
            pipeline_config=pipeline_config,
            search_space_updates=search_space_updates
        )

        # TODO: we cant store the ensemble pipelines with this class as it is initialised for every TAE (target algorithm evaluation).
        # TODO: Therefore we will have to store pipelines using datamanager and load them, see if we only need predictions.
        # TODO: but we will need the whole pipeline as we would like to predict with different dataset, like val or something

        self.splits = self.datamanager.splits
        if self.splits is None:
            raise AttributeError("Must have called create_splits on {}".format(self.datamanager.__class__.__name__))

        self.logger.debug("Search space updates :{}".format(self.search_space_updates))

    def finish_up(self, loss: Dict[str, float], train_loss: Dict[str, float],
                  valid_pred: Optional[np.ndarray],
                  test_pred: Optional[np.ndarray],
                  pipeline_opt_pred: np.ndarray,
                  ensemble_opt_pred: np.ndarray,
                  additional_run_info: Optional[Dict],
                  file_output: bool, status: StatusType,                  
                  ) -> Optional[Tuple[float, float, int, Dict]]:
        """This function does everything necessary after the fitting is done:

        * predicting
        * saving the necessary files
        We use it as the signal handler so we can recycle the code for the
        normal usecase and when the runsolver kills us here :)"""

        self.duration = time.time() - self.starttime

        if file_output:
            loss_, additional_run_info_ = self.file_output(
                ensemble_opt_pred, valid_pred, test_pred
            )
        else:
            loss_ = None
            additional_run_info_ = {}

        validation_loss, test_loss = self.calculate_auxiliary_losses(
            valid_pred, test_pred
        )

        pipeline_loss, _ = self.calculate_auxiliary_losses(
            pipeline_opt_pred, None
        )
        if loss_ is not None:
            return self.duration, loss_, self.seed, additional_run_info_

        cost = loss[self.metric.name]

        additional_run_info = (
            {} if additional_run_info is None else additional_run_info
        )
        for metric_name, value in loss.items():
            additional_run_info[metric_name] = value
        additional_run_info['duration'] = self.duration
        additional_run_info['num_run'] = self.num_run
        if train_loss is not None:
            additional_run_info['train_loss'] = train_loss
        if validation_loss is not None:
            additional_run_info['validation_loss'] = validation_loss
        if test_loss is not None:
            additional_run_info['test_loss'] = test_loss
        if pipeline_loss is not None:
            additional_run_info['pipeline_loss'] = pipeline_loss
        rval_dict = {'loss': cost,
                     'additional_run_info': additional_run_info,
                     'status': status}

        self.queue.put(rval_dict)
        return None

    def file_output(
        self,
        Y_optimization_pred: np.ndarray,
        Y_valid_pred: np.ndarray,
        Y_test_pred: np.ndarray,
    ) -> Tuple[Optional[float], Dict]:

        # Abort in case of shape misalignment
        if self.Y_optimization.shape[0] != Y_optimization_pred.shape[0]:
            return (
                1.0,
                {
                    'error':
                        "Targets %s and prediction %s don't have "
                        "the same length. Probably training didn't "
                        "finish" % (self.Y_optimization.shape, Y_optimization_pred.shape)
                },
            )

        # Abort if predictions contain NaNs
        for y, s in [
            # Y_train_pred deleted here. Fix unittest accordingly.
            [Y_optimization_pred, 'optimization'],
            [Y_valid_pred, 'validation'],
            [Y_test_pred, 'test'],
        ]:
            if y is not None and not np.all(np.isfinite(y)):
                return (
                    1.0,
                    {
                        'error':
                            'Model predictions for %s set contains NaNs.' % s
                    },
                )

        # Abort if we don't want to output anything.
        if hasattr(self, 'disable_file_output'):
            if self.disable_file_output:
                return None, {}
            else:
                self.disabled_file_outputs = []

        # This file can be written independently of the others down bellow
        if 'y_optimization' not in self.disabled_file_outputs:
            if self.output_y_hat_optimization:
                self.backend.save_targets_ensemble(self.Y_optimization)

        if hasattr(self, 'pipeline') and self.pipeline is not None:
            if 'pipeline' not in self.disabled_file_outputs:
                pipeline = self.pipeline
            else:
                pipeline = None
        else:
            pipeline = None

        self.logger.debug("Saving model {}_{}_{} to disk".format(self.seed, self.num_run, self.budget))
        self.backend.save_numrun_to_dir(
            seed=int(self.seed),
            idx=int(self.num_run),
            budget=float(self.budget),
            model=pipeline,
            cv_model=None,
            ensemble_predictions=(
                Y_optimization_pred if 'y_optimization' not in
                                       self.disabled_file_outputs else None
            ),
            valid_predictions=(
                Y_valid_pred if 'y_valid' not in
                                self.disabled_file_outputs else None
            ),
            test_predictions=(
                Y_test_pred if 'y_test' not in
                               self.disabled_file_outputs else None
            ),
        )

        return None, {}

    def fit_predict_and_loss(self) -> None:
        """Fit, predict and compute the loss for cross-validation and
        holdout"""
        assert self.splits is not None, "Can't fit pipeline in {} is datamanager.splits is None" \
            .format(self.__class__.__name__)
        additional_run_info: Optional[Dict] = None
        split_id = 0
        self.logger.info("Starting fit {}".format(split_id))

        pipeline = self._get_pipeline()

        train_split, test_split = self.splits[split_id]
        self.Y_optimization = self.y_train[test_split]
        self.Y_actual_train = self.y_train[train_split]
        (
            y_train_pred,
            y_pipeline_opt_pred,
            y_ensemble_opt_pred,
            y_valid_pred,
            y_test_pred
        ) = self._fit_and_predict(pipeline, split_id,
                                  train_indices=train_split,
                                  test_indices=test_split
                                  )
        train_loss = self._loss(self.y_train[train_split], y_train_pred)
        loss = self._loss(self.y_train[test_split], y_ensemble_opt_pred)

        additional_run_info = pipeline.get_additional_run_info() if hasattr(
            pipeline, 'get_additional_run_info') else {}

        status = StatusType.SUCCESS

        self.logger.debug("In train evaluator.fit_predict_and_loss, num_run: {} loss:{},"
                            " status: {},\nadditional run info:\n{}".format(self.num_run,
                                                                            loss,
                                                                            dict_repr(additional_run_info),
                                                                            status))
        self.finish_up(
            loss=loss,
            train_loss=train_loss,
            ensemble_opt_pred=y_ensemble_opt_pred,
            valid_pred=y_valid_pred,
            test_pred=y_test_pred,
            additional_run_info=additional_run_info,
            file_output=True,
            status=status,
            pipeline_opt_pred=y_pipeline_opt_pred
        )


    def _fit_and_predict(
        self,
        pipeline: BaseEstimator,
        fold: int,
        train_indices: Union[np.ndarray, List],
        test_indices: Union[np.ndarray, List],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:

        # See autoPyTorch/pipeline/components/base_component.py::autoPyTorchComponent for more details
        # about fit_dictionary
        X = {'train_indices': train_indices,
             'val_indices': test_indices,
             'split_id': fold,
             'num_run': self.num_run,
             **self.fit_dictionary}  # fit dictionary
        y = None
        fit_and_suppress_warnings(self.logger, pipeline, X, y)
        self.logger.info("Model fitted, now predicting")
        (
            Y_train_pred, Y_pipeline_opt_pred, Y_ensemble_opt_pred, Y_valid_pred, Y_test_pred
        ) = self._predict(
            pipeline,
            train_indices=train_indices,
            test_indices=test_indices,
        )

        self.pipeline = pipeline

        return Y_train_pred, Y_pipeline_opt_pred, Y_ensemble_opt_pred, Y_valid_pred, Y_test_pred

    def _predict(
        self,
        pipeline: BaseEstimator,
        test_indices: Union[np.ndarray, List],
        train_indices: Union[np.ndarray, List]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
        # TODO: load ensemble members and predict using the whole ensemble.
        # TODO: we need some function to pass this pipeline to the last stored ensemble replace 
        # TODO: model j, where j = ensemble.iteration mod m. then we need to predict
        # TODO: Also, we will pass the predictions from this pipeline as that is what is needed
        # TODO: to create the ensemble. 
        train_pred = self.predict_function(subsampler(self.X_train, train_indices), pipeline,
                                           self.y_train[train_indices])

        pipeline_opt_pred = self.predict_function(subsampler(self.X_train, test_indices), pipeline,
                                         self.y_train[train_indices])

        ensemble_dir = self.backend.get_ensemble_dir()
        if len(os.listdir(ensemble_dir)) >= 1:
            old_ensemble = self.backend.load_ensemble(self.seed)
            assert isinstance(old_ensemble, StackingEnsemble)
            ensemble_opt_pred = old_ensemble.predict_with_current_model(pipeline_opt_pred)
        else:
            ensemble_opt_pred = pipeline_opt_pred.copy()

        if self.X_valid is not None:
            valid_pred = self.predict_function(self.X_valid, pipeline,
                                               self.y_valid)
        else:
            valid_pred = None

        if self.X_test is not None:
            test_pred = self.predict_function(self.X_test, pipeline,
                                              self.y_train[train_indices])
        else:
            test_pred = None

        return train_pred, pipeline_opt_pred, ensemble_opt_pred, valid_pred, test_pred


# create closure for evaluating an algorithm
def eval_function(
    backend: Backend,
    queue: Queue,
    metric: autoPyTorchMetric,
    budget: float,
    config: Optional[Configuration],
    seed: int,
    num_run: int,
    include: Optional[Dict[str, Any]],
    exclude: Optional[Dict[str, Any]],
    disable_file_output: Union[bool, List],
    output_y_hat_optimization: bool,
    pipeline_config: Optional[Dict[str, Any]] = None,
    budget_type: str = None,
    init_params: Optional[Dict[str, Any]] = None,
    logger_port: Optional[int] = None,
    all_supported_metrics: bool = True,
    search_space_updates: Optional[HyperparameterSearchSpaceUpdates] = None,
    instance: str = None,
) -> None:
    """
    This closure allows the communication between the ExecuteTaFuncWithQueue and the
    pipeline trainer (TrainEvaluator).

    Fundamentally, smac calls the ExecuteTaFuncWithQueue.run() method, which internally
    builds a TrainEvaluator. The TrainEvaluator builds a pipeline, stores the output files
    to disc via the backend, and puts the performance result of the run in the queue.


    Attributes:
        backend (Backend):
            An object to interface with the disk storage. In particular, allows to
            access the train and test datasets
        queue (Queue):
            Each worker available will instantiate an evaluator, and after completion,
            it will return the evaluation result via a multiprocessing queue
        metric (autoPyTorchMetric):
            A scorer object that is able to evaluate how good a pipeline was fit. It
            is a wrapper on top of the actual score method (a wrapper on top of scikit
            lean accuracy for example) that formats the predictions accordingly.
        budget: (float):
            The amount of epochs/time a configuration is allowed to run.
        budget_type  (str):
            The budget type, which can be epochs or time
        pipeline_config (Optional[Dict[str, Any]]):
            Defines the content of the pipeline being evaluated. For example, it
            contains pipeline specific settings like logging name, or whether or not
            to use tensorboard.
        config (Union[int, str, Configuration]):
            Determines the pipeline to be constructed.
        seed (int):
            A integer that allows for reproducibility of results
        output_y_hat_optimization (bool):
            Whether this worker should output the target predictions, so that they are
            stored on disk. Fundamentally, the resampling strategy might shuffle the
            Y_train targets, so we store the split in order to re-use them for ensemble
            selection.
        num_run (Optional[int]):
            An identifier of the current configuration being fit. This number is unique per
            configuration.
        include (Optional[Dict[str, Any]]):
            An optional dictionary to include components of the pipeline steps.
        exclude (Optional[Dict[str, Any]]):
            An optional dictionary to exclude components of the pipeline steps.
        disable_file_output (Union[bool, List[str]]):
            By default, the model, it's predictions and other metadata is stored on disk
            for each finished configuration. This argument allows the user to skip
            saving certain file type, for example the model, from being written to disk.
        init_params (Optional[Dict[str, Any]]):
            Optional argument that is passed to each pipeline step. It is the equivalent of
            kwargs for the pipeline steps.
        logger_port (Optional[int]):
            Logging is performed using a socket-server scheme to be robust against many
            parallel entities that want to write to the same file. This integer states the
            socket port for the communication channel. If None is provided, a traditional
            logger is used.
        instance (str):
            An instance on which to evaluate the current pipeline. By default we work
            with a single instance, being the provided X_train, y_train of a single dataset.
            This instance is a compatibility argument for SMAC, that is capable of working
            with multiple datasets at the same time.
    """
    evaluator = StackingEvaluator(
        backend=backend,
        queue=queue,
        metric=metric,
        configuration=config,
        seed=seed,
        num_run=num_run,
        output_y_hat_optimization=output_y_hat_optimization,
        include=include,
        exclude=exclude,
        disable_file_output=disable_file_output,
        init_params=init_params,
        budget=budget,
        budget_type=budget_type,
        logger_port=logger_port,
        all_supported_metrics=all_supported_metrics,
        pipeline_config=pipeline_config,
        search_space_updates=search_space_updates
    )
    evaluator.fit_predict_and_loss()
