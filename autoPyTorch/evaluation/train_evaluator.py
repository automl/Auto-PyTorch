from multiprocessing.queues import Queue
from typing import Any, Dict, List, Optional, Tuple, Union

from ConfigSpace.configuration_space import Configuration

import numpy as np

from sklearn.base import BaseEstimator

from smac.tae import StatusType

from autoPyTorch.automl_common.common.utils.backend import Backend
from autoPyTorch.constants import (
    CLASSIFICATION_TASKS,
    MULTICLASSMULTIOUTPUT
)
from autoPyTorch.datasets.resampling_strategy import CrossValTypes, HoldoutValTypes
from autoPyTorch.evaluation.abstract_evaluator import (
    AbstractEvaluator,
    fit_and_suppress_warnings
)
from autoPyTorch.evaluation.utils import DisableFileOutputParameters
from autoPyTorch.pipeline.components.training.metrics.base import autoPyTorchMetric
from autoPyTorch.utils.common import dict_repr, subsampler
from autoPyTorch.utils.hyperparameter_search_space_update import HyperparameterSearchSpaceUpdates

__all__ = ['TrainEvaluator', 'eval_train_function']


def _get_y_array(y: np.ndarray, task_type: int) -> np.ndarray:
    if task_type in CLASSIFICATION_TASKS and task_type != \
            MULTICLASSMULTIOUTPUT:
        return y.ravel()
    else:
        return y


class TrainEvaluator(AbstractEvaluator):
    """
    This class builds a pipeline using the provided configuration.
    A pipeline implementing the provided configuration is fitted
    using the datamanager object retrieved from disc, via the backend.
    After the pipeline is fitted, it is save to disc and the performance estimate
    is communicated to the main process via a Queue. It is only compatible
    with `CrossValTypes`, `HoldoutValTypes`, i.e, when the training data
    is split and the validation set is used for SMBO optimisation.

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
        pipeline_options (Optional[Dict[str, Any]]):
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
        disable_file_output (Optional[List[Union[str, DisableFileOutputParameters]]]):
            Used as a list to pass more fine-grained
            information on what to save. Must be a member of `DisableFileOutputParameters`.
            Allowed elements in the list are:

            + `y_optimization`:
                do not save the predictions for the optimization set,
                which would later on be used to build an ensemble. Note that SMAC
                optimizes a metric evaluated on the optimization set.
            + `pipeline`:
                do not save any individual pipeline files
            + `pipelines`:
                In case of cross validation, disables saving the joint model of the
                pipelines fit on each fold.
            + `y_test`:
                do not save the predictions for the test set.
            + `all`:
                do not save any of the above.
            For more information check `autoPyTorch.evaluation.utils.DisableFileOutputParameters`.
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
                 pipeline_options: Optional[Dict[str, Any]] = None,
                 seed: int = 1,
                 output_y_hat_optimization: bool = True,
                 num_run: Optional[int] = None,
                 include: Optional[Dict[str, Any]] = None,
                 exclude: Optional[Dict[str, Any]] = None,
                 disable_file_output: Optional[List[Union[str, DisableFileOutputParameters]]] = None,
                 init_params: Optional[Dict[str, Any]] = None,
                 logger_port: Optional[int] = None,
                 keep_models: Optional[bool] = None,
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
            pipeline_options=pipeline_options,
            search_space_updates=search_space_updates
        )

        if not isinstance(self.resampling_strategy, (CrossValTypes, HoldoutValTypes)):
            raise ValueError(
                f'resampling_strategy for TrainEvaluator must be in '
                f'(CrossValTypes, HoldoutValTypes), but got {self.resampling_strategy}'
            )

        self.num_folds: int = len(self.splits)
        self.Y_targets: List[Optional[np.ndarray]] = [None] * self.num_folds
        self.Y_train_targets: np.ndarray = np.ones(self.y_train.shape) * np.NaN
        self.pipelines: List[Optional[BaseEstimator]] = [None] * self.num_folds
        self.indices: List[Optional[Tuple[Union[np.ndarray, List], Union[np.ndarray, List]]]] = [None] * self.num_folds

        self.logger.debug("Search space updates :{}".format(self.search_space_updates))
        self.keep_models = keep_models

    def fit_predict_and_loss(self) -> None:
        """Fit, predict and compute the loss for cross-validation and
        holdout"""
        assert self.splits is not None, "Can't fit pipeline in {} is datamanager.splits is None" \
            .format(self.__class__.__name__)
        additional_run_info: Optional[Dict] = None
        if self.num_folds == 1:
            split_id = 0
            self.logger.info("Starting fit {}".format(split_id))

            pipeline = self._get_pipeline()

            train_split, test_split = self.splits[split_id]
            self.Y_optimization = self.y_train[test_split]
            self.Y_actual_train = self.y_train[train_split]
            y_train_pred, y_opt_pred, y_valid_pred, y_test_pred = self._fit_and_predict(pipeline, split_id,
                                                                                        train_indices=train_split,
                                                                                        test_indices=test_split,
                                                                                        add_pipeline_to_self=True)
            train_loss = self._loss(self.y_train[train_split], y_train_pred)
            loss = self._loss(self.y_train[test_split], y_opt_pred)

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
                opt_pred=y_opt_pred,
                valid_pred=y_valid_pred,
                test_pred=y_test_pred,
                additional_run_info=additional_run_info,
                file_output=True,
                status=status,
            )

        else:
            Y_train_pred: List[Optional[np.ndarray]] = [None] * self.num_folds
            Y_optimization_pred: List[Optional[np.ndarray]] = [None] * self.num_folds
            Y_valid_pred: List[Optional[np.ndarray]] = [None] * self.num_folds
            Y_test_pred: List[Optional[np.ndarray]] = [None] * self.num_folds
            train_splits: List[Optional[Union[np.ndarray, List]]] = [None] * self.num_folds

            self.pipelines = [self._get_pipeline() for _ in range(self.num_folds)]

            # stores train loss of each fold.
            train_losses = [np.NaN] * self.num_folds
            # used as weights when averaging train losses.
            train_fold_weights = [np.NaN] * self.num_folds
            # stores opt (validation) loss of each fold.
            opt_losses = [np.NaN] * self.num_folds
            # weights for opt_losses.
            opt_fold_weights = [np.NaN] * self.num_folds

            additional_run_info = {}

            for i, (train_split, test_split) in enumerate(self.splits):
                pipeline = self.pipelines[i]
                train_pred, opt_pred, valid_pred, test_pred = self._fit_and_predict(pipeline, i,
                                                                                    train_indices=train_split,
                                                                                    test_indices=test_split,
                                                                                    add_pipeline_to_self=False)
                Y_train_pred[i] = train_pred
                Y_optimization_pred[i] = opt_pred
                Y_valid_pred[i] = valid_pred
                Y_test_pred[i] = test_pred
                train_splits[i] = train_split

                self.Y_train_targets[train_split] = self.y_train[train_split]
                self.Y_targets[i] = self.y_train[test_split]
                # Compute train loss of this fold and store it. train_loss could
                # either be a scalar or a dict of scalars with metrics as keys.
                train_loss = self._loss(
                    self.Y_train_targets[train_split],
                    train_pred,
                )
                train_losses[i] = train_loss
                # number of training data points for this fold. Used for weighting
                # the average.
                train_fold_weights[i] = len(train_split)

                # Compute validation loss of this fold and store it.
                optimization_loss = self._loss(
                    self.Y_targets[i],
                    opt_pred,
                )
                opt_losses[i] = optimization_loss
                # number of optimization data points for this fold.
                # Used for weighting the average.
                opt_fold_weights[i] = len(train_split)
                additional_run_info.update(pipeline.get_additional_run_info() if hasattr(
                    pipeline, 'get_additional_run_info') and pipeline.get_additional_run_info() is not None else {})
            # Compute weights of each fold based on the number of samples in each
            # fold.
            train_fold_weights = [w / sum(train_fold_weights)
                                  for w in train_fold_weights]
            opt_fold_weights = [w / sum(opt_fold_weights)
                                for w in opt_fold_weights]

            # train_losses is a list of dicts. It is
            # computed using the target metric (self.metric).
            train_loss = {}
            for metric in train_losses[0].keys():
                train_loss[metric] = np.average(
                    [
                        train_losses[i][metric]
                        for i in range(self.num_folds)
                    ],
                    weights=train_fold_weights
                )

            opt_loss = {}
            # self.logger.debug("OPT LOSSES: {}".format(opt_losses if opt_losses is not None else None))
            for metric in opt_losses[0].keys():
                opt_loss[metric] = np.average(
                    [
                        opt_losses[i][metric]
                        for i in range(self.num_folds)
                    ],
                    weights=opt_fold_weights,
                )
            Y_targets = self.Y_targets
            Y_train_targets = self.Y_train_targets

            Y_optimization_preds = np.concatenate(
                [Y_optimization_pred[i] for i in range(self.num_folds)
                 if Y_optimization_pred[i] is not None])
            Y_targets = np.concatenate([
                Y_targets[i] for i in range(self.num_folds)
                if Y_targets[i] is not None
            ])

            if self.X_valid is not None:
                Y_valid_preds = np.array([Y_valid_pred[i]
                                          for i in range(self.num_folds)
                                          if Y_valid_pred[i] is not None])
                # Average the predictions of several pipelines
                if len(Y_valid_preds.shape) == 3:
                    Y_valid_preds = np.nanmean(Y_valid_preds, axis=0)
            else:
                Y_valid_preds = None

            if self.X_test is not None:
                Y_test_preds = np.array([Y_test_pred[i]
                                         for i in range(self.num_folds)
                                         if Y_test_pred[i] is not None])
                # Average the predictions of several pipelines
                if len(Y_test_preds.shape) == 3:
                    Y_test_preds = np.nanmean(Y_test_preds, axis=0)
            else:
                Y_test_preds = None

            self.Y_optimization = Y_targets
            self.Y_actual_train = Y_train_targets

            self.pipeline = self._get_pipeline()

            status = StatusType.SUCCESS
            self.logger.debug("In train evaluator fit_predict_and_loss, num_run: {} loss:{}".format(
                self.num_run,
                opt_loss
            ))
            self.finish_up(
                loss=opt_loss,
                train_loss=train_loss,
                opt_pred=Y_optimization_preds,
                valid_pred=Y_valid_preds,
                test_pred=Y_test_preds,
                additional_run_info=additional_run_info,
                file_output=True,
                status=status,
            )

    def _fit_and_predict(self, pipeline: BaseEstimator, fold: int, train_indices: Union[np.ndarray, List],
                         test_indices: Union[np.ndarray, List],
                         add_pipeline_to_self: bool
                         ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:

        self.indices[fold] = ((train_indices, test_indices))

        X = {'train_indices': train_indices,
             'val_indices': test_indices,
             'split_id': fold,
             'num_run': self.num_run,
             **self.fit_dictionary}  # fit dictionary
        y = None
        fit_and_suppress_warnings(self.logger, pipeline, X, y)
        self.logger.info("Model fitted, now predicting")
        (
            Y_train_pred,
            Y_opt_pred,
            Y_valid_pred,
            Y_test_pred
        ) = self._predict(
            pipeline,
            train_indices=train_indices,
            test_indices=test_indices,
        )

        if add_pipeline_to_self:
            self.pipeline = pipeline
        else:
            self.pipelines[fold] = pipeline

        return Y_train_pred, Y_opt_pred, Y_valid_pred, Y_test_pred

    def _predict(self, pipeline: BaseEstimator,
                 test_indices: Union[np.ndarray, List],
                 train_indices: Union[np.ndarray, List]
                 ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:

        train_pred = self.predict_function(subsampler(self.X_train, train_indices), pipeline,
                                           self.y_train[train_indices])

        opt_pred = self.predict_function(subsampler(self.X_train, test_indices), pipeline,
                                         self.y_train[train_indices])

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

        return train_pred, opt_pred, valid_pred, test_pred


# create closure for evaluating an algorithm
def eval_train_function(
    backend: Backend,
    queue: Queue,
    metric: autoPyTorchMetric,
    budget: float,
    config: Optional[Configuration],
    seed: int,
    output_y_hat_optimization: bool,
    num_run: int,
    include: Optional[Dict[str, Any]],
    exclude: Optional[Dict[str, Any]],
    disable_file_output: Optional[List[Union[str, DisableFileOutputParameters]]] = None,
    pipeline_options: Optional[Dict[str, Any]] = None,
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
        pipeline_options (Optional[Dict[str, Any]]):
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
    evaluator = TrainEvaluator(
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
        pipeline_options=pipeline_options,
        search_space_updates=search_space_updates,
    )
    evaluator.fit_predict_and_loss()
