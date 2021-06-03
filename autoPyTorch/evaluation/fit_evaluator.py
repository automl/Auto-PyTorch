import time
from multiprocessing.queues import Queue
from typing import Any, Dict, List, Optional, Tuple, Union

from ConfigSpace.configuration_space import Configuration

import numpy as np

from sklearn.base import BaseEstimator

from smac.tae import StatusType

from autoPyTorch.datasets.resampling_strategy import NoResamplingStrategyTypes
from autoPyTorch.evaluation.abstract_evaluator import (
    AbstractEvaluator,
    fit_and_suppress_warnings
)
from autoPyTorch.pipeline.components.training.metrics.base import autoPyTorchMetric
from autoPyTorch.utils.backend import Backend
from autoPyTorch.utils.common import subsampler
from autoPyTorch.utils.hyperparameter_search_space_update import HyperparameterSearchSpaceUpdates


class FitEvaluator(AbstractEvaluator):
    def __init__(self, backend: Backend, queue: Queue,
                 metric: autoPyTorchMetric,
                 budget: float,
                 budget_type: str = None,
                 pipeline_config: Optional[Dict[str, Any]] = None,
                 configuration: Optional[Configuration] = None,
                 seed: int = 1,
                 output_y_hat_optimization: bool = False,
                 num_run: Optional[int] = None,
                 include: Optional[Dict[str, Any]] = None,
                 exclude: Optional[Dict[str, Any]] = None,
                 disable_file_output: Union[bool, List] = False,
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
            pipeline_config=pipeline_config,
            search_space_updates=search_space_updates
        )
        if not isinstance(self.datamanager.resampling_strategy, NoResamplingStrategyTypes):
            raise ValueError(
                "FitEvaluator needs to be fitted on the whole dataset and resampling_strategy "
                "must be `NoResamplingStrategyTypes`, but got {}".format(
                    self.datamanager.resampling_strategy
                ))

        self.splits = self.datamanager.splits
        self.Y_target: Optional[np.ndarray] = None
        self.Y_train_targets: np.ndarray = np.ones(self.y_train.shape) * np.NaN
        self.pipeline: Optional[BaseEstimator] = None

        self.logger.debug("Search space updates :{}".format(self.search_space_updates))
        self.keep_models = keep_models

    def fit_predict_and_loss(self) -> None:
        """Fit, predict and compute the loss for no resampling strategy"""
        assert self.splits is not None, "Can't fit pipeline in {} is datamanager.splits is None" \
            .format(self.__class__.__name__)
        additional_run_info: Optional[Dict] = None
        split_id = 0
        self.logger.info("Starting fit {}".format(split_id))

        pipeline = self._get_pipeline()

        train_split, test_split = self.splits[split_id]
        assert test_split is None
        self.Y_actual_train = self.y_train[train_split]
        y_train_pred, y_valid_pred, y_test_pred = self._fit_and_predict(pipeline, split_id,
                                                                        train_indices=train_split,
                                                                        test_indices=test_split,
                                                                        add_pipeline_to_self=True)
        train_loss = self._loss(self.y_train[train_split], y_train_pred)
        if y_valid_pred is not None:
            loss = self._loss(self.y_valid, y_valid_pred)
        elif y_test_pred is not None:
            loss = self._loss(self.y_test, y_test_pred)
        else:
            loss = train_loss

        additional_run_info = pipeline.get_additional_run_info() if hasattr(
            pipeline, 'get_additional_run_info') else {}

        status = StatusType.SUCCESS

        self.logger.debug("In train evaluator fit_predict_and_loss, num_run: {} loss:{}".format(
            self.num_run,
            loss
        ))
        self.finish_up(
            loss=loss,
            train_loss=train_loss,
            valid_pred=y_valid_pred,
            test_pred=y_test_pred,
            additional_run_info=additional_run_info,
            file_output=True,
            status=status,
            opt_pred=None
        )

    def _fit_and_predict(self, pipeline: BaseEstimator, fold: int, train_indices: Union[np.ndarray, List],
                         test_indices: None,
                         add_pipeline_to_self: bool
                         ) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:

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
            Y_valid_pred,
            Y_test_pred
        ) = self._predict(
            pipeline,
            train_indices=train_indices,
        )

        if add_pipeline_to_self:
            self.pipeline = pipeline

        return Y_train_pred, Y_valid_pred, Y_test_pred

    def _predict(self, pipeline: BaseEstimator,
                 train_indices: Union[np.ndarray, List]
                 ) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:

        train_pred = self.predict_function(subsampler(self.X_train, train_indices), pipeline,
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

        return train_pred, valid_pred, test_pred

    def finish_up(self, loss: Dict[str, float], train_loss: Dict[str, float],
                  valid_pred: Optional[np.ndarray],
                  test_pred: Optional[np.ndarray], additional_run_info: Optional[Dict],
                  file_output: bool, status: StatusType,
                  opt_pred: Optional[np.ndarray]
                  ) -> Optional[Tuple[float, float, int, Dict]]:
        """This function does everything necessary after the fitting is done:

        * predicting
        * saving the necessary files
        We use it as the signal handler so we can recycle the code for the
        normal usecase and when the runsolver kills us here :)"""

        self.duration = time.time() - self.starttime

        if file_output:
            loss_, additional_run_info_ = self.file_output(
                None, valid_pred, test_pred,
            )
        else:
            loss_ = None
            additional_run_info_ = {}

        validation_loss, test_loss = self.calculate_auxiliary_losses(
            valid_pred, test_pred
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

        # Abort if predictions contain NaNs
        for y, s in [
            [Y_valid_pred, 'validation'],
            [Y_test_pred, 'test']
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
            ensemble_predictions=None,
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
    output_y_hat_optimization: bool = False,
    pipeline_config: Optional[Dict[str, Any]] = None,
    budget_type: str = None,
    init_params: Optional[Dict[str, Any]] = None,
    logger_port: Optional[int] = None,
    all_supported_metrics: bool = True,
    search_space_updates: Optional[HyperparameterSearchSpaceUpdates] = None,
    instance: str = None,
) -> None:
    evaluator = FitEvaluator(
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
