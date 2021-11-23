from autoPyTorch.evaluation.train_evaluator import TrainEvaluator
from autoPyTorch.evaluation.abstract_evaluator import DummyClassificationPipeline

from multiprocessing.queues import Queue
from typing import Any, Dict, List, Optional, Tuple, Union, no_type_check, ClassVar
from functools import partial
import warnings

from ConfigSpace.configuration_space import Configuration

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator

from smac.tae import StatusType


from autoPyTorch.pipeline.components.training.metrics.base import autoPyTorchMetric
from autoPyTorch.utils.backend import Backend
from autoPyTorch.utils.common import subsampler
from autoPyTorch.utils.hyperparameter_search_space_update import HyperparameterSearchSpaceUpdates
from autoPyTorch.datasets.time_series_dataset import TimeSeriesForecastingDataset
from autoPyTorch.pipeline.time_series_forecasting import TimeSeriesForecastingPipeline
from autoPyTorch.constants_forecasting import SEASONALITY_MAP


class TimeSeriesForecastingTrainEvaluator(TrainEvaluator):
    def __init__(self, backend: Backend, queue: Queue,
                 metric: autoPyTorchMetric,
                 budget: float,
                 budget_type: str = None,
                 pipeline_config: Optional[Dict[str, Any]] = None,
                 configuration: Optional[Configuration] = None,
                 seed: int = 1,
                 output_y_hat_optimization: bool = True,
                 num_run: Optional[int] = None,
                 include: Optional[Dict[str, Any]] = None,
                 exclude: Optional[Dict[str, Any]] = None,
                 disable_file_output: Union[bool, List] = False,
                 init_params: Optional[Dict[str, Any]] = None,
                 logger_port: Optional[int] = None,
                 keep_models: Optional[bool] = None,
                 all_supported_metrics: bool = True,
                 search_space_updates: Optional[HyperparameterSearchSpaceUpdates] = None) -> None:
        super(TimeSeriesForecastingTrainEvaluator, self).__init__(
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
            keep_models=keep_models,
            all_supported_metrics=all_supported_metrics,
            pipeline_config=pipeline_config,
            search_space_updates=search_space_updates
        )
        self.datamanager: TimeSeriesForecastingDataset
        self.n_prediction_steps = self.datamanager.n_prediction_steps
        self.num_sequences = self.datamanager.num_sequences
        self.seq_length_min = np.min(self.num_sequences)
        seasonality = SEASONALITY_MAP.get(self.datamanager.freq, 1)
        if isinstance(seasonality, list):
            seasonality = min(seasonality)  # Use to calculate MASE
        self.seasonality = int(seasonality)


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

            # TODO move these lines to TimeSeriesForecastingDatasets (Create a new object there that inherents from
            #  np.array while return multiple values by __get_item__)!
            # the +1 in the end indicates that X and y are not aligned (y and x with the same index corresponds to
            # the same time step).
            test_split_base = test_split + np.arange(len(test_split)) * self.n_prediction_steps + 1

            y_test_split = np.repeat(test_split_base, self.n_prediction_steps) + \
                           np.tile(np.arange(self.n_prediction_steps), len(test_split_base))

            self.Y_optimization = self.y_train[y_test_split]

            #self.Y_actual_train = self.y_train[train_split]
            y_train_pred, y_opt_pred, y_valid_pred, y_test_pred = self._fit_and_predict(pipeline, split_id,
                                                                                        train_indices=train_split,
                                                                                        test_indices=test_split,
                                                                                        add_pipeline_to_self=True)

            # use for computing MASE losses
            y_train_forecasting = []
            for seq_idx, test_idx in enumerate(test_split):
                # TODO consider multi-variants here
                seq = self.datamanager[test_idx][0]
                if seq.shape[-1] > 1:
                     y_train_forecasting.append(seq[self.datamanager.target_variables].squeeze())
                else:
                    y_train_forecasting.append(seq.squeeze())

            forecasting_kwargs = {'y_train': y_train_forecasting,
                                  'sp': self.seasonality,
                                  'n_prediction_steps': self.n_prediction_steps,
                                  }

            train_loss = None
            loss = self._loss(self.y_train[y_test_split], y_opt_pred, **forecasting_kwargs)

            additional_run_info = pipeline.get_additional_run_info() if hasattr(
                pipeline, 'get_additional_run_info') else {}

            status = StatusType.SUCCESS

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

            y_train_forecasting_all = []
            for i, (train_split, test_split) in self.splits:
                # use for computing MASE losses
                y_train_forecasting = []
                for seq_idx, test_idx in enumerate(test_split):
                    # TODO consider multi-variants here
                    seq = self.datamanager[test_idx][0]
                    if seq.shape[-1] > 1:
                        y_train_forecasting.append(seq[self.datamanager.target_variables].squeeze())
                    else:
                        y_train_forecasting.append(seq.squeeze())
                y_train_forecasting_all.append(y_train_forecasting)

            for i, (train_split, test_split) in enumerate(self.splits):
                pipeline = self.pipelines[i]
                test_split_base = test_split + np.arange(len(test_split)) * self.n_prediction_steps + 1

                train_pred, opt_pred, valid_pred, test_pred = self._fit_and_predict(pipeline, i,
                                                                                    train_indices=train_split,
                                                                                    test_indices=test_split,
                                                                                    add_pipeline_to_self=False)
                #Y_train_pred[i] = train_pred
                Y_optimization_pred[i] = opt_pred
                Y_valid_pred[i] = valid_pred
                Y_test_pred[i] = test_pred
                train_splits[i] = train_split

                #self.Y_train_targets[train_split] = self.y_train[train_split]

                y_test_split = np.repeat(test_split_base, self.n_prediction_steps) + \
                               np.tile(np.arange(self.n_prediction_steps), len(test_split_base))

                self.Y_targets[i] = self.y_train[y_test_split]
                # Compute train loss of this fold and store it. train_loss could
                # either be a scalar or a dict of scalars with metrics as keys.

                train_loss = 0.
                train_losses[i] = train_loss
                # number of training data points for this fold. Used for weighting
                # the average.
                train_fold_weights[i] = len(train_split)

                forecasting_kwargs = {'y_train': y_train_forecasting_all[i],
                                      'sp': self.seasonality,
                                      'n_prediction_steps': self.n_prediction_steps,
                                      }

                # Compute validation loss of this fold and store it.
                optimization_loss = self._loss(
                    self.Y_targets[i],
                    opt_pred,
                    **forecasting_kwargs
                )
                opt_losses[i] = optimization_loss
                # number of optimization data points for this fold.
                # Used for weighting the average.
                opt_fold_weights[i] = len(train_split)

            # Compute weights of each fold based on the number of samples in each
            # fold.
            train_fold_weights = [w / sum(train_fold_weights)
                                  for w in train_fold_weights]
            opt_fold_weights = [w / sum(opt_fold_weights)
                                for w in opt_fold_weights]

            # train_losses is a list of dicts. It is
            # computed using the target metric (self.metric).
            train_loss = np.average([train_losses[i][str(self.metric)]
                                     for i in range(self.num_folds)],
                                    weights=train_fold_weights,
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
            self.logger.debug("In train evaluator fit_predict_and_loss, loss:{}".format(opt_loss))
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

    def _predict(self, pipeline: BaseEstimator,
                 train_indices: Union[np.ndarray, List],
                 test_indices: Union[np.ndarray, List],
                 ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
        # TODO consider multile outputs
        opt_pred = np.ones([len(test_indices), self.n_prediction_steps])
        for seq_idx, test_idx in enumerate(test_indices):
            opt_pred[seq_idx] = self.predict_function(self.datamanager[test_idx][0], pipeline).squeeze()

        #TODO we consider X_valid and X_test as a multiple sequences???
        if self.X_valid is not None:
            valid_pred = np.ones([len(test_indices), self.n_prediction_steps])
            for seq_idx, val_seq in enumerate(self.datamanager.datasets):
                valid_pred[seq_idx] = self.predict_function(val_seq.X, pipeline).flatten()

            valid_pred = valid_pred.flatten()

        else:
            valid_pred = None

        if self.X_test is not None:
            test_pred = np.ones([len(test_indices), self.n_prediction_steps])
            for seq_idx, test_seq in enumerate(self.datamanager.datasets):
                test_pred[seq_idx] = self.predict_function(test_seq.X, pipeline)

            test_pred = test_pred.flatten()
        else:
            test_pred = None

        return np.empty(1), opt_pred, valid_pred, test_pred

