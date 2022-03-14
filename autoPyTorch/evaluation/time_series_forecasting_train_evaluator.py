from autoPyTorch.evaluation.train_evaluator import TrainEvaluator
from autoPyTorch.evaluation.abstract_evaluator import DummyClassificationPipeline

from multiprocessing.queues import Queue
from typing import Any, Dict, List, Optional, Tuple, Union, no_type_check, ClassVar, Sequence
from functools import partial
import warnings

from ConfigSpace.configuration_space import Configuration

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator

from smac.tae import StatusType

from autoPyTorch.pipeline.components.training.metrics.base import autoPyTorchMetric
from autoPyTorch.pipeline.components.training.metrics.metrics import MASE_LOSSES
from autoPyTorch.automl_common.common.utils.backend import Backend
from autoPyTorch.utils.common import subsampler
from autoPyTorch.evaluation.abstract_evaluator import DummyTimeSeriesForecastingPipeline
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
                 search_space_updates: Optional[HyperparameterSearchSpaceUpdates] = None,
                 max_budget: float = 1.0,
                 min_num_test_instances: Optional[int] = None) -> None:
        """
        Attributes:
            max_budget (Optional[float]):
                maximal budget the optimizer could allocate
            min_num_test_instances: Optional[int]
                minimal number of validation instances to be evaluated, if the size of the validation set is greater
                than this value, then less instances from validation sets will be evaluated. The other predictions
                 will be filled with dummy predictor

        """
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
        self.num_targets = self.datamanager.num_target
        self.seq_length_min = np.min(self.num_sequences)
        seasonality = SEASONALITY_MAP.get(self.datamanager.freq, 1)
        if isinstance(seasonality, list):
            seasonality = min(seasonality)  # Use to calculate MASE
        self.seasonality = int(seasonality)

        self.max_budget = max_budget
        self.min_num_test_instances = min_num_test_instances

        import os
        os.system("sh -c \"scontrol -d show job $SLURM_JOB_ID\"")
        os.system("nvidia-smi.user")


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

            self.Y_optimization = self.datamanager.get_test_target(test_split)


            # self.Y_actual_train = self.y_train[train_split]
            y_train_pred, y_opt_pred, y_valid_pred, y_test_pred = self._fit_and_predict(pipeline, split_id,
                                                                                        train_indices=train_split,
                                                                                        test_indices=test_split,
                                                                                        add_pipeline_to_self=True)

            mase_cofficient = self.generate_mase_coefficient_for_validation(test_split)

            forecasting_kwargs = {'sp': self.seasonality,
                                  'n_prediction_steps': self.n_prediction_steps,
                                  'mase_cofficient': mase_cofficient,
                                  }

            train_loss = None
            loss = self._loss(self.Y_optimization, y_opt_pred, **forecasting_kwargs)

            additional_run_info = pipeline.get_additional_run_info() if hasattr(
                pipeline, 'get_additional_run_info') else {}

            status = StatusType.SUCCESS

            self.Y_optimization *= mase_cofficient

            self.finish_up(
                loss=loss,
                train_loss=train_loss,
                opt_pred=y_opt_pred * mase_cofficient,
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

            mase_coefficient_all = []
            for train_split, test_split in self.splits:
                mase_coefficient = self.generate_mase_coefficient_for_validation(test_split)
                mase_coefficient_all.append(mase_coefficient)

            for i, (train_split, test_split) in enumerate(self.splits):
                pipeline = self.pipelines[i]

                train_pred, opt_pred, valid_pred, test_pred = self._fit_and_predict(pipeline, i,
                                                                                    train_indices=train_split,
                                                                                    test_indices=test_split,
                                                                                    add_pipeline_to_self=False)
                # Y_train_pred[i] = train_pred
                Y_optimization_pred[i] = opt_pred
                Y_valid_pred[i] = valid_pred
                Y_test_pred[i] = test_pred
                train_splits[i] = train_split

                self.Y_targets[i] = self.datamanager.get_test_target(test_split)
                # Compute train loss of this fold and store it. train_loss could
                # either be a scalar or a dict of scalars with metrics as keys.

                # number of training data points for this fold. Used for weighting
                # the average.
                train_fold_weights[i] = len(train_split)

                forecasting_kwargs = {'mase_cofficient': mase_coefficient_all[i],
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

            opt_fold_weights = [w / sum(opt_fold_weights)
                                for w in opt_fold_weights]

            train_loss = None

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
                [Y_optimization_pred[i] * mase_coefficient_all[i] for i in range(self.num_folds)
                 if Y_optimization_pred[i] is not None])
            Y_targets = np.concatenate([
                Y_targets[i] * mase_coefficient_all[i] for i in range(self.num_folds)
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
                opt_pred=Y_optimization_preds.flatten(),
                valid_pred=Y_valid_preds,
                test_pred=Y_test_preds,
                additional_run_info=additional_run_info,
                file_output=True,
                status=status,
            )

    def generate_mase_coefficient_for_validation(self, test_split: Sequence) -> np.ndarray:
        """
        Compute the denominator for Mean Absolute Scaled Losses,
        For detail, please check sktime.performance_metrics.forecasting._functions.mean_absolute_scaled_error

        Parameters:
        ----------
        test_split: Sequence
            test splits, consistent of int
        Return:
        ----------
        mase_coefficient: np.ndarray(self.num_sequence * self.n_prediction_steps)
            inverse of the mase_denominator
        """
        mase_coefficient = np.ones([len(test_split), self.num_targets])
        if any(mase_loss in self.additional_metrics for mase_loss in MASE_LOSSES) or self.metric in MASE_LOSSES:
            for seq_idx, test_idx in enumerate(test_split):
                mase_coefficient[seq_idx] = self.datamanager.get_time_series_seq(test_idx).mase_coefficient

        mase_coefficient = np.repeat(mase_coefficient, self.n_prediction_steps, axis=0)
        return mase_coefficient

    def create_validation_sub_set(self, test_indices: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        num_test_instances = len(test_indices)

        if num_test_instances < self.min_num_test_instances or self.budget >= self.max_budget:
            # if the length of test indices is smaller than the
            return test_indices, None
        num_val_instance = min(num_test_instances,
                               max(self.min_num_test_instances,
                                   int(num_test_instances * self.budget / self.max_budget)
                                   ))
        test_subset_indices = np.linspace(0, num_test_instances, num_val_instance, endpoint=False, dtype=np.int)
        return test_indices[test_subset_indices], test_subset_indices

    def _predict(self, pipeline: BaseEstimator,
                 train_indices: Union[np.ndarray, List],
                 test_indices: Union[np.ndarray, List],
                 ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:

        if self.min_num_test_instances is not None:
            test_indices_subset, test_split_subset_idx = self.create_validation_sub_set(test_indices)
        else:
            test_indices_subset = test_indices
            test_split_subset_idx = None

        val_sets = []

        for test_idx in test_indices_subset:
            val_sets.append(self.datamanager.get_validation_set(test_idx))
        opt_pred = self.predict_function(val_sets, pipeline)
        opt_pred = opt_pred.reshape(-1, self.num_targets)

        if test_split_subset_idx is not None:
            dummy_pipeline = DummyTimeSeriesForecastingPipeline(0, n_prediction_steps=self.n_prediction_steps)
            remaining_indices = np.setdiff1d(np.arange(len(test_indices)), test_indices_subset)
            val_set_remain = []
            for remaining_idx in remaining_indices:
                val_set_remain.append(self.datamanager.get_validation_set(test_indices[remaining_idx]))
            y_opt_full = np.zeros([len(test_indices), self.n_prediction_steps, self.num_targets])
            y_opt_full[remaining_indices] = dummy_pipeline.predict(val_set_remain).reshape([-1,
                                                                                            self.n_prediction_steps,
                                                                                            self.num_targets])
            y_opt_full[test_split_subset_idx] = opt_pred.reshape([-1, self.n_prediction_steps, self.num_targets])
            opt_pred = y_opt_full

        opt_pred = opt_pred.reshape(-1, self.num_targets)

        # TODO we consider X_valid and X_test as a multiple sequences???
        if self.X_valid is not None:
            valid_sets = []
            for val_seq in enumerate(self.datamanager.datasets):
                valid_sets.append(val_seq.X_val)
            valid_pred = self.predict_function(valid_sets, pipeline).flatten()

            valid_pred = valid_pred.reshape(-1, self.num_targets)

        else:
            valid_pred = None

        if self.X_test is not None:
            test_sets = []
            for test_seq in enumerate(self.datamanager.datasets):
                test_sets.append(test_seq.X_test)
            test_pred = self.predict_function(valid_sets, pipeline).flatten()
            test_pred = test_pred.reshape(-1, self.num_targets)
        else:
            test_pred = None

        return np.empty(1), opt_pred, valid_pred, test_pred
