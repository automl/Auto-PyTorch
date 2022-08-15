import copy
import warnings
from multiprocessing.queues import Queue
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from ConfigSpace.configuration_space import Configuration

import numpy as np

from sklearn.base import BaseEstimator

from smac.tae import StatusType

from autoPyTorch.automl_common.common.utils.backend import Backend
from autoPyTorch.constants import SEASONALITY_MAP
from autoPyTorch.evaluation.train_evaluator import TrainEvaluator
from autoPyTorch.evaluation.utils import DisableFileOutputParameters
from autoPyTorch.evaluation.utils_extra import DummyTimeSeriesForecastingPipeline
from autoPyTorch.pipeline.components.training.metrics.base import autoPyTorchMetric
from autoPyTorch.pipeline.components.training.metrics.metrics import MASE_LOSSES
from autoPyTorch.utils.hyperparameter_search_space_update import HyperparameterSearchSpaceUpdates


class TimeSeriesForecastingTrainEvaluator(TrainEvaluator):
    """
    This class is  similar to the TrainEvaluator. Except that given the specific

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
        max_budget (float):
            maximal budget value available for the optimizer. This is applied to compute the size of the proxy
            validation sets
        min_num_test_instances (Optional[int]):
            minimal number of instances to be validated. We do so to ensure that there are enough instances in
            the validation set

    """
    def __init__(self, backend: Backend, queue: Queue,
                 metric: autoPyTorchMetric,
                 budget: float,
                 budget_type: str = None,
                 pipeline_options: Optional[Dict[str, Any]] = None,
                 configuration: Optional[Configuration] = None,
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
                 search_space_updates: Optional[HyperparameterSearchSpaceUpdates] = None,
                 max_budget: float = 1.0,
                 min_num_test_instances: Optional[int] = None) -> None:
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
            pipeline_options=pipeline_options,
            search_space_updates=search_space_updates
        )
        self.datamanager = backend.load_datamanager()
        self.n_prediction_steps = self.datamanager.n_prediction_steps
        self.num_sequences = self.datamanager.num_sequences
        self.num_targets = self.datamanager.num_targets
        self.seq_length_min = np.min(self.num_sequences)
        seasonality = SEASONALITY_MAP.get(self.datamanager.freq, 1)
        if isinstance(seasonality, list):
            seasonality = min(seasonality)  # Use to calculate MASE
        self.seasonality = int(seasonality)  # type: ignore[call-overload]

        self.max_budget = max_budget
        self.min_num_test_instances = min_num_test_instances
        self.eval_test_tensors = True

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

            mase_coefficient_val = self.generate_mase_coefficient_for_validation(test_split)  # type: ignore[arg-type]

            forecasting_kwargs = {'sp': self.seasonality,
                                  'n_prediction_steps': self.n_prediction_steps,
                                  }
            forecasting_kwargs_val = copy.copy(forecasting_kwargs)
            forecasting_kwargs_val['mase_coefficient'] = mase_coefficient_val
            if self.y_test is not None:
                mase_coefficient_test = self.generate_mase_coefficient_for_test_set()
                forecasting_kwargs['mase_coefficient'] = mase_coefficient_test

            train_loss = None

            loss = self._loss(self.Y_optimization, y_opt_pred, **forecasting_kwargs_val)  # type: ignore[arg-type]

            additional_run_info = pipeline.get_additional_run_info() if hasattr(
                pipeline, 'get_additional_run_info') else {}

            status = StatusType.SUCCESS
            # self.Y_optimization and y_opt_pred need to be applied to construct ensembles. We simply scale them here
            self.Y_optimization *= mase_coefficient_val

            self.finish_up(
                loss=loss,
                train_loss=train_loss,  # type: ignore[arg-type]
                opt_pred=y_opt_pred * mase_coefficient_val,
                valid_pred=y_valid_pred,
                test_pred=y_test_pred,
                additional_run_info=additional_run_info,
                file_output=True,
                status=status,
                **forecasting_kwargs
            )

        else:
            Y_optimization_pred: List[Optional[np.ndarray]] = [None] * self.num_folds
            Y_valid_pred: List[Optional[np.ndarray]] = [None] * self.num_folds
            Y_test_pred: List[Optional[np.ndarray]] = [None] * self.num_folds
            train_splits: List[Optional[Union[np.ndarray, List]]] = [None] * self.num_folds

            self.pipelines = [self._get_pipeline() for _ in range(self.num_folds)]

            # Train losses is not applied here as it might become too expensive

            # used as weights when averaging train losses.
            train_fold_weights = [np.NaN] * self.num_folds
            # stores opt (validation) loss of each fold.
            opt_losses = [np.NaN] * self.num_folds
            # weights for opt_losses.
            opt_fold_weights = [np.NaN] * self.num_folds

            mase_coefficient_val_all = []
            for train_split, test_split in self.splits:
                mase_coefficient = self.generate_mase_coefficient_for_validation(test_split)  # type: ignore[arg-type]
                mase_coefficient_val_all.append(mase_coefficient)

            forecasting_kwargs = {'sp': self.seasonality,
                                  'n_prediction_steps': self.n_prediction_steps}

            if self.y_test is not None:
                mase_coefficient_test = self.generate_mase_coefficient_for_test_set()
                forecasting_kwargs['mase_coefficient'] = mase_coefficient_test

            for i, (train_split, test_split) in enumerate(self.splits):
                if i > 0:
                    self.eval_test_tensors = False
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

                forecasting_kwargs_val = copy.copy(forecasting_kwargs)
                forecasting_kwargs_val['mase_coefficient'] = mase_coefficient_val_all[i]

                # Compute validation loss of this fold and store it.
                optimization_loss = self._loss(
                    self.Y_targets[i],  # type: ignore[arg-type]
                    opt_pred,
                    **forecasting_kwargs_val
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
                [Y_optimization_pred[i] * mase_coefficient_val_all[i] for i in range(self.num_folds)
                 if Y_optimization_pred[i] is not None])
            Y_targets = np.concatenate([
                Y_targets[i] * mase_coefficient_val_all[i] for i in range(self.num_folds)
                if Y_targets[i] is not None
            ])

            if self.y_valid is not None:
                warnings.warn('valid_pred is currently unsupported for fore casting tasks!')
            Y_valid_preds = None

            if self.y_test is not None:
                Y_test_preds = np.array([Y_test_pred[i] * mase_coefficient_val_all[0]
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
                train_loss=train_loss,  # type: ignore[arg-type]
                opt_pred=Y_optimization_preds.flatten(),
                valid_pred=Y_valid_preds,
                test_pred=Y_test_preds,
                additional_run_info=additional_run_info,
                file_output=True,
                status=status,
                **forecasting_kwargs,
            )

    def generate_mase_coefficient_for_validation(self, test_split: Sequence[int]) -> np.ndarray:
        """
        Compute the denominator for Mean Absolute Scaled Losses,
        For detail, please check sktime.performance_metrics.forecasting._functions.mean_absolute_scaled_error

        Parameters:
        ----------
        test_split (Sequence):
            test splits, consistent of int
        Return:
        ----------
        mase_coefficient (np.ndarray(self.num_sequence * self.n_prediction_steps)):
            inverse of the mase_denominator
        """
        mase_coefficient = np.ones([len(test_split), self.num_targets])
        if self.additional_metrics is not None:
            if any(mase_loss in self.additional_metrics for mase_loss in MASE_LOSSES) or self.metric in MASE_LOSSES:
                for seq_idx, test_idx in enumerate(test_split):
                    mase_coefficient[seq_idx] = self.datamanager.get_time_series_seq(test_idx).mase_coefficient

        mase_coefficient = np.repeat(mase_coefficient, self.n_prediction_steps, axis=0)
        return mase_coefficient

    def generate_mase_coefficient_for_test_set(self) -> np.ndarray:
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
        mase_coefficient = np.ones([len(self.datamanager.datasets), self.num_targets])
        if self.additional_metrics is not None:
            if any(mase_loss in self.additional_metrics for mase_loss in MASE_LOSSES) or self.metric in MASE_LOSSES:
                for seq_idx, test_idx in enumerate(self.datamanager.datasets):
                    mase_coefficient[seq_idx] = self.datamanager.datasets[seq_idx].mase_coefficient
        mase_coefficient = np.repeat(mase_coefficient, self.n_prediction_steps, axis=0)
        return mase_coefficient

    def create_validation_sub_set(self, test_indices: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        if self.min_num_test_instances is not None:
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
        else:
            return test_indices, None

    def _predict(self, pipeline: BaseEstimator,
                 test_indices: Union[np.ndarray, List],
                 train_indices: Union[np.ndarray, List],
                 ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
        test_indices_subset, test_split_subset_idx = self.create_validation_sub_set(test_indices)

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

        if self.y_valid is not None:
            warnings.warn('valid_pred is current unsupported for forecasting tasks!')
        valid_pred = None

        if self.y_test is not None and self.eval_test_tensors:
            test_seq = self.datamanager.generate_test_seqs()
            test_pred = self.predict_function(test_seq, pipeline).reshape(-1, self.num_targets)
        else:
            test_pred = None

        return np.empty(1), opt_pred, valid_pred, test_pred


# create closure for evaluating an algorithm
def forecasting_eval_train_function(
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
    max_budget: float = 1.0,
    min_num_test_instances: Optional[int] = None
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
        max_budget (float):
            maximal budget value available for the optimizer. This is applied to compute the size of the proxy
            validation sets
        min_num_test_instances (Optional[int]):
            minimal number of instances to be validated. We do so to ensure that there are enough instances in
            the validation set
    """
    evaluator = TimeSeriesForecastingTrainEvaluator(
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
        max_budget=max_budget,
        min_num_test_instances=min_num_test_instances,
    )
    evaluator.fit_predict_and_loss()
