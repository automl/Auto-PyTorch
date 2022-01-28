from multiprocessing.queues import Queue
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from sklearn.base import BaseEstimator

from smac.tae import StatusType

from autoPyTorch.datasets.resampling_strategy import (
    CrossValTypes,
    NoResamplingStrategyTypes,
    check_resampling_strategy
)
from autoPyTorch.evaluation.abstract_evaluator import (
    AbstractEvaluator,
    EvaluationResults,
    fit_pipeline
)
from autoPyTorch.evaluation.abstract_evaluator import EvaluatorParams, FixedPipelineParams
from autoPyTorch.utils.common import dict_repr, subsampler

__all__ = ['Evaluator', 'eval_fn']


class _CrossValidationResultsManager:
    def __init__(self, num_folds: int):
        self.additional_run_info: Dict = {}
        self.opt_preds: List[Optional[np.ndarray]] = [None] * num_folds
        self.valid_preds: List[Optional[np.ndarray]] = [None] * num_folds
        self.test_preds: List[Optional[np.ndarray]] = [None] * num_folds
        self.train_loss: Dict[str, float] = {}
        self.opt_loss: Dict[str, float] = {}
        self.n_train, self.n_opt = 0, 0

    @staticmethod
    def _update_loss_dict(loss_sum_dict: Dict[str, float], loss_dict: Dict[str, float], n_datapoints: int) -> None:
        loss_sum_dict.update({
            metric_name: loss_sum_dict.get(metric_name, 0) + loss_dict[metric_name] * n_datapoints
            for metric_name in loss_dict.keys()
        })

    def update(self, split_id: int, results: EvaluationResults, n_train: int, n_opt: int) -> None:
        self.n_train += n_train
        self.n_opt += n_opt
        self.opt_preds[split_id] = results.opt_pred
        self.valid_preds[split_id] = results.valid_pred
        self.test_preds[split_id] = results.test_pred

        if results.additional_run_info is not None:
            self.additional_run_info.update(results.additional_run_info)

        self._update_loss_dict(self.train_loss, loss_dict=results.train_loss, n_datapoints=n_train)
        self._update_loss_dict(self.opt_loss, loss_dict=results.opt_loss, n_datapoints=n_opt)

    def get_average_loss(self) -> Tuple[Dict[str, float], Dict[str, float]]:
        train_avg_loss = {metric_name: val / float(self.n_train) for metric_name, val in self.train_loss.items()}
        opt_avg_loss = {metric_name: val / float(self.n_opt) for metric_name, val in self.opt_loss.items()}
        return train_avg_loss, opt_avg_loss

    def _merge_predictions(self, preds: List[Optional[np.ndarray]]) -> Optional[np.ndarray]:
        merged_pred = np.array([pred for pred in preds if pred is not None])
        if merged_pred.size == 0:
            return None

        if len(merged_pred.shape) != 3:
            # merged_pred.shape := (n_splits, n_datapoints, n_class or 1)
            raise ValueError(
                f'each pred must have the shape (n_datapoints, n_class or 1), but got {merged_pred.shape[1:]}'
            )

        return np.nanmean(merged_pred, axis=0)

    def get_result_dict(self) -> Dict[str, Any]:
        train_loss, opt_loss = self.get_average_loss()
        return dict(
            opt_loss=opt_loss,
            train_loss=train_loss,
            opt_pred=np.concatenate([pred for pred in self.opt_preds if pred is not None]),
            valid_pred=self._merge_predictions(self.valid_preds),
            test_pred=self._merge_predictions(self.test_preds),
            additional_run_info=self.additional_run_info
        )


class Evaluator(AbstractEvaluator):
    """
    This class builds a pipeline using the provided configuration.
    A pipeline implementing the provided configuration is fitted
    using the datamanager object retrieved from disc, via the backend.
    After the pipeline is fitted, it is save to disc and the performance estimate
    is communicated to the main process via a Queue.

    Args:
        queue (Queue):
            Each worker available will instantiate an evaluator, and after completion,
            it will append the result to a multiprocessing queue
        fixed_pipeline_params (FixedPipelineParams):
            Fixed parameters for a pipeline
        evaluator_params (EvaluatorParams):
            The parameters for an evaluator.

    Attributes:
        train (bool):
            Whether the training data is split and the validation set is used for SMBO optimisation.
        cross_validation (bool):
            Whether we use cross validation or not.
    """
    def __init__(self, queue: Queue, fixed_pipeline_params: FixedPipelineParams, evaluator_params: EvaluatorParams):
        resampling_strategy = fixed_pipeline_params.backend.load_datamanager().resampling_strategy
        self.train = not isinstance(resampling_strategy, NoResamplingStrategyTypes)
        self.cross_validation = isinstance(resampling_strategy, CrossValTypes)

        if not self.train and fixed_pipeline_params.save_y_opt:
            # TODO: Add the test to cover here
            # No resampling can not be used for building ensembles. save_y_opt=False ensures it
            fixed_pipeline_params = fixed_pipeline_params._replace(save_y_opt=False)

        super().__init__(queue=queue, fixed_pipeline_params=fixed_pipeline_params, evaluator_params=evaluator_params)

        if self.train:
            self.logger.debug("Search space updates :{}".format(self.fixed_pipeline_params.search_space_updates))

    def _evaluate_on_split(self, split_id: int) -> EvaluationResults:
        """
        Fit on the training split in the i-th split and evaluate on
        the holdout split (i.e. opt_split) in the i-th split.

        Args:
            split_id (int):
                Which split to take.

        Returns:
            results (EvaluationResults):
                The results from the training and validation.
        """
        self.logger.info("Starting fit {}".format(split_id))
        # We create pipeline everytime to avoid non-fitted pipelines to be in self.pipelines
        pipeline = self._get_pipeline()

        train_split, opt_split = self.splits[split_id]
        train_pred, opt_pred, valid_pred, test_pred = self._fit_and_evaluate_loss(
            pipeline,
            split_id,
            train_indices=train_split,
            opt_indices=opt_split
        )

        return EvaluationResults(
            pipeline=pipeline,
            opt_loss=self._loss(labels=self.y_train[opt_split] if self.train else self.y_test, preds=opt_pred),
            train_loss=self._loss(labels=self.y_train[train_split], preds=train_pred),
            opt_pred=opt_pred,
            valid_pred=valid_pred,
            test_pred=test_pred,
            status=StatusType.SUCCESS,
            additional_run_info=getattr(pipeline, 'get_additional_run_info', lambda: {})()
        )

    def _cross_validation(self) -> EvaluationResults:
        """
        Perform cross validation and return the merged results.

        Returns:
            results (EvaluationResults):
                The results that merge every split.
        """
        cv_results = _CrossValidationResultsManager(self.num_folds)
        Y_opt: List[Optional[np.ndarray]] = [None] * self.num_folds

        for split_id in range(len(self.splits)):
            train_split, opt_split = self.splits[split_id]
            Y_opt[split_id] = self.y_train[opt_split]
            results = self._evaluate_on_split(split_id)

            self.pipelines[split_id] = results.pipeline
            assert opt_split is not None  # mypy redefinition
            cv_results.update(split_id, results, len(train_split), len(opt_split))

        self.y_opt = np.concatenate([y_opt for y_opt in Y_opt if y_opt is not None])

        return EvaluationResults(status=StatusType.SUCCESS, **cv_results.get_result_dict())

    def evaluate_loss(self) -> None:
        """Fit, predict and compute the loss for cross-validation and holdout"""
        if self.splits is None:
            raise ValueError(f"cannot fit pipeline {self.__class__.__name__} with datamanager.splits None")

        if self.cross_validation:
            results = self._cross_validation()
        else:
            _, opt_split = self.splits[0]
            results = self._evaluate_on_split(split_id=0)
            self.pipelines[0] = results.pipeline
            self.y_opt = self.y_train[opt_split] if self.train else self.y_test

        self.logger.debug(
            f"In evaluate_loss, num_run: {self.num_run}, loss:{results.opt_loss},"
            f" status: {results.status},\nadditional run info:\n{dict_repr(results.additional_run_info)}"
        )
        self.record_evaluation(results=results)

    def _fit_and_evaluate_loss(
        self,
        pipeline: BaseEstimator,
        split_id: int,
        train_indices: Union[np.ndarray, List],
        opt_indices: Union[np.ndarray, List]
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:

        X = dict(train_indices=train_indices, val_indices=opt_indices, split_id=split_id, num_run=self.num_run)
        X.update(self.fit_dictionary)
        fit_pipeline(self.logger, pipeline, X, y=None)
        self.logger.info("Model fitted, now predicting")

        kwargs = {'pipeline': pipeline, 'unique_train_labels': self.unique_train_labels[split_id]}
        train_pred = self.predict(subsampler(self.X_train, train_indices), **kwargs)
        test_pred = self.predict(self.X_test, **kwargs)
        valid_pred = self.predict(self.X_valid, **kwargs)

        # No resampling ===> evaluate on test dataset
        opt_pred = self.predict(subsampler(self.X_train, opt_indices), **kwargs) if self.train else test_pred

        assert train_pred is not None and opt_pred is not None  # mypy check
        return train_pred, opt_pred, valid_pred, test_pred


def eval_fn(queue: Queue, fixed_pipeline_params: FixedPipelineParams, evaluator_params: EvaluatorParams) -> None:
    """
    This closure allows the communication between the TargetAlgorithmQuery and the
    pipeline trainer (Evaluator).

    Fundamentally, smac calls the TargetAlgorithmQuery.run() method, which internally
    builds an Evaluator. The Evaluator builds a pipeline, stores the output files
    to disc via the backend, and puts the performance result of the run in the queue.

    Args:
        queue (Queue):
            Each worker available will instantiate an evaluator, and after completion,
            it will append the result to a multiprocessing queue
        fixed_pipeline_params (FixedPipelineParams):
            Fixed parameters for a pipeline
        evaluator_params (EvaluatorParams):
            The parameters for an evaluator.
    """
    resampling_strategy = fixed_pipeline_params.backend.load_datamanager().resampling_strategy
    check_resampling_strategy(resampling_strategy)

    # NoResamplingStrategyTypes ==> test evaluator, otherwise ==> train evaluator
    evaluator = Evaluator(
        queue=queue,
        evaluator_params=evaluator_params,
        fixed_pipeline_params=fixed_pipeline_params
    )
    evaluator.evaluate_loss()
