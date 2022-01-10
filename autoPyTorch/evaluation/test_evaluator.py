# -*- encoding: utf-8 -*-
from multiprocessing.queues import Queue
from typing import Any, Dict, List, Optional, Tuple, Union

from ConfigSpace.configuration_space import Configuration

import numpy as np

from sklearn.base import BaseEstimator

from smac.tae import StatusType

from autoPyTorch.automl_common.common.utils.backend import Backend
from autoPyTorch.evaluation.abstract_evaluator import (
    AbstractEvaluator,
    fit_and_suppress_warnings
)
from autoPyTorch.evaluation.utils import DisableFileOutputParameters
from autoPyTorch.pipeline.components.base_component import find_components
from autoPyTorch.pipeline.components.training.metrics.base import autoPyTorchMetric
from autoPyTorch.utils.common import dict_repr, subsampler
from autoPyTorch.utils.hyperparameter_search_space_update import HyperparameterSearchSpaceUpdates
from test.test_evaluation import test_abstract_evaluator


__all__ = [
    'eval_test_function',
    'TestEvaluator'
]


class TestEvaluator(AbstractEvaluator):

    def __init__(
        self, backend: Backend, queue: Queue,
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
                 disable_file_output: Optional[List[Union[str, DisableFileOutputParameters]]] = None,
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

        self.pipeline = self._get_pipeline()

    def fit_predict_and_loss(self) -> None:
        fit_and_suppress_warnings(self.logger, self.pipeline, self.X_train, self.y_train)
        train_loss, _ = self.predict_and_loss(train=True)
        test_loss, test_pred = self.predict_and_loss()
        self.finish_up(
            loss=test_loss,
            train_loss=train_loss,
            opt_pred=test_pred,
            valid_pred=None,
            test_pred=None,
            file_output=False,
            additional_run_info=None,
            status=StatusType.SUCCESS,
        )

    def predict_and_loss(
        self, train: bool = False
    ) -> Tuple[Dict[str, float], np.ndarray]:

        if train:
            y_pred = self.predict_function(
                self.X_train,
                self.pipeline,
                self.y_train
                )
            err = self._loss(self.y_train, y_pred)
        else:
            y_pred = self.predict_function(
                self.X_test,
                self.pipeline,
                self.y_train
                )
            err = self._loss(self.y_test, y_pred)

        return err, y_pred


# create closure for evaluating an algorithm
# Has a stupid name so pytest doesn't regard it as a test
def eval_test_function(
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
    pipeline_config: Optional[Dict[str, Any]] = None,
    budget_type: str = None,
    init_params: Optional[Dict[str, Any]] = None,
    logger_port: Optional[int] = None,
    all_supported_metrics: bool = True,
    search_space_updates: Optional[HyperparameterSearchSpaceUpdates] = None,
    instance: str = None,
) -> None:
    evaluator = TestEvaluator(backend=backend,
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
        search_space_updates=search_space_updates)

    evaluator.fit_predict_and_loss()