from multiprocessing.queues import Queue
from typing import Any, Dict, List, Optional, Tuple, Union

from ConfigSpace.configuration_space import Configuration

import numpy as np

from smac.tae import StatusType

from autoPyTorch.automl_common.common.utils.backend import Backend
from autoPyTorch.datasets.resampling_strategy import NoResamplingStrategyTypes
from autoPyTorch.evaluation.abstract_evaluator import (
    AbstractEvaluator,
    fit_and_suppress_warnings
)
from autoPyTorch.evaluation.utils import DisableFileOutputParameters
from autoPyTorch.pipeline.components.training.metrics.base import autoPyTorchMetric
from autoPyTorch.utils.hyperparameter_search_space_update import HyperparameterSearchSpaceUpdates


__all__ = [
    'eval_test_function',
    'TestEvaluator'
]


class TestEvaluator(AbstractEvaluator):
    """
    This class builds a pipeline using the provided configuration.
    A pipeline implementing the provided configuration is fitted
    using the datamanager object retrieved from disc, via the backend.
    After the pipeline is fitted, it is save to disc and the performance estimate
    is communicated to the main process via a Queue. It is only compatible
    with `NoResamplingStrategyTypes`, i.e, when the training data
    is not split and the test set is used for SMBO optimisation. It can not
    be used for building ensembles which is ensured by having
    `output_y_hat_optimisation`=False

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
    def __init__(
        self,
        backend: Backend, queue: Queue,
        metric: autoPyTorchMetric,
        budget: float,
        configuration: Union[int, str, Configuration],
        budget_type: str = None,
        pipeline_options: Optional[Dict[str, Any]] = None,
        seed: int = 1,
        output_y_hat_optimization: bool = False,
        num_run: Optional[int] = None,
        include: Optional[Dict[str, Any]] = None,
        exclude: Optional[Dict[str, Any]] = None,
        disable_file_output: Optional[List[Union[str, DisableFileOutputParameters]]] = None,
        init_params: Optional[Dict[str, Any]] = None,
        logger_port: Optional[int] = None,
        all_supported_metrics: bool = True,
        search_space_updates: Optional[HyperparameterSearchSpaceUpdates] = None
    ) -> None:
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

        if not isinstance(self.resampling_strategy, (NoResamplingStrategyTypes)):
            raise ValueError(
                f'resampling_strategy for TestEvaluator must be in '
                f'NoResamplingStrategyTypes, but got {self.resampling_strategy}'
            )

    def fit_predict_and_loss(self) -> None:

        split_id = 0
        train_indices, test_indices = self.splits[split_id]

        self.pipeline = self._get_pipeline()
        X = {'train_indices': train_indices,
             'val_indices': test_indices,
             'split_id': split_id,
             'num_run': self.num_run,
             **self.fit_dictionary}  # fit dictionary
        y = None
        fit_and_suppress_warnings(self.logger, self.pipeline, X, y)
        train_loss, _ = self.predict_and_loss(train=True)
        test_loss, test_pred = self.predict_and_loss()
        self.Y_optimization = self.y_test
        self.finish_up(
            loss=test_loss,
            train_loss=train_loss,
            opt_pred=test_pred,
            valid_pred=None,
            test_pred=test_pred,
            file_output=True,
            additional_run_info=None,
            status=StatusType.SUCCESS,
        )

    def predict_and_loss(
        self, train: bool = False
    ) -> Tuple[Dict[str, float], np.ndarray]:
        labels = self.y_train if train else self.y_test
        feats = self.X_train if train else self.X_test
        preds = self.predict_function(
            X=feats,
            pipeline=self.pipeline,
            Y_train=self.y_train  # Need this as we need to know all the classes in train splits
        )
        loss_dict = self._loss(labels, preds)

        return loss_dict, preds


# create closure for evaluating an algorithm
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
    pipeline_options: Optional[Dict[str, Any]] = None,
    budget_type: str = None,
    init_params: Optional[Dict[str, Any]] = None,
    logger_port: Optional[int] = None,
    all_supported_metrics: bool = True,
    search_space_updates: Optional[HyperparameterSearchSpaceUpdates] = None,
    instance: str = None,
) -> None:
    evaluator = TestEvaluator(
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
        search_space_updates=search_space_updates)

    evaluator.fit_predict_and_loss()
