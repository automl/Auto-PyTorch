import logging.handlers
import time
import warnings
from multiprocessing.queues import Queue
from typing import Any, Dict, List, Optional, Tuple, Union, no_type_check

from ConfigSpace import Configuration

import numpy as np

import pandas as pd

from sklearn.base import BaseEstimator
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import VotingClassifier

from smac.tae import StatusType

import autoPyTorch.pipeline.image_classification
import autoPyTorch.pipeline.tabular_classification
import autoPyTorch.pipeline.tabular_regression
import autoPyTorch.pipeline.traditional_tabular_classification
from autoPyTorch.constants import (
    CLASSIFICATION_TASKS,
    IMAGE_TASKS,
    MULTICLASS,
    REGRESSION_TASKS,
    STRING_TO_OUTPUT_TYPES,
    STRING_TO_TASK_TYPES,
    TABULAR_TASKS,
)
from autoPyTorch.datasets.base_dataset import BaseDataset
from autoPyTorch.datasets.tabular_dataset import TabularDataset
from autoPyTorch.evaluation.utils import (
    VotingRegressorWrapper,
    convert_multioutput_multiclass_to_multilabel
)
from autoPyTorch.pipeline.base_pipeline import BasePipeline
from autoPyTorch.pipeline.components.training.metrics.base import autoPyTorchMetric
from autoPyTorch.pipeline.components.training.metrics.utils import (
    calculate_loss,
    get_metrics,
)
from autoPyTorch.utils.backend import Backend
from autoPyTorch.utils.common import subsampler
from autoPyTorch.utils.hyperparameter_search_space_update import HyperparameterSearchSpaceUpdates
from autoPyTorch.utils.logging_ import PicklableClientLogger, get_named_client_logger
from autoPyTorch.utils.pipeline import get_dataset_requirements

__all__ = [
    'AbstractEvaluator',
    'fit_and_suppress_warnings'
]


class MyTraditionalTabularClassificationPipeline(BaseEstimator):
    """
    A wrapper class that holds a pipeline for traditional classification.
    Estimators like CatBoost, and Random Forest are considered traditional machine
    learning models and are fitted before neural architecture search.

    This class is an interface to fit a pipeline containing a traditional machine
    learning model, and is the final object that is stored for inference.

    Attributes:
        dataset_properties (Dict[str, Any]):
            A dictionary containing dataset specific information
        random_state (Optional[Union[int, np.random.RandomState]]):
            Object that contains a seed and allows for reproducible results
        init_params  (Optional[Dict]):
            An optional dictionary that is passed to the pipeline's steps. It complies
            a similar function as the kwargs
    """
    def __init__(self, config: str,
                 dataset_properties: Dict[str, Any],
                 random_state: Optional[Union[int, np.random.RandomState]] = None,
                 init_params: Optional[Dict] = None):
        self.config = config
        self.dataset_properties = dataset_properties
        self.random_state = random_state
        self.init_params = init_params
        self.pipeline = autoPyTorch.pipeline.traditional_tabular_classification.\
            TraditionalTabularClassificationPipeline(dataset_properties=dataset_properties)
        configuration_space = self.pipeline.get_hyperparameter_search_space()
        default_configuration = configuration_space.get_default_configuration().get_dictionary()
        default_configuration['model_trainer:tabular_classifier:classifier'] = config
        self.configuration = Configuration(configuration_space, default_configuration)
        self.pipeline.set_hyperparameters(self.configuration)

    def fit(self, X: Dict[str, Any], y: Any,
            sample_weight: Optional[np.ndarray] = None) -> object:
        return self.pipeline.fit(X, y)

    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame],
                      batch_size: int = 1000) -> np.array:
        return self.pipeline.predict_proba(X, batch_size=batch_size)

    def predict(self, X: Union[np.ndarray, pd.DataFrame],
                batch_size: int = 1000) -> np.array:
        return self.pipeline.predict(X, batch_size=batch_size)

    def estimator_supports_iterative_fit(self) -> bool:  # pylint: disable=R0201
        return False

    def get_additional_run_info(self) -> Dict[str, Any]:  # pylint: disable=R0201
        """
        Can be used to return additional info for the run.
        Returns:
            Dict[str, Any]:
            Currently contains
                1. pipeline_configuration: the configuration of the pipeline, i.e, the traditional model used
                2. trainer_configuration: the parameters for the traditional model used.
                    Can be found in autoPyTorch/pipeline/components/setup/traditional_ml/classifier_configs
        """
        return {'pipeline_configuration': self.configuration,
                'trainer_configuration': self.pipeline.named_steps['model_trainer'].choice.model.get_config()}

    def get_pipeline_representation(self) -> Dict[str, str]:
        return self.pipeline.get_pipeline_representation()

    @staticmethod
    def get_default_pipeline_options() -> Dict[str, Any]:
        return autoPyTorch.pipeline.traditional_tabular_classification. \
            TraditionalTabularClassificationPipeline.get_default_pipeline_options()


class DummyClassificationPipeline(DummyClassifier):
    """
    A wrapper class that holds a pipeline for dummy classification.

    A wrapper over DummyClassifier of scikit learn. This estimator is considered the
    worst performing model. In case of failure, at least this model will be fitted.

    Attributes:
        dataset_properties (Dict[str, Any]):
            A dictionary containing dataset specific information
        random_state (Optional[Union[int, np.random.RandomState]]):
            Object that contains a seed and allows for reproducible results
        init_params  (Optional[Dict]):
            An optional dictionary that is passed to the pipeline's steps. It complies
            a similar function as the kwargs
    """
    def __init__(self, config: Configuration,
                 random_state: Optional[Union[int, np.random.RandomState]] = None,
                 init_params: Optional[Dict] = None
                 ) -> None:
        self.config = config
        self.init_params = init_params
        self.random_state = random_state
        if config == 1:
            super(DummyClassificationPipeline, self).__init__(strategy="uniform")
        else:
            super(DummyClassificationPipeline, self).__init__(strategy="most_frequent")

    def fit(self, X: Dict[str, Any], y: Any,
            sample_weight: Optional[np.ndarray] = None) -> object:
        X_train = subsampler(X['X_train'], X['train_indices'])
        y_train = subsampler(X['y_train'], X['train_indices'])
        return super(DummyClassificationPipeline, self).fit(np.ones((X_train.shape[0], 1)), y_train,
                                                            sample_weight=sample_weight)

    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame],
                      batch_size: int = 1000) -> np.array:
        new_X = np.ones((X.shape[0], 1))
        probas = super(DummyClassificationPipeline, self).predict_proba(new_X)
        probas = convert_multioutput_multiclass_to_multilabel(probas).astype(
            np.float32)
        return probas

    def predict(self, X: Union[np.ndarray, pd.DataFrame],
                batch_size: int = 1000) -> np.array:
        new_X = np.ones((X.shape[0], 1))
        return super(DummyClassificationPipeline, self).predict(new_X).astype(np.float32)

    def estimator_supports_iterative_fit(self) -> bool:  # pylint: disable=R0201
        return False

    def get_additional_run_info(self) -> Dict:  # pylint: disable=R0201
        return {}

    def get_pipeline_representation(self) -> Dict[str, str]:
        return {
            'Preprocessing': 'None',
            'Estimator': 'Dummy',
        }

    @staticmethod
    def get_default_pipeline_options() -> Dict[str, Any]:
        return {'budget_type': 'epochs',
                'epochs': 1,
                'runtime': 1}


class DummyRegressionPipeline(DummyRegressor):
    """
    A wrapper class that holds a pipeline for dummy regression.

    A wrapper over DummyRegressor of scikit learn. This estimator is considered the
    worst performing model. In case of failure, at least this model will be fitted.

    Attributes:
        dataset_properties (Dict[str, Any]):
            A dictionary containing dataset specific information
        random_state (Optional[Union[int, np.random.RandomState]]):
            Object that contains a seed and allows for reproducible results
        init_params  (Optional[Dict]):
            An optional dictionary that is passed to the pipeline's steps. It complies
            a similar function as the kwargs
    """
    def __init__(self, config: Configuration,
                 random_state: Optional[Union[int, np.random.RandomState]] = None,
                 init_params: Optional[Dict] = None) -> None:
        self.config = config
        self.init_params = init_params
        self.random_state = random_state
        if config == 1:
            super(DummyRegressionPipeline, self).__init__(strategy='mean')
        else:
            super(DummyRegressionPipeline, self).__init__(strategy='median')

    def fit(self, X: Dict[str, Any], y: Any,
            sample_weight: Optional[np.ndarray] = None) -> object:
        X_train = subsampler(X['X_train'], X['train_indices'])
        y_train = subsampler(X['y_train'], X['train_indices'])
        return super(DummyRegressionPipeline, self).fit(np.ones((X_train.shape[0], 1)), y_train,
                                                        sample_weight=sample_weight)

    def predict(self, X: Union[np.ndarray, pd.DataFrame],
                batch_size: int = 1000) -> np.array:
        new_X = np.ones((X.shape[0], 1))
        return super(DummyRegressionPipeline, self).predict(new_X).astype(np.float32)

    def estimator_supports_iterative_fit(self) -> bool:  # pylint: disable=R0201
        return False

    def get_additional_run_info(self) -> Dict:  # pylint: disable=R0201
        return {}

    @staticmethod
    def get_default_pipeline_options() -> Dict[str, Any]:
        return {'budget_type': 'epochs',
                'epochs': 1,
                'runtime': 1}


def fit_and_suppress_warnings(logger: PicklableClientLogger, pipeline: BaseEstimator,
                              X: Dict[str, Any], y: Any
                              ) -> BaseEstimator:
    @no_type_check
    def send_warnings_to_log(message, category, filename, lineno,
                             file=None, line=None) -> None:
        logger.debug('%s:%s: %s:%s',
                     filename, lineno, category.__name__, message)
        return

    with warnings.catch_warnings():
        warnings.showwarning = send_warnings_to_log
        pipeline.fit(X, y)

    return pipeline


class AbstractEvaluator(object):
    def __init__(self, backend: Backend,
                 queue: Queue,
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
                 disable_file_output: Union[bool, List[str]] = False,
                 init_params: Optional[Dict[str, Any]] = None,
                 logger_port: Optional[int] = None,
                 all_supported_metrics: bool = True,
                 search_space_updates: Optional[HyperparameterSearchSpaceUpdates] = None
                 ) -> None:

        self.starttime = time.time()

        self.configuration = configuration
        self.backend: Backend = backend
        self.queue = queue

        self.datamanager: BaseDataset = self.backend.load_datamanager()

        assert self.datamanager.task_type is not None, \
            "Expected dataset {} to have task_type got None".format(self.datamanager.__class__.__name__)
        self.task_type = STRING_TO_TASK_TYPES[self.datamanager.task_type]
        self.output_type = STRING_TO_OUTPUT_TYPES[self.datamanager.output_type]
        self.issparse = self.datamanager.issparse

        self.include = include
        self.exclude = exclude
        self.search_space_updates = search_space_updates

        self.X_train, self.y_train = self.datamanager.train_tensors

        if self.datamanager.val_tensors is not None:
            self.X_valid, self.y_valid = self.datamanager.val_tensors
        else:
            self.X_valid, self.y_valid = None, None

        if self.datamanager.test_tensors is not None:
            self.X_test, self.y_test = self.datamanager.test_tensors
        else:
            self.X_test, self.y_test = None, None

        self.metric = metric

        self.seed = seed

        # Flag to save target for ensemble
        self.output_y_hat_optimization = output_y_hat_optimization

        if isinstance(disable_file_output, bool):
            self.disable_file_output: bool = disable_file_output
        elif isinstance(disable_file_output, List):
            self.disabled_file_outputs: List[str] = disable_file_output
        else:
            raise ValueError('disable_file_output should be either a bool or a list')

        self.pipeline_class: Optional[Union[BaseEstimator, BasePipeline]] = None
        info: Dict[str, Any] = {'task_type': self.datamanager.task_type,
                                'output_type': self.datamanager.output_type,
                                'issparse': self.issparse}
        if self.task_type in REGRESSION_TASKS:
            if isinstance(self.configuration, int):
                self.pipeline_class = DummyClassificationPipeline
            elif isinstance(self.configuration, str):
                raise ValueError("Only tabular classifications tasks "
                                 "are currently supported with traditional methods")
            elif isinstance(self.configuration, Configuration):
                self.pipeline_class = autoPyTorch.pipeline.tabular_regression.TabularRegressionPipeline
            else:
                raise ValueError('task {} not available'.format(self.task_type))
            self.predict_function = self._predict_regression
        else:
            if isinstance(self.configuration, int):
                self.pipeline_class = DummyClassificationPipeline
            elif isinstance(self.configuration, str):
                if self.task_type in TABULAR_TASKS:
                    self.pipeline_class = MyTraditionalTabularClassificationPipeline
                else:
                    raise ValueError("Only tabular classifications tasks "
                                     "are currently supported with traditional methods")
            elif isinstance(self.configuration, Configuration):
                if self.task_type in TABULAR_TASKS:
                    self.pipeline_class = autoPyTorch.pipeline.tabular_classification.TabularClassificationPipeline
                elif self.task_type in IMAGE_TASKS:
                    self.pipeline_class = autoPyTorch.pipeline.image_classification.ImageClassificationPipeline
                else:
                    raise ValueError('task {} not available'.format(self.task_type))
            self.predict_function = self._predict_proba
        if self.task_type in TABULAR_TASKS:
            assert isinstance(self.datamanager, TabularDataset)
            info.update({'numerical_columns': self.datamanager.numerical_columns,
                         'categorical_columns': self.datamanager.categorical_columns})
        self.dataset_properties = self.datamanager.get_dataset_properties(get_dataset_requirements(info))

        self.additional_metrics: Optional[List[autoPyTorchMetric]] = None
        if all_supported_metrics:
            self.additional_metrics = get_metrics(dataset_properties=self.dataset_properties,
                                                  all_supported_metrics=all_supported_metrics)

        self.fit_dictionary: Dict[str, Any] = {'dataset_properties': self.dataset_properties}
        self._init_params = init_params
        self.fit_dictionary.update({
            'X_train': self.X_train,
            'y_train': self.y_train,
            'X_test': self.X_test,
            'y_test': self.y_test,
            'backend': self.backend,
            'logger_port': logger_port,
        })

        # Update fit dictionary with metrics passed to the evaluator
        metrics_dict: Dict[str, List[str]] = {'additional_metrics': []}
        metrics_dict['additional_metrics'].append(self.metric.name)
        if all_supported_metrics:
            assert self.additional_metrics is not None
            for metric in self.additional_metrics:
                metrics_dict['additional_metrics'].append(metric.name)
        self.fit_dictionary.update(metrics_dict)

        assert self.pipeline_class is not None, "Could not infer pipeline class"
        pipeline_config = pipeline_config if pipeline_config is not None \
            else self.pipeline_class.get_default_pipeline_options()
        self.budget_type = pipeline_config['budget_type'] if budget_type is None else budget_type
        self.budget = pipeline_config[self.budget_type] if budget == 0 else budget
        self.fit_dictionary = {**pipeline_config, **self.fit_dictionary}

        # If the budget is epochs, we want to limit that in the fit dictionary
        if self.budget_type == 'epochs':
            self.fit_dictionary['epochs'] = budget

        self.num_run = 0 if num_run is None else num_run

        logger_name = '%s(%d)' % (self.__class__.__name__.split('.')[-1],
                                  self.seed)
        if logger_port is None:
            logger_port = logging.handlers.DEFAULT_TCP_LOGGING_PORT
        self.logger = get_named_client_logger(
            name=logger_name,
            port=logger_port,
        )
        self.backend.setup_logger(name=logger_name, port=logger_port)

        self.Y_optimization: Optional[np.ndarray] = None
        self.Y_actual_train: Optional[np.ndarray] = None
        self.pipelines: Optional[List[BaseEstimator]] = None
        self.pipeline: Optional[BaseEstimator] = None
        self.logger.debug("Fit dictionary in Abstract evaluator: {}".format(self.fit_dictionary))
        self.logger.debug("Search space updates :{}".format(self.search_space_updates))

    def _get_pipeline(self) -> BaseEstimator:
        assert self.pipeline_class is not None, "Can't return pipeline, pipeline_class not initialised"
        if isinstance(self.configuration, int):
            pipeline = self.pipeline_class(config=self.configuration,
                                           random_state=np.random.RandomState(self.seed),
                                           init_params=self._init_params)
        elif isinstance(self.configuration, Configuration):
            pipeline = self.pipeline_class(config=self.configuration,
                                           dataset_properties=self.dataset_properties,
                                           random_state=np.random.RandomState(self.seed),
                                           include=self.include,
                                           exclude=self.exclude,
                                           init_params=self._init_params,
                                           search_space_updates=self.search_space_updates)
        elif isinstance(self.configuration, str):
            pipeline = self.pipeline_class(config=self.configuration,
                                           dataset_properties=self.dataset_properties,
                                           random_state=np.random.RandomState(self.seed),
                                           init_params=self._init_params)
        else:
            raise ValueError("Invalid configuration entered")
        return pipeline

    def _loss(self, y_true: np.ndarray, y_hat: np.ndarray) -> Dict[str, float]:
        """SMAC follows a minimization goal, so the make_scorer
        sign is used as a guide to obtain the value to reduce.
        The calculate_loss internally translate a score function to
        a minimization problem

        """

        if isinstance(self.configuration, int):
            # We do not calculate performance of the dummy configurations
            return {self.metric.name: self.metric._optimum - self.metric._sign * self.metric._worst_possible_result}

        if self.additional_metrics is not None:
            metrics = self.additional_metrics
        else:
            metrics = [self.metric]

        return calculate_loss(
            y_true, y_hat, self.task_type, metrics)

    def finish_up(self, loss: Dict[str, float], train_loss: Dict[str, float],
                  valid_pred: Optional[np.ndarray], test_pred: Optional[np.ndarray],
                  additional_run_info: Optional[Dict], file_output: bool, status: StatusType,
                  opt_pred: Optional[np.ndarray],
                  ) -> Optional[Tuple[float, float, int, Dict]]:
        """This function does everything necessary after the fitting is done:

        * predicting
        * saving the files for the ensembles_statistics
        * generate output for SMAC
        We use it as the signal handler so we can recycle the code for the
        normal usecase and when the runsolver kills us here :)"""

        assert opt_pred is not None, "Cases where 'opt_pred' is None should be handled " \
                                     "specifically with special child classes"

        self.duration = time.time() - self.starttime

        if file_output:
            loss_, additional_run_info_ = self.file_output(
                opt_pred, valid_pred, test_pred,
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

    def calculate_auxiliary_losses(
            self,
            Y_valid_pred: np.ndarray,
            Y_test_pred: np.ndarray,
    ) -> Tuple[Optional[float], Optional[float]]:

        validation_loss: Optional[float] = None

        if Y_valid_pred is not None:
            if self.y_valid is not None:
                validation_loss_dict = self._loss(self.y_valid, Y_valid_pred)
                validation_loss = validation_loss_dict[self.metric.name]

        test_loss: Optional[float] = None
        if Y_test_pred is not None:
            if self.y_test is not None:
                test_loss_dict = self._loss(self.y_test, Y_test_pred)
                test_loss = test_loss_dict[self.metric.name]

        return validation_loss, test_loss

    def file_output(
            self,
            Y_optimization_pred: np.ndarray,
            Y_valid_pred: np.ndarray,
            Y_test_pred: np.ndarray
    ) -> Tuple[Optional[float], Dict]:
        # Abort if self.Y_optimization is None
        # self.Y_optimization can be None if we use partial-cv, then,
        # obviously no output should be saved.
        if self.Y_optimization is None:
            return None, {}

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

        # This file can be written independently of the others down bellow
        if 'y_optimization' not in self.disabled_file_outputs:
            if self.output_y_hat_optimization:
                self.backend.save_targets_ensemble(self.Y_optimization)

        if hasattr(self, 'pipelines') and self.pipelines is not None:
            if self.pipelines[0] is not None and len(self.pipelines) > 0:
                if 'pipelines' not in self.disabled_file_outputs:
                    if self.task_type in CLASSIFICATION_TASKS:
                        pipelines = VotingClassifier(estimators=None, voting='soft', )
                    else:
                        pipelines = VotingRegressorWrapper(estimators=None)
                    pipelines.estimators_ = self.pipelines
                else:
                    pipelines = None
            else:
                pipelines = None
        else:
            pipelines = None

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
            cv_model=pipelines,
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

    def _predict_proba(self, X: np.ndarray, pipeline: BaseEstimator,
                       Y_train: Optional[np.ndarray] = None) -> np.ndarray:
        @no_type_check
        def send_warnings_to_log(message, category, filename, lineno,
                                 file=None, line=None):
            self.logger.debug('%s:%s: %s:%s' %
                              (filename, lineno, category.__name__, message))
            return

        with warnings.catch_warnings():
            warnings.showwarning = send_warnings_to_log
            Y_pred = pipeline.predict_proba(X, batch_size=1000)

        Y_pred = self._ensure_prediction_array_sizes(Y_pred, Y_train)
        return Y_pred

    def _predict_regression(self, X: np.ndarray, pipeline: BaseEstimator,
                            Y_train: Optional[np.ndarray] = None) -> np.ndarray:
        @no_type_check
        def send_warnings_to_log(message, category, filename, lineno,
                                 file=None, line=None):
            self.logger.debug('%s:%s: %s:%s' %
                              (filename, lineno, category.__name__, message))
            return

        with warnings.catch_warnings():
            warnings.showwarning = send_warnings_to_log
            Y_pred = pipeline.predict(X, batch_size=1000)

        if len(Y_pred.shape) == 1:
            Y_pred = Y_pred.reshape((-1, 1))

        return Y_pred

    def _ensure_prediction_array_sizes(self, prediction: np.ndarray,
                                       Y_train: np.ndarray) -> np.ndarray:
        assert self.datamanager.num_classes is not None, "Called function on wrong task"
        num_classes: int = self.datamanager.num_classes

        if self.output_type == MULTICLASS and \
                prediction.shape[1] < num_classes:
            if Y_train is None:
                raise ValueError('Y_train must not be None!')
            classes = list(np.unique(Y_train))

            mapping = dict()
            for class_number in range(num_classes):
                if class_number in classes:
                    index = classes.index(class_number)
                    mapping[index] = class_number
            new_predictions = np.zeros((prediction.shape[0], num_classes),
                                       dtype=np.float32)

            for index in mapping:
                class_index = mapping[index]
                new_predictions[:, class_index] = prediction[:, index]

            return new_predictions

        return prediction
