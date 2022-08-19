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
try:
    import autoPyTorch.pipeline.time_series_forecasting
    forecasting_dependencies_installed = True
except ModuleNotFoundError:
    forecasting_dependencies_installed = False
import autoPyTorch.pipeline.traditional_tabular_classification
import autoPyTorch.pipeline.traditional_tabular_regression
from autoPyTorch.automl_common.common.utils.backend import Backend
from autoPyTorch.constants import (
    CLASSIFICATION_TASKS,
    FORECASTING_BUDGET_TYPE,
    FORECASTING_TASKS,
    ForecastingDependenciesNotInstalledMSG,
    IMAGE_TASKS,
    MULTICLASS,
    REGRESSION_TASKS,
    STRING_TO_OUTPUT_TYPES,
    STRING_TO_TASK_TYPES,
    TABULAR_TASKS
)
from autoPyTorch.datasets.base_dataset import (
    BaseDataset,
    BaseDatasetPropertiesType
)
from autoPyTorch.evaluation.utils import (
    DisableFileOutputParameters,
    VotingRegressorWrapper,
    convert_multioutput_multiclass_to_multilabel
)
try:
    from autoPyTorch.evaluation.utils_extra import DummyTimeSeriesForecastingPipeline
    forecasting_dependencies_installed = True
except ModuleNotFoundError:
    forecasting_dependencies_installed = False
from autoPyTorch.pipeline.base_pipeline import BasePipeline
from autoPyTorch.pipeline.components.training.metrics.base import autoPyTorchMetric
from autoPyTorch.pipeline.components.training.metrics.utils import (
    calculate_loss,
    get_metrics
)
from autoPyTorch.utils.common import dict_repr, subsampler
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
        dataset_properties (Dict[str, BaseDatasetPropertiesType]):
            A dictionary containing dataset specific information
        random_state (Optional[np.random.RandomState]):
            Object that contains a seed and allows for reproducible results
        init_params  (Optional[Dict]):
            An optional dictionary that is passed to the pipeline's steps. It complies
            a similar function as the kwargs
    """

    def __init__(self, config: str,
                 dataset_properties: Dict[str, BaseDatasetPropertiesType],
                 random_state: Optional[Union[int, np.random.RandomState]] = None,
                 init_params: Optional[Dict] = None):
        self.config = config
        self.dataset_properties = dataset_properties
        self.random_state = random_state
        self.init_params = init_params
        self.pipeline = autoPyTorch.pipeline.traditional_tabular_classification. \
            TraditionalTabularClassificationPipeline(dataset_properties=dataset_properties,
                                                     random_state=self.random_state)
        configuration_space = self.pipeline.get_hyperparameter_search_space()
        default_configuration = configuration_space.get_default_configuration().get_dictionary()
        default_configuration['model_trainer:tabular_traditional_model:traditional_learner'] = config
        self.configuration = Configuration(configuration_space, default_configuration)
        self.pipeline.set_hyperparameters(self.configuration)

    def fit(self, X: Dict[str, Any], y: Any,
            sample_weight: Optional[np.ndarray] = None) -> object:
        return self.pipeline.fit(X, y)

    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame],
                      batch_size: int = 1000) -> np.ndarray:
        return self.pipeline.predict_proba(X, batch_size=batch_size)

    def predict(self, X: Union[np.ndarray, pd.DataFrame],
                batch_size: int = 1000) -> np.ndarray:
        return self.pipeline.predict(X, batch_size=batch_size)

    def get_additional_run_info(self) -> Dict[str, Any]:
        """
        Can be used to return additional info for the run.
        Returns:
            Dict[str, Any]:
            Currently contains
                1. pipeline_configuration: the configuration of the pipeline, i.e, the traditional model used
                2. trainer_configuration: the parameters for the traditional model used.
                    Can be found in autoPyTorch/pipeline/components/setup/traditional_ml/estimator_configs
        """
        return {'pipeline_configuration': self.configuration,
                'trainer_configuration': self.pipeline.named_steps['model_trainer'].choice.model.get_config(),
                'configuration_origin': 'traditional'}

    def get_pipeline_representation(self) -> Dict[str, str]:
        return self.pipeline.get_pipeline_representation()

    @staticmethod
    def get_default_pipeline_options() -> Dict[str, Any]:
        return autoPyTorch.pipeline.traditional_tabular_classification. \
            TraditionalTabularClassificationPipeline.get_default_pipeline_options()


class MyTraditionalTabularRegressionPipeline(BaseEstimator):
    """
    A wrapper class that holds a pipeline for traditional regression.
    Estimators like CatBoost, and Random Forest are considered traditional machine
    learning models and are fitted before neural architecture search.

    This class is an interface to fit a pipeline containing a traditional machine
    learning model, and is the final object that is stored for inference.

    Attributes:
        dataset_properties (Dict[str, Any]):
            A dictionary containing dataset specific information
        random_state (Optional[np.random.RandomState]):
            Object that contains a seed and allows for reproducible results
        init_params  (Optional[Dict]):
            An optional dictionary that is passed to the pipeline's steps. It complies
            a similar function as the kwargs
    """
    def __init__(self, config: str,
                 dataset_properties: Dict[str, Any],
                 random_state: Optional[np.random.RandomState] = None,
                 init_params: Optional[Dict] = None):
        self.config = config
        self.dataset_properties = dataset_properties
        self.random_state = random_state
        self.init_params = init_params
        self.pipeline = autoPyTorch.pipeline.traditional_tabular_regression. \
            TraditionalTabularRegressionPipeline(dataset_properties=dataset_properties,
                                                 random_state=self.random_state)
        configuration_space = self.pipeline.get_hyperparameter_search_space()
        default_configuration = configuration_space.get_default_configuration().get_dictionary()
        default_configuration['model_trainer:tabular_traditional_model:traditional_learner'] = config
        self.configuration = Configuration(configuration_space, default_configuration)
        self.pipeline.set_hyperparameters(self.configuration)

    def fit(self, X: Dict[str, Any], y: Any,
            sample_weight: Optional[np.ndarray] = None) -> object:
        return self.pipeline.fit(X, y)

    def predict(self, X: Union[np.ndarray, pd.DataFrame],
                batch_size: int = 1000) -> np.ndarray:
        return self.pipeline.predict(X, batch_size=batch_size)

    def get_additional_run_info(self) -> Dict[str, Any]:
        """
        Can be used to return additional info for the run.
        Returns:
            Dict[str, Any]:
            Currently contains
                1. pipeline_configuration: the configuration of the pipeline, i.e, the traditional model used
                2. trainer_configuration: the parameters for the traditional model used.
                    Can be found in autoPyTorch/pipeline/components/setup/traditional_ml/estimator_configs
        """
        return {'pipeline_configuration': self.configuration,
                'trainer_configuration': self.pipeline.named_steps['model_trainer'].choice.model.get_config(),
                'configuration_origin': 'traditional'}

    def get_pipeline_representation(self) -> Dict[str, str]:
        return self.pipeline.get_pipeline_representation()

    @staticmethod
    def get_default_pipeline_options() -> Dict[str, Any]:
        return autoPyTorch.pipeline.traditional_tabular_regression.\
            TraditionalTabularRegressionPipeline.get_default_pipeline_options()


class DummyClassificationPipeline(DummyClassifier):
    """
    A wrapper class that holds a pipeline for dummy classification.

    A wrapper over DummyClassifier of scikit learn. This estimator is considered the
    worst performing model. In case of failure, at least this model will be fitted.

    Attributes:
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
                      batch_size: int = 1000) -> np.ndarray:
        new_X = np.ones((X.shape[0], 1))
        probas = super(DummyClassificationPipeline, self).predict_proba(new_X)
        probas = convert_multioutput_multiclass_to_multilabel(probas).astype(
            np.float32)
        return probas

    def predict(self, X: Union[np.ndarray, pd.DataFrame],
                batch_size: int = 1000) -> np.ndarray:
        new_X = np.ones((X.shape[0], 1))
        return super(DummyClassificationPipeline, self).predict(new_X).astype(np.float32)

    def get_additional_run_info(self) -> Dict:  # pylint: disable=R0201
        return {'configuration_origin': 'DUMMY'}

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
                batch_size: int = 1000) -> np.ndarray:
        new_X = np.ones((X.shape[0], 1))
        return super(DummyRegressionPipeline, self).predict(new_X).astype(np.float32)

    def get_additional_run_info(self) -> Dict:  # pylint: disable=R0201
        return {'configuration_origin': 'DUMMY'}

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
    """
    This method defines the interface that pipeline evaluators should follow, when
    interacting with SMAC through ExecuteTaFuncWithQueue.

    An evaluator is an object that:
        + constructs a pipeline (i.e. a classification or regression estimator) for a given
          pipeline_options and run settings (budget, seed)
        + Fits and trains this pipeline (TrainEvaluator) or tests a given
          configuration (TestEvaluator)

    The provided configuration determines the type of pipeline created. For more
    details, please read the get_pipeline() method.

    Attributes:
        backend (Backend):
            An object that allows interaction with the disk storage. In particular, allows to
            access the train and test datasets
        queue (Queue):
            Each worker available will instantiate an evaluator, and after completion,
            it will append the result to a multiprocessing queue
        metric (autoPyTorchMetric):
            A scorer object that is able to evaluate how good a pipeline was fit. It
            is a wrapper on top of the actual score method (a wrapper on top of
            scikit-learn accuracy for example) that formats the predictions accordingly.
        budget: (float):
            The amount of epochs/time a configuration is allowed to run.
        budget_type  (str):
            The budget type. Currently, only epoch and time are allowed.
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
            socket port for the communication channel.
            If None is provided, the logging.handlers.DEFAULT_TCP_LOGGING_PORT is used.
        all_supported_metrics  (bool):
            Whether all supported metrics should be calculated for every configuration.
        search_space_updates (Optional[HyperparameterSearchSpaceUpdates]):
            An object used to fine tune the hyperparameter search space of the pipeline
    """
    def __init__(self, backend: Backend,
                 queue: Queue,
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
                 all_supported_metrics: bool = True,
                 search_space_updates: Optional[HyperparameterSearchSpaceUpdates] = None
                 ) -> None:

        self.starttime = time.time()

        self.configuration = configuration
        self.backend: Backend = backend
        self.queue = queue

        self.include = include
        self.exclude = exclude
        self.search_space_updates = search_space_updates

        self.metric = metric

        self.seed = seed

        self._init_datamanager_info()

        # Flag to save target for ensemble
        self.output_y_hat_optimization = output_y_hat_optimization

        disable_file_output = disable_file_output if disable_file_output is not None else []
        # check compatibility of disable file output
        DisableFileOutputParameters.check_compatibility(disable_file_output)

        self.disable_file_output = disable_file_output

        self.pipeline_class: Optional[Union[BaseEstimator, BasePipeline]] = None
        if self.task_type in REGRESSION_TASKS:
            if isinstance(self.configuration, int):
                self.pipeline_class = DummyRegressionPipeline
            elif isinstance(self.configuration, str):
                self.pipeline_class = MyTraditionalTabularRegressionPipeline
            elif isinstance(self.configuration, Configuration):
                self.pipeline_class = autoPyTorch.pipeline.tabular_regression.TabularRegressionPipeline
            else:
                raise ValueError('task {} not available'.format(self.task_type))
            self.predict_function = self._predict_regression
        elif self.task_type in CLASSIFICATION_TASKS:
            if isinstance(self.configuration, int):
                self.pipeline_class = DummyClassificationPipeline
            elif isinstance(self.configuration, str):
                if self.task_type in TABULAR_TASKS:
                    self.pipeline_class = MyTraditionalTabularClassificationPipeline
                else:
                    raise ValueError("Only tabular tasks are currently supported with traditional methods")
            elif isinstance(self.configuration, Configuration):
                if self.task_type in TABULAR_TASKS:
                    self.pipeline_class = autoPyTorch.pipeline.tabular_classification.TabularClassificationPipeline
                elif self.task_type in IMAGE_TASKS:
                    self.pipeline_class = autoPyTorch.pipeline.image_classification.ImageClassificationPipeline
                else:
                    raise ValueError('task {} not available'.format(self.task_type))
            self.predict_function = self._predict_proba
        elif self.task_type in FORECASTING_TASKS:
            if isinstance(self.configuration, int):
                if not forecasting_dependencies_installed:
                    raise ModuleNotFoundError(ForecastingDependenciesNotInstalledMSG)
                self.pipeline_class = DummyTimeSeriesForecastingPipeline
            elif isinstance(self.configuration, str):
                raise ValueError("Only tabular classifications tasks "
                                 "are currently supported with traditional methods")
            elif isinstance(self.configuration, Configuration):
                self.pipeline_class = autoPyTorch.pipeline.time_series_forecasting.TimeSeriesForecastingPipeline
            else:
                raise ValueError('task {} not available'.format(self.task_type))
            self.predict_function = self._predict_regression

        self.additional_metrics: Optional[List[autoPyTorchMetric]] = None
        metrics_dict: Optional[Dict[str, List[str]]] = None
        if all_supported_metrics:
            self.additional_metrics = get_metrics(dataset_properties=self.dataset_properties,
                                                  all_supported_metrics=all_supported_metrics)
            # Update fit dictionary with metrics passed to the evaluator
            metrics_dict = {'additional_metrics': []}
            metrics_dict['additional_metrics'].append(self.metric.name)
            for metric in self.additional_metrics:
                metrics_dict['additional_metrics'].append(metric.name)

        self._init_params = init_params

        assert self.pipeline_class is not None, "Could not infer pipeline class"
        pipeline_options = pipeline_options if pipeline_options is not None \
            else self.pipeline_class.get_default_pipeline_options()
        self.budget_type = pipeline_options['budget_type'] if budget_type is None else budget_type
        self.budget = pipeline_options[self.budget_type] if budget == 0 else budget

        self.num_run = 0 if num_run is None else num_run

        logger_name = '%s(%d)' % (self.__class__.__name__.split('.')[-1],
                                  self.seed)
        if logger_port is None:
            logger_port = logging.handlers.DEFAULT_TCP_LOGGING_PORT
        self.logger = get_named_client_logger(
            name=logger_name,
            port=logger_port,
        )

        self._init_fit_dictionary(logger_port=logger_port, pipeline_options=pipeline_options, metrics_dict=metrics_dict)
        self.Y_optimization: Optional[np.ndarray] = None
        self.Y_actual_train: Optional[np.ndarray] = None
        self.pipelines: Optional[List[BaseEstimator]] = None
        self.pipeline: Optional[BaseEstimator] = None
        self.logger.debug("Fit dictionary in Abstract evaluator: {}".format(dict_repr(self.fit_dictionary)))
        self.logger.debug("Search space updates :{}".format(self.search_space_updates))

    def _init_datamanager_info(
        self,
    ) -> None:
        """
        Initialises instance attributes that come from the datamanager.
        For example,
            X_train, y_train, etc.
        """

        datamanager: BaseDataset = self.backend.load_datamanager()

        assert datamanager.task_type is not None, \
            "Expected dataset {} to have task_type got None".format(datamanager.__class__.__name__)
        self.task_type = STRING_TO_TASK_TYPES[datamanager.task_type]
        self.output_type = STRING_TO_OUTPUT_TYPES[datamanager.output_type]
        self.issparse = datamanager.issparse

        self.X_train, self.y_train = datamanager.train_tensors

        if datamanager.val_tensors is not None:
            self.X_valid, self.y_valid = datamanager.val_tensors
        else:
            self.X_valid, self.y_valid = None, None

        if datamanager.test_tensors is not None:
            self.X_test, self.y_test = datamanager.test_tensors
        else:
            self.X_test, self.y_test = None, None

        self.resampling_strategy = datamanager.resampling_strategy

        self.num_classes: Optional[int] = getattr(datamanager, "num_classes", None)

        self.dataset_properties = datamanager.get_dataset_properties(
            get_dataset_requirements(info=datamanager.get_required_dataset_info(),
                                     include=self.include,
                                     exclude=self.exclude,
                                     search_space_updates=self.search_space_updates
                                     ))
        self.splits = datamanager.splits
        if self.splits is None:
            raise AttributeError(f"create_splits on {datamanager.__class__.__name__} must be called "
                                 f"before the instantiation of {self.__class__.__name__}")

        # delete datamanager from memory
        del datamanager

    def _init_fit_dictionary(
        self,
        logger_port: int,
        pipeline_options: Dict[str, Any],
        metrics_dict: Optional[Dict[str, List[str]]] = None,
    ) -> None:
        """
        Initialises the fit dictionary

        Args:
            logger_port (int):
                Logging is performed using a socket-server scheme to be robust against many
                parallel entities that want to write to the same file. This integer states the
                socket port for the communication channel.
            pipeline_options (Dict[str, Any]):
                Defines the content of the pipeline being evaluated. For example, it
                contains pipeline specific settings like logging name, or whether or not
                to use tensorboard.
            metrics_dict (Optional[Dict[str, List[str]]]):
            Contains a list of metric names to be evaluated in Trainer with key `additional_metrics`. Defaults to None.

        Returns:
            None
        """

        self.fit_dictionary: Dict[str, Any] = {'dataset_properties': self.dataset_properties}

        if metrics_dict is not None:
            self.fit_dictionary.update(metrics_dict)

        self.fit_dictionary.update({
            'X_train': self.X_train,
            'y_train': self.y_train,
            'X_test': self.X_test,
            'y_test': self.y_test,
            'backend': self.backend,
            'logger_port': logger_port,
            'optimize_metric': self.metric.name
        })

        self.fit_dictionary.update(pipeline_options)
        # If the budget is epochs, we want to limit that in the fit dictionary
        if self.budget_type == 'epochs':
            self.fit_dictionary['epochs'] = self.budget
            self.fit_dictionary.pop('runtime', None)
        elif self.budget_type == 'runtime':
            self.fit_dictionary['runtime'] = self.budget
            self.fit_dictionary.pop('epochs', None)
        elif self.budget_type == 'resolution' and self.task_type in FORECASTING_TASKS:
            self.fit_dictionary['sample_interval'] = int(np.ceil(1.0 / self.budget))
            self.fit_dictionary.pop('epochs', None)
            self.fit_dictionary.pop('runtime', None)
        elif self.budget_type == 'num_seq':
            self.fit_dictionary['fraction_seq'] = self.budget
            self.fit_dictionary.pop('epochs', None)
            self.fit_dictionary.pop('runtime', None)
        elif self.budget_type == 'num_sample_per_seq':
            self.fit_dictionary['fraction_samples_per_seq'] = self.budget
            self.fit_dictionary.pop('epochs', None)
            self.fit_dictionary.pop('runtime', None)
        else:
            raise ValueError(f"budget type must be `epochs` or `runtime` or {FORECASTING_BUDGET_TYPE} "
                             f"(Only used by forecasting taskss), but got {self.budget_type}")

    def _get_pipeline(self) -> BaseEstimator:
        """
        Implements a pipeline object based on the self.configuration attribute.
        int: A dummy classifier/dummy regressor is created. This estimator serves
             as a baseline model to ignore all models that perform worst than this
             fixed estimator. Also, in the worst case scenario, this is the final
             estimator created (for instance, in case not enough memory was allocated).
        str: A pipeline with traditional classifiers like random forest, SVM, etc is created,
             as the configuration will contain an estimator name defining the configuration
             to use, for example 'RandomForest'
        Configuration: A pipeline object matching this configuration is created. This
             is the case of neural architecture search, where different backbones
             and head can be passed in the form of a configuration object.

        Returns
            pipeline (BaseEstimator):
                A scikit-learn compliant pipeline which is not yet fit to the data.
        """
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

    def _loss(self, y_true: np.ndarray, y_hat: np.ndarray, **metric_kwargs: Any) -> Dict[str, float]:
        """SMAC follows a minimization goal, so the make_scorer
        sign is used as a guide to obtain the value to reduce.
        The calculate_loss internally translate a score function to
        a minimization problem

        Args:
            y_true (np.ndarray):
                The expect labels given by the original dataset
            y_hat (np.ndarray):
                The prediction of the current pipeline being fit
        Returns:
            (Dict[str, float]):
                A dictionary with metric_name -> metric_loss, for every
                supported metric
        """

        if isinstance(self.configuration, int):
            # We do not calculate performance of the dummy configurations
            return {self.metric.name: self.metric._optimum - self.metric._sign * self.metric._worst_possible_result}

        if self.additional_metrics is not None:
            metrics = self.additional_metrics
        else:
            metrics = [self.metric]
        return calculate_loss(
            y_true, y_hat, self.task_type, metrics, **metric_kwargs)

    def finish_up(self, loss: Dict[str, float], train_loss: Dict[str, float],
                  opt_pred: np.ndarray, valid_pred: Optional[np.ndarray],
                  test_pred: Optional[np.ndarray], additional_run_info: Optional[Dict],
                  file_output: bool, status: StatusType, **metric_kwargs: Any
                  ) -> Optional[Tuple[float, float, int, Dict]]:
        """This function does everything necessary after the fitting is done:

        * predicting
        * saving the files for the ensembles_statistics
        * generate output for SMAC
        We use it as the signal handler so we can recycle the code for the
        normal usecase and when the runsolver kills us here :)

        Args:
            loss (Dict[str, float]):
                The optimization loss, calculated on the validation set. This will
                be the cost used in SMAC
            train_loss (Dict[str, float]):
                The train loss, calculated on the train set
            opt_pred (np.ndarray):
                The predictions on the validation set. This validation set is created
                from the resampling strategy
            valid_pred (Optional[np.ndarray]):
                Predictions on a user provided validation set
            test_pred (Optional[np.ndarray]):
                Predictions on a user provided test set
            additional_run_info (Optional[Dict]):
                A dictionary with additional run information, like duration or
                the crash error msg, if any.
            file_output (bool):
                Whether or not this pipeline should output information to disk
            status (StatusType)
                The status of the run, following SMAC StatusType syntax.
            metric_kwargs (Any)
                Additional arguments for computing metrics

        Returns:
            duration (float):
                The elapsed time of the training of this evaluator
            loss (float):
                The optimization loss of this run
            seed (int):
                The seed used while fitting the pipeline
            additional_info (Dict):
                Additional run information, like train/test loss
        """

        self.duration = time.time() - self.starttime

        if file_output:
            loss_, additional_run_info_ = self.file_output(
                opt_pred, valid_pred, test_pred,
            )
        else:
            loss_ = None
            additional_run_info_ = {}

        validation_loss, test_loss = self.calculate_auxiliary_losses(
            valid_pred, test_pred, **metric_kwargs
        )

        if loss_ is not None:
            return self.duration, loss_, self.seed, additional_run_info_

        cost = loss[self.metric.name]

        additional_run_info = (
            {} if additional_run_info is None else additional_run_info
        )
        additional_run_info['opt_loss'] = loss
        additional_run_info['duration'] = self.duration
        additional_run_info['num_run'] = self.num_run
        if train_loss is not None:
            additional_run_info['train_loss'] = train_loss
        if validation_loss is not None:
            additional_run_info['validation_loss'] = validation_loss
        if test_loss is not None:
            additional_run_info['test_loss'] = test_loss

        # Add information to additional info that can be useful for other functionalities
        additional_run_info['configuration'] = self.configuration \
            if not isinstance(self.configuration, Configuration) else self.configuration.get_dictionary()
        additional_run_info['budget'] = self.budget

        rval_dict = {'loss': cost,
                     'additional_run_info': additional_run_info,
                     'status': status}

        self.queue.put(rval_dict)
        return None

    def calculate_auxiliary_losses(
        self,
        Y_valid_pred: np.ndarray,
        Y_test_pred: np.ndarray,
        **metric_kwargs: Any
    ) -> Tuple[Optional[Dict[str, float]], Optional[Dict[str, float]]]:
        """
        A helper function to calculate the performance estimate of the
        current pipeline in the user provided validation/test set.

        Args:
            Y_valid_pred (np.ndarray):
                predictions on a validation set provided by the user,
                matching self.y_valid
            Y_test_pred (np.ndarray):
                predictions on a test set provided by the user,
                matching self.y_test
            metric_kwargs (Any)
                additional argument for evaluating the loss metric

        Returns:
            validation_loss_dict (Optional[Dict[str, float]]):
                Various validation losses available.
            test_loss_dict (Optional[Dict[str, float]]):
                Various test losses available.
        """

        validation_loss_dict: Optional[Dict[str, float]] = None

        if Y_valid_pred is not None:
            if self.y_valid is not None:
                validation_loss_dict = self._loss(self.y_valid, Y_valid_pred, **metric_kwargs)

        test_loss_dict: Optional[Dict[str, float]] = None
        if Y_test_pred is not None:
            if self.y_test is not None:
                test_loss_dict = self._loss(self.y_test, Y_test_pred, **metric_kwargs)

        return validation_loss_dict, test_loss_dict

    def file_output(
        self,
        Y_optimization_pred: np.ndarray,
        Y_valid_pred: np.ndarray,
        Y_test_pred: np.ndarray
    ) -> Tuple[Optional[float], Dict]:
        """
        This method decides what file outputs are written to disk.

        It is also the interface to the backed save_numrun_to_dir
        which stores all the pipeline related information to a single
        directory for easy identification of the current run.

        Args:
            Y_optimization_pred (np.ndarray):
                The pipeline predictions on the validation set internally created
                from self.y_train
            Y_valid_pred (np.ndarray):
                The pipeline predictions on the user provided validation set,
                which should match self.y_valid
            Y_test_pred (np.ndarray):
                The pipeline predictions on the user provided test set,
                which should match self.y_test
        Returns:
            loss (Optional[float]):
                A loss in case the run failed to store files to
                disk
            error_dict (Dict):
                A dictionary with an error that explains why a run
                was not successfully stored to disk.
        """
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
        if 'all' in self.disable_file_output:
            return None, {}

        # This file can be written independently of the others down bellow
        if 'y_optimization' not in self.disable_file_output:
            if self.output_y_hat_optimization:
                self.backend.save_targets_ensemble(self.Y_optimization)

        if getattr(self, 'pipelines', None) is not None:
            if self.pipelines[0] is not None and len(self.pipelines) > 0:  # type: ignore[index, arg-type]
                if 'pipelines' not in self.disable_file_output:
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

        if getattr(self, 'pipeline', None) is not None:
            if 'pipeline' not in self.disable_file_output:
                pipeline = self.pipeline
            else:
                pipeline = None
        else:
            pipeline = None

        self.logger.debug("Saving directory {}, {}, {}".format(self.seed, self.num_run, self.budget))
        self.backend.save_numrun_to_dir(
            seed=int(self.seed),
            idx=int(self.num_run),
            budget=float(self.budget),
            model=pipeline,
            cv_model=pipelines,
            ensemble_predictions=(
                Y_optimization_pred if 'y_optimization' not in
                                       self.disable_file_output else None
            ),
            valid_predictions=(
                Y_valid_pred if 'y_valid' not in
                                self.disable_file_output else None
            ),
            test_predictions=(
                Y_test_pred if 'y_test' not in
                               self.disable_file_output else None
            ),
        )

        return None, {}

    def _predict_proba(self, X: np.ndarray, pipeline: BaseEstimator,
                       Y_train: Optional[np.ndarray] = None) -> np.ndarray:
        """
        A wrapper function to handle the prediction of classification tasks.
        It also makes sure that the predictions has the same dimensionality
        as the expected labels

        Args:
            X (np.ndarray):
                A set of features to feed to the pipeline
            pipeline (BaseEstimator):
                A model that will take the features X return a prediction y
                This pipeline must be a classification estimator that supports
                the predict_proba method.
            Y_train (Optional[np.ndarray]):
        Returns:
            (np.ndarray):
                The predictions of pipeline for the given features X
        """
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
        """
        A wrapper function to handle the prediction of regression tasks.
        It is a wrapper to provide the same interface to _predict_proba

        Regression predictions expects an unraveled dimensionality.
        To comply with scikit-learn VotingRegressor requirement, if the estimator
        predicts a (N,) shaped array, it is converted to (N, 1)

        Args:
            X (np.ndarray):
                A set of features to feed to the pipeline
            pipeline (BaseEstimator):
                A model that will take the features X return a prediction y
            Y_train (Optional[np.ndarray]):
        Returns:
            (np.ndarray):
                The predictions of pipeline for the given features X
        """
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
        """
        This method formats a prediction to match the dimensionality of the provided
        labels (Y_train). This should be used exclusively for classification tasks

        Args:
            prediction (np.ndarray):
                The un-formatted predictions of a pipeline
            Y_train (np.ndarray):
                The labels from the dataset to give an intuition of the expected
                predictions dimensionality
        Returns:
            (np.ndarray):
                The formatted prediction
        """
        assert self.num_classes is not None, "Called function on wrong task"

        if self.output_type == MULTICLASS and \
                prediction.shape[1] < self.num_classes:
            if Y_train is None:
                raise ValueError('Y_train must not be None!')
            classes = list(np.unique(Y_train))

            mapping = dict()
            for class_number in range(self.num_classes):
                if class_number in classes:
                    index = classes.index(class_number)
                    mapping[index] = class_number
            new_predictions = np.zeros((prediction.shape[0], self.num_classes),
                                       dtype=np.float32)

            for index in mapping:
                class_index = mapping[index]
                new_predictions[:, class_index] = prediction[:, index]

            return new_predictions

        return prediction
