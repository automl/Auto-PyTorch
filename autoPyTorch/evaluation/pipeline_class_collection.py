import json
import os
from typing import Any, Dict, Optional, Union

from ConfigSpace import Configuration

import numpy as np

import pandas as pd

from sklearn.base import BaseEstimator
from sklearn.dummy import DummyClassifier, DummyRegressor

import autoPyTorch.pipeline.image_classification
import autoPyTorch.pipeline.tabular_classification
import autoPyTorch.pipeline.tabular_regression
import autoPyTorch.pipeline.traditional_tabular_classification
import autoPyTorch.pipeline.traditional_tabular_regression
from autoPyTorch.constants import (
    IMAGE_TASKS,
    REGRESSION_TASKS,
    TABULAR_TASKS,
)
from autoPyTorch.datasets.base_dataset import BaseDatasetPropertiesType
from autoPyTorch.evaluation.utils import convert_multioutput_multiclass_to_multilabel
from autoPyTorch.pipeline.base_pipeline import BasePipeline
from autoPyTorch.utils.common import replace_string_bool_to_bool, subsampler


def get_default_pipeline_config(choice: str) -> Dict[str, Any]:
    choices = ('default', 'dummy')
    if choice not in choices:
        raise ValueError(f'choice must be in {choices}, but got {choice}')

    return _get_default_pipeline_config() if choice == 'default' else _get_dummy_pipeline_config()


def _get_default_pipeline_config() -> Dict[str, Any]:
    file_path = os.path.join(os.path.dirname(__file__), '../configs/default_pipeline_options.json')
    return replace_string_bool_to_bool(json.load(open(file_path)))


def _get_dummy_pipeline_config() -> Dict[str, Any]:
    file_path = os.path.join(os.path.dirname(__file__), '../configs/dummy_pipeline_options.json')
    return replace_string_bool_to_bool(json.load(open(file_path)))


def get_pipeline_class(
    config: Union[int, str, Configuration],
    task_type: int
) -> Union[BaseEstimator, BasePipeline]:

    pipeline_class: Optional[Union[BaseEstimator, BasePipeline]] = None
    if task_type in REGRESSION_TASKS:
        if isinstance(config, int):
            pipeline_class = DummyRegressionPipeline
        elif isinstance(config, str):
            pipeline_class = MyTraditionalTabularRegressionPipeline
        elif isinstance(config, Configuration):
            pipeline_class = autoPyTorch.pipeline.tabular_regression.TabularRegressionPipeline
        else:
            raise ValueError('task {} not available'.format(task_type))
    else:
        if isinstance(config, int):
            pipeline_class = DummyClassificationPipeline
        elif isinstance(config, str):
            if task_type in TABULAR_TASKS:
                pipeline_class = MyTraditionalTabularClassificationPipeline
            else:
                raise ValueError("Only tabular tasks are currently supported with traditional methods")
        elif isinstance(config, Configuration):
            if task_type in TABULAR_TASKS:
                pipeline_class = autoPyTorch.pipeline.tabular_classification.TabularClassificationPipeline
            elif task_type in IMAGE_TASKS:
                pipeline_class = autoPyTorch.pipeline.image_classification.ImageClassificationPipeline
            else:
                raise ValueError('task {} not available'.format(task_type))

    if pipeline_class is None:
        raise RuntimeError("could not infer pipeline class")

    return pipeline_class


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
                'trainer_configuration': self.pipeline.named_steps['model_trainer'].choice.model.get_config()}

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
