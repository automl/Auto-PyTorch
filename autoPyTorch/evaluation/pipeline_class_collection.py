import json
import os
from typing import Any, Dict, Optional, Type, Union

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


def get_default_pipeline_config(choice: str = 'default') -> Dict[str, Any]:
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

    is_reg = (task_type in REGRESSION_TASKS)

    if isinstance(config, int):
        return DummyRegressionPipeline if is_reg else DummyClassificationPipeline
    elif isinstance(config, str):
        if is_reg:
            return MyTraditionalTabularRegressionPipeline

        if task_type not in TABULAR_TASKS:
            # Time series and image tasks
            raise NotImplementedError(f'classification task on {task_type} for traditional methods is not available')

        return MyTraditionalTabularClassificationPipeline
    elif isinstance(config, Configuration):
        if is_reg:
            return autoPyTorch.pipeline.tabular_regression.TabularRegressionPipeline

        if task_type in TABULAR_TASKS:
            return autoPyTorch.pipeline.tabular_classification.TabularClassificationPipeline
        elif task_type in IMAGE_TASKS:
            return autoPyTorch.pipeline.image_classification.ImageClassificationPipeline
        else:
            raise NotImplementedError(f'classification task on {task_type} for traditional methods is not available')
    else:
        raise RuntimeError("could not infer pipeline class")


class BaseMyTraditionalPipeline:
    """
    A wrapper class that holds a pipeline for traditional regression/classification.
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
    def __init__(
        self,
        config: str,
        pipeline_class: Union[
            Type[autoPyTorch.pipeline.traditional_tabular_regression.TraditionalTabularRegressionPipeline],
            Type[autoPyTorch.pipeline.traditional_tabular_classification.TraditionalTabularClassificationPipeline]
        ],
        dataset_properties: Dict[str, BaseDatasetPropertiesType],
        random_state: Optional[Union[int, np.random.RandomState]] = None,
        init_params: Optional[Dict] = None
    ):
        self.config = config
        self.dataset_properties = dataset_properties
        self.random_state = random_state
        self.init_params = init_params
        self.pipeline = pipeline_class(dataset_properties=dataset_properties, random_state=self.random_state)

        configuration_space = self.pipeline.get_hyperparameter_search_space()
        default_configuration = configuration_space.get_default_configuration().get_dictionary()
        default_configuration['model_trainer:tabular_traditional_model:traditional_learner'] = config
        self.configuration = Configuration(configuration_space, default_configuration)
        self.pipeline.set_hyperparameters(self.configuration)

    def fit(self, X: Dict[str, Any], y: Any, sample_weight: Optional[np.ndarray] = None) -> object:
        return self.pipeline.fit(X, y)

    def predict(self, X: Union[np.ndarray, pd.DataFrame], batch_size: int = 1000) -> np.ndarray:
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
        return {
            'pipeline_configuration': self.configuration,
            'trainer_configuration': self.pipeline.named_steps['model_trainer'].choice.model.get_config(),
            'configuration_origin': 'traditional'
        }

    def get_pipeline_representation(self) -> Dict[str, str]:
        return self.pipeline.get_pipeline_representation()

    @staticmethod
    def get_default_pipeline_config() -> Dict[str, Any]:
        return _get_default_pipeline_config()


class MyTraditionalTabularClassificationPipeline(BaseMyTraditionalPipeline, BaseEstimator):
    """ A wrapper class that holds a pipeline for traditional classification. """
    def __init__(
        self,
        config: str,
        dataset_properties: Dict[str, BaseDatasetPropertiesType],
        random_state: Optional[Union[int, np.random.RandomState]] = None,
        init_params: Optional[Dict] = None
    ):

        _pl = autoPyTorch.pipeline.traditional_tabular_classification.TraditionalTabularClassificationPipeline
        BaseMyTraditionalPipeline.__init__(
            self,
            config=config,
            dataset_properties=dataset_properties,
            random_state=random_state,
            init_params=init_params,
            pipeline_class=_pl
        )

    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame], batch_size: int = 1000) -> np.ndarray:
        return self.pipeline.predict_proba(X, batch_size=batch_size)


class MyTraditionalTabularRegressionPipeline(BaseMyTraditionalPipeline, BaseEstimator):
    """ A wrapper class that holds a pipeline for traditional regression. """
    def __init__(
        self,
        config: str,
        dataset_properties: Dict[str, Any],
        random_state: Optional[np.random.RandomState] = None,
        init_params: Optional[Dict] = None
    ):

        BaseMyTraditionalPipeline.__init__(
            self,
            config=config,
            dataset_properties=dataset_properties,
            random_state=random_state,
            init_params=init_params,
            pipeline_class=autoPyTorch.pipeline.traditional_tabular_regression.TraditionalTabularRegressionPipeline
        )


class BaseDummyPipeline:
    """
    Base class for wrapper classes that hold a pipeline for
    dummy {classification/regression}.

    This estimator is considered the worst performing model.
    In case of failure, at least this model will be fitted.
    """
    def __init__(
        self,
        config: int,
        random_state: Optional[Union[int, np.random.RandomState]] = None,
        init_params: Optional[Dict] = None
    ):
        self.config = config
        self.init_params = init_params
        self.random_state = random_state

    def get_additional_run_info(self) -> Dict:  # pylint: disable=R0201
        return {'configuration_origin': 'DUMMY'}

    def get_pipeline_representation(self) -> Dict[str, str]:
        return {'Preprocessing': 'None', 'Estimator': 'Dummy'}

    @staticmethod
    def get_default_pipeline_config() -> Dict[str, Any]:
        return _get_dummy_pipeline_config()


class DummyClassificationPipeline(DummyClassifier, BaseDummyPipeline):
    """ A wrapper over DummyClassifier of scikit learn. """
    def __init__(
        self,
        config: int,
        random_state: Optional[Union[int, np.random.RandomState]] = None,
        init_params: Optional[Dict] = None
    ):
        BaseDummyPipeline.__init__(self, config=config, random_state=random_state, init_params=init_params)
        DummyClassifier.__init__(self, strategy="uniform" if config == 1 else "most_frequent")

    def fit(self, X: Dict[str, Any], y: Any, sample_weight: Optional[np.ndarray] = None) -> object:
        X_train = subsampler(X['X_train'], X['train_indices'])
        y_train = subsampler(X['y_train'], X['train_indices'])
        X_new = np.ones((X_train.shape[0], 1))
        return super(DummyClassificationPipeline, self).fit(X_new, y_train, sample_weight=sample_weight)

    def predict(self, X: Union[np.ndarray, pd.DataFrame], batch_size: int = 1000) -> np.ndarray:
        new_X = np.ones((X.shape[0], 1))
        return super(DummyClassificationPipeline, self).predict(new_X).astype(np.float32)

    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame], batch_size: int = 1000) -> np.ndarray:
        new_X = np.ones((X.shape[0], 1))
        probas = super(DummyClassificationPipeline, self).predict_proba(new_X)
        return convert_multioutput_multiclass_to_multilabel(probas).astype(np.float32)


class DummyRegressionPipeline(DummyRegressor, BaseDummyPipeline):
    """ A wrapper over DummyRegressor of scikit learn. """
    def __init__(
        self,
        config: int,
        random_state: Optional[Union[int, np.random.RandomState]] = None,
        init_params: Optional[Dict] = None
    ):
        BaseDummyPipeline.__init__(self, config=config, random_state=random_state, init_params=init_params)
        DummyRegressor.__init__(self, strategy='mean' if config == 1 else 'median')

    def fit(self, X: Dict[str, Any], y: Any, sample_weight: Optional[np.ndarray] = None) -> object:
        X_train = subsampler(X['X_train'], X['train_indices'])
        y_train = subsampler(X['y_train'], X['train_indices'])
        X_new = np.ones((X_train.shape[0], 1))
        return super(DummyRegressionPipeline, self).fit(X_new, y_train, sample_weight=sample_weight)

    def predict(self, X: Union[np.ndarray, pd.DataFrame], batch_size: int = 1000) -> np.ndarray:
        new_X = np.ones((X.shape[0], 1))
        return super(DummyRegressionPipeline, self).predict(new_X).astype(np.float32)
