from math import ceil, floor
from typing import Any, Dict, List, Optional

from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import (
    UniformFloatHyperparameter,
    UniformIntegerHyperparameter,
)

import numpy as np

import sklearn.kernel_approximation
from sklearn.base import BaseEstimator

from autoPyTorch.datasets.base_dataset import BaseDatasetPropertiesType
from autoPyTorch.pipeline.components.preprocessing.tabular_preprocessing.feature_preprocessing. \
    base_feature_preprocessor import autoPyTorchFeaturePreprocessingComponent
from autoPyTorch.utils.common import HyperparameterSearchSpace, add_hyperparameter


class RandomKitchenSinks(autoPyTorchFeaturePreprocessingComponent):
    def __init__(self, n_components: int = 100,
                 gamma: float = 1.0,
                 random_state: Optional[np.random.RandomState] = None
                 ):
        self.n_components = n_components
        self.gamma = gamma
        super().__init__(random_state=random_state)

    def fit(self, X: Dict[str, Any], y: Any = None) -> BaseEstimator:

        self.preprocessor['numerical'] = sklearn.kernel_approximation.RBFSampler(
            self.gamma, self.n_components, self.random_state)
        return self

    @staticmethod
    def get_hyperparameter_search_space(
        dataset_properties: Optional[Dict[str, BaseDatasetPropertiesType]] = None,
        n_components: HyperparameterSearchSpace = HyperparameterSearchSpace(hyperparameter='n_components',
                                                                            value_range=(0.5, 0.9),
                                                                            default_value=0.5,
                                                                            ),
        gamma: HyperparameterSearchSpace = HyperparameterSearchSpace(hyperparameter='gamma',
                                                                     value_range=(3.0517578125e-05, 8),
                                                                     default_value=1.0,
                                                                     log=True),
    ) -> ConfigurationSpace:
        cs = ConfigurationSpace()

        if dataset_properties is not None:
            n_features = len(dataset_properties['numerical_columns']) \
                if isinstance(dataset_properties['numerical_columns'], List) else 0
            if n_features == 1:
                log = False
            else:
                log = n_components.log
            n_components = HyperparameterSearchSpace(hyperparameter='n_components',
                                                     value_range=(
                                                         floor(float(n_components.value_range[0]) * n_features),
                                                         ceil(float(n_components.value_range[1]) * n_features)),
                                                     default_value=ceil(float(n_components.default_value) * n_features),
                                                     log=log)
        else:
            n_components = HyperparameterSearchSpace(hyperparameter='n_components',
                                                     value_range=(10, 2000),
                                                     default_value=100,
                                                     log=n_components.log)

        add_hyperparameter(cs, n_components, UniformIntegerHyperparameter)

        add_hyperparameter(cs, gamma, UniformFloatHyperparameter)
        return cs

    @staticmethod
    def get_properties(dataset_properties: Optional[Dict[str, BaseDatasetPropertiesType]] = None) -> Dict[str, Any]:
        return {'shortname': 'KitchenSink',
                'name': 'Random Kitchen Sinks',
                'handles_sparse': True
                }
