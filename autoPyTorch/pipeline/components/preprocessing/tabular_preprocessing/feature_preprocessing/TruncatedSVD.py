from math import floor
from typing import Any, Dict, Optional, Union

from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import (
    UniformIntegerHyperparameter,
)

import numpy as np

import sklearn.decomposition
from sklearn.base import BaseEstimator

from autoPyTorch.pipeline.components.preprocessing.tabular_preprocessing.feature_preprocessing \
    .base_feature_preprocessor import autoPyTorchFeaturePreprocessingComponent
from autoPyTorch.utils.common import HyperparameterSearchSpace, add_hyperparameter


class TruncatedSVD(autoPyTorchFeaturePreprocessingComponent):
    def __init__(self, target_dim: int = 128,
                 random_state: Optional[Union[int, np.random.RandomState]] = None):
        self.target_dim = target_dim

        self.random_state = random_state
        super().__init__()

    def fit(self, X: Dict[str, Any], y: Any = None) -> BaseEstimator:

        self.preprocessor['numerical'] = sklearn.decomposition.TruncatedSVD(self.target_dim, algorithm="randomized")

        return self

    @staticmethod
    def get_properties(dataset_properties: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        return {'shortname': 'TruncSVD',
                'name': 'Truncated Singular Value Decomposition',
                'handles_sparse': True}

    @staticmethod
    def get_hyperparameter_search_space(
        dataset_properties: Optional[Dict[str, str]] = None,
        target_dim: HyperparameterSearchSpace = HyperparameterSearchSpace(hyperparameter='target_dim',
                                                                          value_range=(0.5, 0.9),
                                                                          default_value=0.5,
                                                                          ),
    ) -> ConfigurationSpace:
        cs = ConfigurationSpace()

        if dataset_properties is not None:
            n_features = len(dataset_properties['numerical_columns'])
            target_dim = HyperparameterSearchSpace(hyperparameter=target_dim.hyperparameter,
                                                   value_range=(floor(float(target_dim.value_range[0]) * n_features),
                                                                floor(float(target_dim.value_range[1]) * n_features)),
                                                   default_value=floor(float(target_dim.default_value) * n_features),
                                                   log=target_dim.log)
        else:
            target_dim = HyperparameterSearchSpace(hyperparameter=target_dim.hyperparameter,
                                                   value_range=(10, 256),
                                                   default_value=128,
                                                   log=target_dim.log)

        add_hyperparameter(cs, target_dim, UniformIntegerHyperparameter)
        return cs
