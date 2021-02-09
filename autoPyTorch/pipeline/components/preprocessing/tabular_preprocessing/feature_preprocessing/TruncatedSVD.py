from math import floor, ceil

from typing import Any, Dict, Optional, Tuple, Union

from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import (
    UniformIntegerHyperparameter,
)

import numpy as np

import sklearn.decomposition
from sklearn.base import BaseEstimator

from autoPyTorch.pipeline.components.preprocessing.tabular_preprocessing.feature_preprocessing\
    .base_feature_preprocessor import autoPyTorchFeaturePreprocessingComponent


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
        target_dim: Tuple[Tuple, float] = ((0.5, 0.9), 0.5),
    ) -> ConfigurationSpace:
        cs = ConfigurationSpace()

        if dataset_properties is not None:
            n_features = len(dataset_properties['numerical_columns'])
            target_dim = ((floor(target_dim[0][0] * n_features), ceil(target_dim[0][1] * n_features)),
                          floor(target_dim[1] * n_features))
        else:
            target_dim = ((10, 256), 128)
        target_dim = UniformIntegerHyperparameter("target_dim", lower=target_dim[0][0],
                                                  upper=target_dim[0][1], default_value=target_dim[1])
        cs.add_hyperparameters([target_dim])

        return cs
