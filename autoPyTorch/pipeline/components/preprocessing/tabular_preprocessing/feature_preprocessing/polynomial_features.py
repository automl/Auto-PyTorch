from typing import Any, Dict, Optional, Tuple, Union

from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import (
    CategoricalHyperparameter,
    UniformIntegerHyperparameter,
)

import numpy as np

import sklearn.decomposition
from sklearn.base import BaseEstimator

from autoPyTorch.pipeline.components.preprocessing.tabular_preprocessing.feature_preprocessing.\
    base_feature_preprocessor import autoPyTorchFeaturePreprocessingComponent


class PolynomialFeatures(autoPyTorchFeaturePreprocessingComponent):
    def __init__(self, degree: int = 2, interaction_only: bool = False,
                 include_bias: bool = False,
                 random_state: Optional[Union[int, np.random.RandomState]] = None):
        self.degree = degree
        self.interaction_only = interaction_only
        self.include_bias = include_bias

        self.random_state = random_state
        super().__init__()

    def fit(self, X: Dict[str, Any], y: Any = None) -> BaseEstimator:

        self.preprocessor['numerical'] = sklearn.preprocessing.PolynomialFeatures(
            degree=self.degree, interaction_only=self.interaction_only,
            include_bias=self.include_bias)
        return self

    @staticmethod
    def get_properties(dataset_properties: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        return {'shortname': 'PolynomialFeatures',
                'name': 'PolynomialFeatures',
                'handles_sparse': True}

    @staticmethod
    def get_hyperparameter_search_space(
        dataset_properties: Optional[Dict[str, str]] = None,
        degree: Tuple[Tuple, int] = ((2, 3), 2),
        intersection_only: Tuple[Tuple, bool] = ((True, False), False),
        include_bias: Tuple[Tuple, bool] = ((True, False), False)
    ) -> ConfigurationSpace:

        degree = UniformIntegerHyperparameter("degree", lower=degree[0][0], upper=degree[0][1], default_value=degree[1])
        interaction_only = CategoricalHyperparameter("interaction_only",
                                                     choices=intersection_only[0],
                                                     default_value=intersection_only[1])
        include_bias = CategoricalHyperparameter("include_bias",
                                                 choices=include_bias[0],
                                                 default_value=include_bias[1])

        cs = ConfigurationSpace()
        cs.add_hyperparameters([degree, interaction_only, include_bias])

        return cs
