from typing import Any, Dict, Optional, Union

from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import (
    CategoricalHyperparameter,
    UniformIntegerHyperparameter,
)

import numpy as np

import sklearn.decomposition
from sklearn.base import BaseEstimator

from autoPyTorch.pipeline.components.preprocessing.tabular_preprocessing.feature_preprocessing. \
    base_feature_preprocessor import autoPyTorchFeaturePreprocessingComponent
from autoPyTorch.utils.common import HyperparameterSearchSpace, add_hyperparameter


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
        degree: HyperparameterSearchSpace = HyperparameterSearchSpace(hyperparameter='degree',
                                                                      value_range=(2, 3),
                                                                      default_value=2,
                                                                      log=True),
        interaction_only: HyperparameterSearchSpace = HyperparameterSearchSpace(hyperparameter='interaction_only',
                                                                                value_range=(True, False),
                                                                                default_value=False,
                                                                                ),
        include_bias: HyperparameterSearchSpace = HyperparameterSearchSpace(hyperparameter='include_bias',
                                                                            value_range=(True, False),
                                                                            default_value=False,
                                                                            ),
    ) -> ConfigurationSpace:
        cs = ConfigurationSpace()
        add_hyperparameter(cs, degree, UniformIntegerHyperparameter)
        add_hyperparameter(cs, interaction_only, CategoricalHyperparameter)
        add_hyperparameter(cs, include_bias, CategoricalHyperparameter)

        return cs
