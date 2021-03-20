from typing import Any, Dict, Optional, Tuple, Union

from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import (
    CategoricalHyperparameter,
)

import numpy as np

import sklearn.preprocessing
from sklearn.base import BaseEstimator

from autoPyTorch.pipeline.components.preprocessing.tabular_preprocessing.feature_preprocessing. \
    base_feature_preprocessor import autoPyTorchFeaturePreprocessingComponent
from autoPyTorch.utils.common import HyperparameterSearchSpace, add_hyperparameter


class PowerTransformer(autoPyTorchFeaturePreprocessingComponent):
    def __init__(self, standardize: bool = True,
                 random_state: Optional[Union[int, np.random.RandomState]] = None):
        self.standardize = standardize

        self.random_state = random_state
        super().__init__()

    def fit(self, X: Dict[str, Any], y: Any = None) -> BaseEstimator:
        self.preprocessor['numerical'] = sklearn.preprocessing.PowerTransformer(method="yeo-johnson",
                                                                                standardize=self.standardize,
                                                                                copy=False)
        return self

    @staticmethod
    def get_properties(dataset_properties: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        return {'shortname': 'PowerTransformer',
                'name': 'Power Transformer',
                'handles_sparse': True}

    @staticmethod
    def get_hyperparameter_search_space(
            dataset_properties: Optional[Dict[str, str]] = None,
            standardize: HyperparameterSearchSpace = HyperparameterSearchSpace(hyperparameter='standardize',
                                                                     value_range=(True, False),
                                                                     default_value=True,
                                                                     log=False),
    ) -> ConfigurationSpace:

        cs = ConfigurationSpace()
        add_hyperparameter(cs, standardize, CategoricalHyperparameter)

        return cs
