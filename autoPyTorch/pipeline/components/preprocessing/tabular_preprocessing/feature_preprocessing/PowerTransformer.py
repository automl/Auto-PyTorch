from typing import Any, Dict, Optional

from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import (
    CategoricalHyperparameter,
)

import numpy as np

import sklearn.preprocessing
from sklearn.base import BaseEstimator

from autoPyTorch.datasets.base_dataset import BaseDatasetPropertiesType
from autoPyTorch.pipeline.components.preprocessing.tabular_preprocessing.feature_preprocessing. \
    base_feature_preprocessor import autoPyTorchFeaturePreprocessingComponent
from autoPyTorch.utils.common import HyperparameterSearchSpace, add_hyperparameter


class PowerTransformer(autoPyTorchFeaturePreprocessingComponent):
    def __init__(self, standardize: bool = True,
                 random_state: Optional[np.random.RandomState] = None):
        self.standardize = standardize

        super().__init__(random_state=random_state)

    def fit(self, X: Dict[str, Any], y: Any = None) -> BaseEstimator:
        self.preprocessor['numerical'] = sklearn.preprocessing.PowerTransformer(method="yeo-johnson",
                                                                                standardize=self.standardize,
                                                                                copy=False)
        return self

    @staticmethod
    def get_properties(dataset_properties: Optional[Dict[str, BaseDatasetPropertiesType]] = None) -> Dict[str, Any]:
        return {'shortname': 'PowerTransformer',
                'name': 'Power Transformer',
                'handles_sparse': True}

    @staticmethod
    def get_hyperparameter_search_space(
        dataset_properties: Optional[Dict[str, BaseDatasetPropertiesType]] = None,
        standardize: HyperparameterSearchSpace = HyperparameterSearchSpace(hyperparameter='standardize',
                                                                           value_range=(True, False),
                                                                           default_value=True,
                                                                           ),
    ) -> ConfigurationSpace:
        cs = ConfigurationSpace()
        add_hyperparameter(cs, standardize, CategoricalHyperparameter)

        return cs
