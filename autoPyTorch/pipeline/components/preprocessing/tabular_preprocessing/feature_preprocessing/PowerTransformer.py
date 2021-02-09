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
            standardize: Tuple[Tuple, bool] = ((True, False), True)
    ) -> ConfigurationSpace:
        standardize = CategoricalHyperparameter("standardize",
                                                choices=standardize[0],
                                                default_value=standardize[1])

        cs = ConfigurationSpace()
        cs.add_hyperparameters([standardize])

        return cs
