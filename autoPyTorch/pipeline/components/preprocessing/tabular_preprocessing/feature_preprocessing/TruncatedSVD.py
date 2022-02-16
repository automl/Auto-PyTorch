from typing import Any, Dict, Optional

from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import (
    UniformIntegerHyperparameter,
)

import numpy as np

import sklearn.decomposition
from sklearn.base import BaseEstimator

from autoPyTorch.datasets.base_dataset import BaseDatasetPropertiesType
from autoPyTorch.pipeline.components.preprocessing.tabular_preprocessing.feature_preprocessing \
    .base_feature_preprocessor import autoPyTorchFeaturePreprocessingComponent
from autoPyTorch.pipeline.components.preprocessing.tabular_preprocessing.feature_preprocessing. \
    utils import percentage_value_range_to_integer_range
from autoPyTorch.utils.common import HyperparameterSearchSpace, add_hyperparameter


class TruncatedSVD(autoPyTorchFeaturePreprocessingComponent):
    """
    Linear dimensionality reduction by means of truncated singular value decomposition (SVD).

    Args:
        target_dim (int):
            Desired dimensionality of output data.
            Note:
                This number needs to be less than the total number of
                features. To keep the hyperparameter search space general
                to different datasets, autoPyTorch defines its value
                range as the percentage of the number of features (in float).
                This is then used to construct the range of target_dim using
                target_dim = percentage of features * number of features. Defaults to 128.
    """
    def __init__(self, target_dim: int = 128,
                 random_state: Optional[np.random.RandomState] = None):
        self.target_dim = target_dim

        super().__init__(random_state=random_state)

    def fit(self, X: Dict[str, Any], y: Any = None) -> BaseEstimator:

        self.preprocessor['numerical'] = sklearn.decomposition.TruncatedSVD(self.target_dim, algorithm="randomized",
                                                                            random_state=self.random_state)

        return self

    @staticmethod
    def get_properties(dataset_properties: Optional[Dict[str, BaseDatasetPropertiesType]] = None) -> Dict[str, Any]:
        return {'shortname': 'TruncSVD',
                'name': 'Truncated Singular Value Decomposition',
                'handles_sparse': True,
                'handles_classification': True,
                'handles_regression': True}

    @staticmethod
    def get_hyperparameter_search_space(
        dataset_properties: Optional[Dict[str, BaseDatasetPropertiesType]] = None,
        target_dim: HyperparameterSearchSpace = HyperparameterSearchSpace(hyperparameter='target_dim',
                                                                          value_range=(0.5, 0.9),
                                                                          default_value=0.5,
                                                                          ),
    ) -> ConfigurationSpace:
        cs = ConfigurationSpace()

        target_dim = percentage_value_range_to_integer_range(
            hyperparameter_search_space=target_dim,
            default_value_range=(10, 256),
            default_value=128,
            dataset_properties=dataset_properties,
        )

        add_hyperparameter(cs, target_dim, UniformIntegerHyperparameter)
        return cs
