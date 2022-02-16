from typing import Any, Dict, Optional

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
from autoPyTorch.pipeline.components.preprocessing.tabular_preprocessing.feature_preprocessing. \
    utils import percentage_value_range_to_integer_range
from autoPyTorch.utils.common import HyperparameterSearchSpace, add_hyperparameter


class RandomKitchenSinks(autoPyTorchFeaturePreprocessingComponent):
    """
    Approximate a RBF kernel feature map using random Fourier features.

    Args:
        n_components (int):
            Number of Monte Carlo samples per original feature.
            Equals the dimensionality of the computed feature space.
            Note:
                This number needs to be less than the total number of
                features. To keep the hyperparameter search space general
                to different datasets, autoPyTorch defines its value
                range as the percentage of the number of features (in float).
                This is then used to construct the range of n_components using
                n_components = percentage of features * number of features.
                Defaults to 100.
        gamma (float):
            Parameter of RBF kernel: exp(-gamma * x^2). Defaults to 1.0.
    """
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

        n_components = percentage_value_range_to_integer_range(
            hyperparameter_search_space=n_components,
            default_value_range=(10, 2000),
            default_value=100,
            dataset_properties=dataset_properties,
        )

        add_hyperparameter(cs, n_components, UniformIntegerHyperparameter)

        add_hyperparameter(cs, gamma, UniformFloatHyperparameter)
        return cs

    @staticmethod
    def get_properties(dataset_properties: Optional[Dict[str, BaseDatasetPropertiesType]] = None) -> Dict[str, Any]:
        return {'shortname': 'KitchenSink',
                'name': 'Random Kitchen Sinks',
                'handles_sparse': True,
                'handles_classification': True,
                'handles_regression': True
                }
