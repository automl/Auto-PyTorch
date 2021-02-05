from typing import Any, Dict, Optional, Tuple, Union

from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import (
    UniformFloatHyperparameter,
    UniformIntegerHyperparameter,
)

import numpy as np

import sklearn.kernel_approximation
from sklearn.base import BaseEstimator

from autoPyTorch.pipeline.components.preprocessing.tabular_preprocessing.feature_preprocessing.\
    base_feature_preprocessor import autoPyTorchFeaturePreprocessingComponent


class RandomKitchenSinks(autoPyTorchFeaturePreprocessingComponent):
    def __init__(self, n_components: int = 100,
                 gamma: float = 1.0,
                 random_state: Optional[Union[int, np.random.RandomState]] = None
                 ) -> None:
        self.n_components = n_components
        self.gamma = gamma
        self.random_state = random_state
        super().__init__()

    def fit(self, X: Dict[str, Any], y: Any = None) -> BaseEstimator:

        self.preprocessor['numerical'] = sklearn.kernel_approximation.RBFSampler(
            self.gamma, self.n_components, self.random_state)
        return self

    @staticmethod
    def get_hyperparameter_search_space(
        dataset_properties: Optional[Dict[str, str]] = None,
        n_components: Tuple[Tuple, int, bool] = ((10, 2000), 100, True),
        gamma: Tuple[Tuple, float, bool] = ((3.0517578125e-05, 8), 1.0, True),
        degree: Tuple[Tuple, int] = ((2, 5), 3),
        coef0: Tuple[Tuple, float] = ((-1, 1), 0)
    ) -> ConfigurationSpace:
        n_components = UniformIntegerHyperparameter(
            "n_components", lower=n_components[0][0], upper=n_components[0][1],
            default_value=n_components[1], log=n_components[2])
        gamma = UniformFloatHyperparameter(
            "gamma",
            lower=gamma[0][0], upper=gamma[0][1],
            log=gamma[2],
            default_value=gamma[1],
        )
        cs = ConfigurationSpace()
        cs.add_hyperparameters([n_components, gamma])
        return cs

    @staticmethod
    def get_properties(dataset_properties: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        return {'shortname': 'KitchenSink',
                'name': 'Random Kitchen Sinks',
                'handles_sparse': True
                }
