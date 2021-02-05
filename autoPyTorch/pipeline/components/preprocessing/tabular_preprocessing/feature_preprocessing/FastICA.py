from typing import Any, Dict, Optional, Tuple, Union

from ConfigSpace.conditions import EqualsCondition
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


class FastICA(autoPyTorchFeaturePreprocessingComponent):
    def __init__(self, n_components: int = 100,
                 algorithm: str = 'parallel',
                 whiten: bool = False,
                 fun: str = 'logcosh',
                 random_state: Optional[Union[int, np.random.RandomState]] = None
                 ) -> None:
        self.n_components = n_components
        self.algorithm = algorithm
        self.whiten = whiten
        self.fun = fun
        self.random_state = random_state

        super().__init__()

    def fit(self, X: Dict[str, Any], y: Any = None) -> BaseEstimator:

        self.preprocessor['numerical'] = sklearn.decomposition.FastICA(
            n_components=self.n_components, algorithm=self.algorithm,
            fun=self.fun, whiten=self.whiten, random_state=self.random_state
        )

        return self

    @staticmethod
    def get_hyperparameter_search_space(
        dataset_properties: Optional[Dict[str, str]] = None,
        n_components: Tuple[Tuple, int] = ((10, 2000), 100),
        algorithm: Tuple[Tuple, str] = (('parallel', 'deflation'), 'parallel'),
        whiten: Tuple[Tuple, bool] = ((True, False), False),
        fun: Tuple[Tuple, str] = (('logcosh', 'exp', 'cube'), 'logcosh')
    ) -> ConfigurationSpace:
        n_components = UniformIntegerHyperparameter(
            "n_components", lower=n_components[0][0], upper=n_components[0][1], default_value=n_components[1])
        algorithm = CategoricalHyperparameter('algorithm', choices=algorithm[0], default_value=algorithm[1])
        whiten = CategoricalHyperparameter('whiten', choices=whiten[0], default_value=whiten[1])
        fun = CategoricalHyperparameter('fun', choices=fun[0], default_value=fun[1])
        cs = ConfigurationSpace()
        cs.add_hyperparameters([n_components, algorithm, whiten, fun])

        cs.add_condition(EqualsCondition(n_components, whiten, True))

        return cs

    @staticmethod
    def get_properties(dataset_properties: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        return {'shortname': 'FastICA',
                'name': 'Fast Independent Component Analysis',
                'handles_sparse': True
                }
