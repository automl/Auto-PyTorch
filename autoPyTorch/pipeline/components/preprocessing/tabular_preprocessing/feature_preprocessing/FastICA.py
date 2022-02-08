from math import ceil, floor
from typing import Any, Callable, Dict, List, Optional, Union

from ConfigSpace.conditions import EqualsCondition
from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import (
    CategoricalHyperparameter,
    UniformIntegerHyperparameter,
)

import numpy as np

from sklearn.decomposition import FastICA
from sklearn.base import BaseEstimator

from autoPyTorch.datasets.base_dataset import BaseDatasetPropertiesType
from autoPyTorch.pipeline.components.preprocessing.tabular_preprocessing.feature_preprocessing. \
    base_feature_preprocessor import autoPyTorchFeaturePreprocessingComponent
from autoPyTorch.utils.common import HyperparameterSearchSpace, add_hyperparameter, get_hyperparameter


class FastICA(autoPyTorchFeaturePreprocessingComponent):
    def __init__(self, n_components: Optional[int] = None,
                 algorithm: str = 'euclidean', whiten: str = 'ward',
                 fun: str = "max",
                 random_state: Optional[np.random.RandomState] = None
                 ):
        self.n_components = n_components
        self.algorithm = algorithm
        self.whiten = whiten
        self.fun = fun

        super().__init__(random_state=random_state)

    def fit(self, X: Dict[str, Any], y: Any = None) -> BaseEstimator:

        self.preprocessor['numerical'] = FastICA(
            n_components=self.n_components, algorithm=self.algorithm,
            fun=self.fun, whiten=self.whiten, random_state=self.random_state)

        return self

    @staticmethod
    def get_hyperparameter_search_space(
        dataset_properties: Optional[Dict[str, BaseDatasetPropertiesType]] = None,
        n_components: HyperparameterSearchSpace = HyperparameterSearchSpace(hyperparameter='n_components',
                                                                            value_range=(0.5, 0.9),
                                                                            default_value=0.5,
                                                                            ),
        algorithm: HyperparameterSearchSpace = HyperparameterSearchSpace(hyperparameter='algorithm',
                                                                         value_range=('parallel', 'deflation'),
                                                                         default_value='parallel',
                                                                         ),
        whiten: HyperparameterSearchSpace = HyperparameterSearchSpace(hyperparameter='whiten',
                                                                      value_range=(True, False),
                                                                      default_value=False,
                                                                      ),
        fun: HyperparameterSearchSpace = HyperparameterSearchSpace(hyperparameter='fun',
                                                                   value_range=('logcosh', 'exp', 'cube'),
                                                                   default_value='logcash',
                                                                   ),
    ) -> ConfigurationSpace:
        if dataset_properties is not None:
            n_features = len(dataset_properties['numerical_columns']) if isinstance(
                dataset_properties['numerical_columns'], List) else 0
            if n_features == 1:
                log = False
            else:
                log = n_components.log
            n_components = HyperparameterSearchSpace(hyperparameter='n_components',
                                                     value_range=(
                                                         floor(float(n_components.value_range[0]) * n_features),
                                                         ceil(float(n_components.value_range[1]) * n_features)),
                                                     default_value=ceil(float(n_components.default_value) * n_features),
                                                     log=log)
        else:
            n_components = HyperparameterSearchSpace(hyperparameter='n_components',
                                                     value_range=(10, 2000),
                                                     default_value=100,
                                                     log=n_components.log)
        cs = ConfigurationSpace()

        n_components_hp = get_hyperparameter(n_components, UniformIntegerHyperparameter)
        whiten_hp = get_hyperparameter(whiten, CategoricalHyperparameter)
        add_hyperparameter(cs, algorithm, CategoricalHyperparameter)
        add_hyperparameter(cs, fun, CategoricalHyperparameter)
        cs.add_hyperparameter(whiten_hp)

        if True in whiten_hp.choices:
            cs.add_hyperparameter(n_components_hp)
            cs.add_condition(EqualsCondition(n_components_hp, whiten_hp, True))

        return cs


    @staticmethod
    def get_properties(dataset_properties: Optional[Dict[str, BaseDatasetPropertiesType]] = None) -> Dict[str, Any]:
        return {'shortname': 'FastICA',
                'name': 'Fast Independent Component Analysis',
                'handles_sparse': False,
                'handles_classification': True,
                'handles_regression': True
                }
