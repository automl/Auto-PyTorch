from math import ceil, floor
from typing import Any, Dict, Optional, Union

from ConfigSpace.conditions import EqualsCondition, InCondition
from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import (
    CategoricalHyperparameter,
    UniformFloatHyperparameter,
    UniformIntegerHyperparameter,
)

import numpy as np

import sklearn.kernel_approximation
from sklearn.base import BaseEstimator

from autoPyTorch.pipeline.components.preprocessing.tabular_preprocessing.feature_preprocessing. \
    base_feature_preprocessor import autoPyTorchFeaturePreprocessingComponent
from autoPyTorch.utils.common import HyperparameterSearchSpace, add_hyperparameter, get_hyperparameter


class Nystroem(autoPyTorchFeaturePreprocessingComponent):
    def __init__(self, n_components: int = 10,
                 kernel: str = 'rbf', degree: int = 3,
                 gamma: float = 0.01, coef0: float = 0.0,
                 random_state: Optional[Union[int, np.random.RandomState]] = None
                 ) -> None:
        self.n_components = n_components
        self.kernel = kernel
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
        self.random_state = random_state
        super().__init__()

    def fit(self, X: Dict[str, Any], y: Any = None) -> BaseEstimator:

        self.preprocessor['numerical'] = sklearn.kernel_approximation.Nystroem(
            n_components=self.n_components, kernel=self.kernel,
            degree=self.degree, gamma=self.gamma, coef0=self.coef0,
            random_state=self.random_state)

        return self

    @staticmethod
    def get_hyperparameter_search_space(
        dataset_properties: Optional[Dict[str, str]] = None,
        n_components: HyperparameterSearchSpace = HyperparameterSearchSpace(hyperparameter='n_components',
                                                                            value_range=(0.5, 0.9),
                                                                            default_value=0.5,
                                                                            ),
        kernel: HyperparameterSearchSpace = HyperparameterSearchSpace(hyperparameter='kernel',
                                                                      value_range=('poly', 'rbf', 'sigmoid', 'cosine'),
                                                                      default_value='rbf',
                                                                      ),
        gamma: HyperparameterSearchSpace = HyperparameterSearchSpace(hyperparameter='gamma',
                                                                     value_range=(3.0517578125e-05, 8),
                                                                     default_value=0.01,
                                                                     log=True),
        degree: HyperparameterSearchSpace = HyperparameterSearchSpace(hyperparameter='degree',
                                                                      value_range=(2, 5),
                                                                      default_value=3,
                                                                      log=True),
        coef0: HyperparameterSearchSpace = HyperparameterSearchSpace(hyperparameter='coef0',
                                                                     value_range=(-1, 1),
                                                                     default_value=0,
                                                                     )
    ) -> ConfigurationSpace:

        cs = ConfigurationSpace()

        if dataset_properties is not None:
            n_features = len(dataset_properties['numerical_columns'])
            # if numerical features are 1, set log to False
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

        add_hyperparameter(cs, n_components, UniformIntegerHyperparameter)
        kernel_hp = get_hyperparameter(kernel, CategoricalHyperparameter)
        gamma = get_hyperparameter(gamma, UniformFloatHyperparameter)
        coef0 = get_hyperparameter(coef0, UniformFloatHyperparameter)
        cs.add_hyperparameters([kernel_hp, gamma, coef0])

        if "poly" in kernel_hp.choices:
            degree = get_hyperparameter(degree, UniformIntegerHyperparameter)
            cs.add_hyperparameters([degree])
            degree_depends_on_poly = EqualsCondition(degree, kernel_hp, "poly")
            cs.add_conditions([degree_depends_on_poly])
        kernels = []
        if "sigmoid" in kernel_hp.choices:
            kernels.append("sigmoid")
        if "poly" in kernel_hp.choices:
            kernels.append("poly")
        coef0_condition = InCondition(coef0, kernel_hp, kernels)
        kernels = []
        if "rbf" in kernel_hp.choices:
            kernels.append("rbf")
        if "poly" in kernel_hp.choices:
            kernels.append("poly")
        gamma_condition = InCondition(gamma, kernel_hp, kernels)
        cs.add_conditions([coef0_condition, gamma_condition])
        return cs

    @staticmethod
    def get_properties(dataset_properties: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        return {'shortname': 'Nystroem',
                'name': 'Nystroem kernel approximation',
                'handles_sparse': True
                }
