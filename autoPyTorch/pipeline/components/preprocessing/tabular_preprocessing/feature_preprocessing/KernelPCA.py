from typing import Any, Dict, Optional, Tuple, Union

from ConfigSpace.conditions import EqualsCondition, InCondition
from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import (
    CategoricalHyperparameter,
    UniformFloatHyperparameter,
    UniformIntegerHyperparameter,
)

import numpy as np

import sklearn.decomposition
from sklearn.base import BaseEstimator

from autoPyTorch.pipeline.components.preprocessing.tabular_preprocessing.feature_preprocessing.\
    base_feature_preprocessor import autoPyTorchFeaturePreprocessingComponent
from autoPyTorch.utils.common import FitRequirement


class KernelPCA(autoPyTorchFeaturePreprocessingComponent):
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

        self.add_fit_requirements([
            FitRequirement('issparse', (bool,), user_defined=True, dataset_property=True)])

    def fit(self, X: Dict[str, Any], y: Any = None) -> BaseEstimator:

        self.preprocessor['numerical'] = sklearn.decomposition.KernelPCA(
            n_components=self.n_components, kernel=self.kernel,
            degree=self.degree, gamma=self.gamma, coef0=self.coef0,
            remove_zero_eig=True, random_state=self.random_state)

        return self

    @staticmethod
    def get_hyperparameter_search_space(
        dataset_properties: Optional[Dict[str, str]] = None,
        n_components: Tuple[Tuple, float] = ((0.5, 0.9), 0.5),
        kernel: Tuple[Tuple, str] = (('poly', 'rbf', 'sigmoid', 'cosine'), 'rbf'),
        gamma: Tuple[Tuple, float, bool] = ((3.0517578125e-05, 8), 0.01, True),
        degree: Tuple[Tuple, int] = ((2, 5), 3),
        coef0: Tuple[Tuple, float] = ((-1, 1), 0)
    ) -> ConfigurationSpace:

        if dataset_properties is not None:
            n_features = len(dataset_properties['numerical_columns'])
            n_components = ((int(n_components[0][0] * n_features), int(n_components[0][1] * n_features)),
                            int(n_components[1] * n_features))
        else:
            n_components = ((10, 2000), 100)

        n_components = UniformIntegerHyperparameter(
            "n_components", lower=n_components[0][0], upper=n_components[0][1], default_value=n_components[1])
        kernel_hp = CategoricalHyperparameter('kernel', choices=kernel[0], default_value=kernel[1])
        gamma = UniformFloatHyperparameter(
            "gamma",
            lower=gamma[0][0], upper=gamma[0][1],
            log=gamma[2],
            default_value=gamma[1],
        )
        coef0 = UniformFloatHyperparameter("coef0", lower=coef0[0][0], upper=coef0[0][1], default_value=coef0[1])
        cs = ConfigurationSpace()
        cs.add_hyperparameters([n_components, kernel_hp, gamma, coef0])

        if "poly" in kernel_hp.choices:
            degree = UniformIntegerHyperparameter('degree', lower=degree[0][0], upper=degree[0][1],
                                                  default_value=degree[1])
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
        return {'shortname': 'KernelPCA',
                'name': 'Kernel Principal Component Analysis',
                'handles_sparse': True
                }
