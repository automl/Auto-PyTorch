import warnings
from typing import Any, Dict, Optional

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

from autoPyTorch.datasets.base_dataset import BaseDatasetPropertiesType
from autoPyTorch.pipeline.components.preprocessing.tabular_preprocessing.feature_preprocessing. \
    base_feature_preprocessor import autoPyTorchFeaturePreprocessingComponent
from autoPyTorch.pipeline.components.preprocessing.tabular_preprocessing.feature_preprocessing. \
    utils import percentage_value_range_to_integer_range
from autoPyTorch.utils.common import FitRequirement, HyperparameterSearchSpace, add_hyperparameter, get_hyperparameter


class Nystroem(autoPyTorchFeaturePreprocessingComponent):
    """
    Construct an approximate feature map for an arbitrary kernel using a subset of the data as basis.

    Args:
        n_components (int):
            Note:
                This number needs to be less than the total number of
                features. To keep the hyperparameter search space general
                to different datasets, autoPyTorch defines its value
                range as the percentage of the number of features (in float).
                This is then used to construct the range of n_components using
                n_components = percentage of features * number of features. Defaults to 10.
        kernel (str):
            Kernel map to be approximated. Defaults to 'rbf'.
        degree (int):
            Degree of the polynomial kernel. Defaults to 3.
        gamma (float):
            Gamma parameter for the RBF, laplacian, polynomial, exponential chi2 and
            sigmoid kernels. Defaults to 0.01.
        coef0 (float):
            Zero coefficient for polynomial and sigmoid kernels. Defaults to 0.0.
    """
    def __init__(self, n_components: int = 10,
                 kernel: str = 'rbf', degree: int = 3,
                 gamma: float = 0.01, coef0: float = 0.0,
                 random_state: Optional[np.random.RandomState] = None
                 ):
        self.n_components = n_components
        self.kernel = kernel
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
        super().__init__(random_state=random_state)
        self.add_fit_requirements([
            FitRequirement('issigned', (bool,), user_defined=True, dataset_property=True)])

    def fit(self, X: Dict[str, Any], y: Any = None) -> BaseEstimator:

        self.preprocessor['numerical'] = sklearn.kernel_approximation.Nystroem(
            n_components=self.n_components, kernel=self.kernel,
            degree=self.degree, gamma=self.gamma, coef0=self.coef0,
            random_state=self.random_state)

        return self

    @staticmethod
    def get_hyperparameter_search_space(
        dataset_properties: Optional[Dict[str, BaseDatasetPropertiesType]] = None,
        n_components: HyperparameterSearchSpace = HyperparameterSearchSpace(hyperparameter='n_components',
                                                                            value_range=(0.5, 0.9),
                                                                            default_value=0.5,
                                                                            ),
        kernel: HyperparameterSearchSpace = HyperparameterSearchSpace(hyperparameter='kernel',
                                                                      value_range=('poly',
                                                                                   'rbf',
                                                                                   'sigmoid',
                                                                                   'cosine',
                                                                                   'chi2'),
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

        n_components = percentage_value_range_to_integer_range(
            hyperparameter_search_space=n_components,
            default_value_range=(10, 2000),
            default_value=100,
            dataset_properties=dataset_properties,
        )

        add_hyperparameter(cs, n_components, UniformIntegerHyperparameter)
        value_range = list(kernel.value_range)

        allow_chi = True

        if dataset_properties is not None:
            if (
                dataset_properties.get("issigned")
                or dataset_properties.get("issparse")
            ):
                # chi kernel does not support negative numbers or
                # a sparse matrix
                allow_chi = False
            else:
                allow_chi = True
        if not allow_chi:
            value_range = [value for value in value_range if value != "chi2"]
            if len(value_range) == 0:
                value_range = ["poly"]

        if value_range != list(kernel.value_range):
            warnings.warn(f"Given choices for `score_func` are not compatible with the dataset. "
                          f"Updating choices to {value_range}")

        kernel = HyperparameterSearchSpace(hyperparameter='kernel',
                                           value_range=value_range,
                                           default_value=value_range[-1],
                                           )

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
    def get_properties(dataset_properties: Optional[Dict[str, BaseDatasetPropertiesType]] = None) -> Dict[str, Any]:
        return {'shortname': 'Nystroem',
                'name': 'Nystroem kernel approximation',
                'handles_sparse': True,
                'handles_classification': True,
                'handles_regression': True,
                'handles_signed': True
                }
