from typing import Any, Dict, Optional

from ConfigSpace.conditions import EqualsCondition
from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import (
    CategoricalHyperparameter,
    UniformIntegerHyperparameter,
)

import numpy as np

from sklearn.base import BaseEstimator
from sklearn.decomposition import FastICA as SklearnFastICA

from autoPyTorch.datasets.base_dataset import BaseDatasetPropertiesType
from autoPyTorch.pipeline.components.preprocessing.tabular_preprocessing.feature_preprocessing. \
    base_feature_preprocessor import autoPyTorchFeaturePreprocessingComponent
from autoPyTorch.pipeline.components.preprocessing.tabular_preprocessing.feature_preprocessing. \
    utils import percentage_value_range_to_integer_range
from autoPyTorch.utils.common import HyperparameterSearchSpace, add_hyperparameter, get_hyperparameter


class FastICA(autoPyTorchFeaturePreprocessingComponent):
    """
    Reduce number of features by separating a multivariate signal into
    additive subcomponents that are maximally independent.

    Args:
        n_components (int):
            Number of components to use
            Note:
                This number needs to be less than the total number of
                features. To keep the hyperparameter search space general
                to different datasets, autoPyTorch defines its value
                range as the percentage of the number of features (in float).
                This is then used to construct the range of n_components using
                n_components = percentage of features * number of features.
                Defaults to 100.
        algorithm (str):
            Apply parallel or deflational algorithm for FastICA.
            Defaults to 'parallel'.
        whiten (bool):
            If whiten is false, the data is already considered to be whitened,
            and no whitening is performed. Defaults to False.
        fun (str):
            The functional form of the G function used in the approximation to neg-entropy.
            Defaults to "logcosh".
    """
    def __init__(self, n_components: int = 100,
                 algorithm: str = 'parallel', whiten: bool = False,
                 fun: str = "logcosh",
                 random_state: Optional[np.random.RandomState] = None
                 ):
        self.n_components = n_components
        self.algorithm = algorithm
        self.whiten = whiten
        self.fun = fun

        super().__init__(random_state=random_state)

    def fit(self, X: Dict[str, Any], y: Any = None) -> BaseEstimator:

        self.check_requirements(X, y)

        self.preprocessor['numerical'] = SklearnFastICA(
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
                                                                   default_value='logcosh',
                                                                   ),
    ) -> ConfigurationSpace:
        n_components = percentage_value_range_to_integer_range(
            hyperparameter_search_space=n_components,
            default_value_range=(10, 2000),
            default_value=100,
            dataset_properties=dataset_properties,
        )
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
