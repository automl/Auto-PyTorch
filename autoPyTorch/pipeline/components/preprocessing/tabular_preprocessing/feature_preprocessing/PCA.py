from typing import Any, Dict, Optional

from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import (
    CategoricalHyperparameter,
    UniformFloatHyperparameter,
)

import numpy as np

import sklearn.decomposition
from sklearn.base import BaseEstimator

from autoPyTorch.datasets.base_dataset import BaseDatasetPropertiesType
from autoPyTorch.pipeline.components.preprocessing.tabular_preprocessing.feature_preprocessing. \
    base_feature_preprocessor import autoPyTorchFeaturePreprocessingComponent
from autoPyTorch.utils.common import FitRequirement, HyperparameterSearchSpace, add_hyperparameter


class PCA(autoPyTorchFeaturePreprocessingComponent):
    def __init__(self, keep_variance: float = 0.9999,
                 whiten: bool = False,
                 random_state: Optional[np.random.RandomState] = None
                 ):
        self.keep_variance = keep_variance
        self.whiten = whiten
        super().__init__(random_state=random_state)

        self.add_fit_requirements([
            FitRequirement('issparse', (bool,), user_defined=True, dataset_property=True)])

    def fit(self, X: Dict[str, Any], y: Any = None) -> BaseEstimator:
        self.check_requirements(X, y)

        n_components = float(self.keep_variance)
        self.preprocessor['numerical'] = sklearn.decomposition.PCA(
            n_components=n_components, whiten=self.whiten,
            random_state=self.random_state)

        return self

    @staticmethod
    def get_hyperparameter_search_space(
        dataset_properties: Optional[Dict[str, BaseDatasetPropertiesType]] = None,
        keep_variance: HyperparameterSearchSpace = HyperparameterSearchSpace(hyperparameter='keep_variance',
                                                                             value_range=(0.5, 0.9999),
                                                                             default_value=0.9999,
                                                                             log=True),
        whiten: HyperparameterSearchSpace = HyperparameterSearchSpace(hyperparameter='whiten',
                                                                      value_range=(True, False),
                                                                      default_value=False,
                                                                      ),
    ) -> ConfigurationSpace:

        cs = ConfigurationSpace()

        add_hyperparameter(cs, keep_variance, UniformFloatHyperparameter)
        add_hyperparameter(cs, whiten, CategoricalHyperparameter)

        return cs

    @staticmethod
    def get_properties(dataset_properties: Optional[Dict[str, BaseDatasetPropertiesType]] = None) -> Dict[str, Any]:
        return {'shortname': 'PCA',
                'name': 'Principal Component Analysis',
                'handles_sparse': False,
                'handles_classification': True,
                'handles_regression': True
                }
