from functools import partial
from math import ceil, floor
from typing import Any, Callable, Dict, List, Optional, Union

from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import (
    CategoricalHyperparameter,
    UniformFloatHyperparameter,
)

import numpy as np

from sklearn.feature_selection import GenericUnivariateSelect, chi2, f_classif, mutual_info_classif
from sklearn.base import BaseEstimator

from autoPyTorch.datasets.base_dataset import BaseDatasetPropertiesType
from autoPyTorch.pipeline.components.preprocessing.tabular_preprocessing.feature_preprocessing. \
    base_feature_preprocessor import autoPyTorchFeaturePreprocessingComponent
from autoPyTorch.utils.common import HyperparameterSearchSpace, add_hyperparameter


class SelectRatesClassification(autoPyTorchFeaturePreprocessingComponent):
    def __init__(self, alpha: float = 0.1,
                 score_func: str = "chi2",
                 mode: str = "fpr",
                 random_state: Optional[np.random.RandomState] = None
                 ):
        self.mode = mode
        self.alpha = alpha
        if score_func == "chi2":
            self.score_func = chi2
        elif score_func == "f_classif":
            self.score_func = f_classif
        else:
            raise ValueError("score_func must be in ('chi2, 'f_classif'), "
                             "but is: %s" % score_func)

        super().__init__(random_state=random_state)

    def fit(self, X: Dict[str, Any], y: Any = None) -> BaseEstimator:

        self.preprocessor['numerical'] = GenericUnivariateSelect(
            mode=self.mode, score_func=self.score_func, param=self.alpha)

        return self

    @staticmethod
    def get_hyperparameter_search_space(
        dataset_properties: Optional[Dict[str, BaseDatasetPropertiesType]] = None,
        alpha: HyperparameterSearchSpace = HyperparameterSearchSpace(hyperparameter="alpha",
                                                                     value_range=(0.01, 0.5),
                                                                     default_value=0.1,
                                                                     ),
        mode: HyperparameterSearchSpace = HyperparameterSearchSpace(hyperparameter="mode",
                                                                    value_range=('fpr', 'fdr', 'fwe'),
                                                                    default_value='fpr',
                                                                    ),
        score_func: HyperparameterSearchSpace = HyperparameterSearchSpace(hyperparameter="score_func",
                                                                          value_range=("chi2", "f_classif"),
                                                                          default_value="chi2",
                                                                          ),
    ) -> ConfigurationSpace:
        if dataset_properties is not None:
            if 'issparse' in dataset_properties and dataset_properties['issparse']:
                score_func = HyperparameterSearchSpace(hyperparameter="score_func",
                                                       value_range=("chi2",),
                                                       default_value="chi2",
                                                       )
        cs = ConfigurationSpace()

        add_hyperparameter(cs, score_func, CategoricalHyperparameter)
        add_hyperparameter(cs, alpha, UniformFloatHyperparameter)
        add_hyperparameter(cs, mode, CategoricalHyperparameter)

        return cs


    @staticmethod
    def get_properties(dataset_properties: Optional[Dict[str, BaseDatasetPropertiesType]] = None) -> Dict[str, Any]:
        return {'shortname': 'SRC',
                'name': 'Select Rates Classification',
                'handles_sparse': True,
                'handles_regression': False,
                'handles_classification': True
                }
