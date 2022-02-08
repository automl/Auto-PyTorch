from functools import partial
from math import ceil, floor
from typing import Any, Callable, Dict, List, Optional, Union

from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import (
    CategoricalHyperparameter,
    UniformIntegerHyperparameter,
)

import numpy as np

from sklearn.feature_selection import SelectPercentile, chi2, f_classif, mutual_info_classif
from sklearn.base import BaseEstimator

from autoPyTorch.datasets.base_dataset import BaseDatasetPropertiesType
from autoPyTorch.pipeline.components.preprocessing.tabular_preprocessing.feature_preprocessing. \
    base_feature_preprocessor import autoPyTorchFeaturePreprocessingComponent
from autoPyTorch.utils.common import HyperparameterSearchSpace, add_hyperparameter


class SelectPercentileClassification(autoPyTorchFeaturePreprocessingComponent):
    def __init__(self, score_func: str = "chi2",
                 percentile: int = 50, 
                 random_state: Optional[np.random.RandomState] = None
                 ):
        self.percentile = percentile
        if score_func == "chi2":
            self.score_func = chi2
        elif score_func == "f_classif":
            self.score_func = f_classif
        elif score_func == "mutual_info":
            self.score_func = partial(mutual_info_classif, random_state=self.random_state)
        else:
            raise ValueError("score_func must be in ('chi2, 'f_classif', 'mutual_info'), "
                             "but is: %s" % score_func)

        super().__init__(random_state=random_state)

    def fit(self, X: Dict[str, Any], y: Any = None) -> BaseEstimator:

        self.preprocessor['numerical'] = SelectPercentile(
            percentile=self.percentile, score_func=self.score_func)

        return self

    @staticmethod
    def get_hyperparameter_search_space(
        dataset_properties: Optional[Dict[str, BaseDatasetPropertiesType]] = None,
        percentile: HyperparameterSearchSpace = HyperparameterSearchSpace(hyperparameter="percentile",
                                                                            value_range=(1, 99),
                                                                            default_value=50,
                                                                            ),
        score_func: HyperparameterSearchSpace = HyperparameterSearchSpace(hyperparameter="score_func",
                                                                         value_range=("chi2", "f_classif", "mutual_info"),
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
        add_hyperparameter(cs, percentile, UniformIntegerHyperparameter)

        return cs


    @staticmethod
    def get_properties(dataset_properties: Optional[Dict[str, BaseDatasetPropertiesType]] = None) -> Dict[str, Any]:
        return {'shortname': 'SPC',
                'name': 'Select Percentile Classification',
                'handles_sparse': True,
                'handles_regression': False,
                'handles_classification': True
                }
