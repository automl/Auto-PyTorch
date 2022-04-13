from functools import partial
from typing import Any, Dict, Optional

from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import (
    CategoricalHyperparameter,
    UniformIntegerHyperparameter,
)

import numpy as np

from sklearn.base import BaseEstimator
from sklearn.feature_selection import SelectPercentile, f_regression, mutual_info_regression

from autoPyTorch.datasets.base_dataset import BaseDatasetPropertiesType
from autoPyTorch.pipeline.components.preprocessing.tabular_preprocessing.feature_preprocessing. \
    base_feature_preprocessor import autoPyTorchFeaturePreprocessingComponent
from autoPyTorch.utils.common import FitRequirement, HyperparameterSearchSpace, add_hyperparameter


SCORE_FUNC_CHOICES = ('f_regression', 'mutual_info')


class SelectPercentileRegression(autoPyTorchFeaturePreprocessingComponent):
    """
    Select features according to a percentile of the highest scores.
    Scores are calculated using one of SCORE_FUNC_CHOICES
    """
    def __init__(self, score_func: str = "f_regression",
                 percentile: int = 50,
                 random_state: Optional[np.random.RandomState] = None
                 ):
        self.percentile = percentile
        if score_func == "f_regression":
            self.score_func = f_regression
        elif score_func == "mutual_info":
            self.score_func = partial(mutual_info_regression, random_state=random_state)
        else:
            raise ValueError(f"score_func of {self.__class__.__name__} must be in {SCORE_FUNC_CHOICES}, "
                             "but is: {score_func}")

        super().__init__(random_state=random_state)
        self.add_fit_requirements([
            FitRequirement('issigned', (bool,), user_defined=True, dataset_property=True)])

    def fit(self, X: Dict[str, Any], y: Any = None) -> BaseEstimator:

        self.check_requirements(X, y)

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
                                                                          value_range=SCORE_FUNC_CHOICES,
                                                                          default_value="f_regression",
                                                                          ),
    ) -> ConfigurationSpace:
        cs = ConfigurationSpace()

        add_hyperparameter(cs, score_func, CategoricalHyperparameter)
        add_hyperparameter(cs, percentile, UniformIntegerHyperparameter)

        return cs

    @staticmethod
    def get_properties(dataset_properties: Optional[Dict[str, BaseDatasetPropertiesType]] = None) -> Dict[str, Any]:
        return {'shortname': 'SPR',
                'name': 'Select Percentile Regression',
                'handles_sparse': True,
                'handles_regression': True,
                'handles_classification': False
                }
