from functools import partial
from typing import Any, Dict, Optional

from ConfigSpace.conditions import NotEqualsCondition
from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import (
    CategoricalHyperparameter,
    UniformFloatHyperparameter,
)

import numpy as np

from sklearn.base import BaseEstimator
from sklearn.feature_selection import GenericUnivariateSelect, chi2, f_classif, mutual_info_classif

from autoPyTorch.datasets.base_dataset import BaseDatasetPropertiesType
from autoPyTorch.pipeline.components.preprocessing.tabular_preprocessing.feature_preprocessing. \
    base_feature_preprocessor import autoPyTorchFeaturePreprocessingComponent
from autoPyTorch.pipeline.components.preprocessing.tabular_preprocessing.feature_preprocessing.utils \
    import filter_score_func_choices
from autoPyTorch.utils.common import FitRequirement, HyperparameterSearchSpace, add_hyperparameter, get_hyperparameter


SCORE_FUNC_CHOICES = ("chi2", "mutual_info_classif", "f_classif")


class SelectRatesClassification(autoPyTorchFeaturePreprocessingComponent):
    """
    Univariate feature selector by selecting the best features based on
    univariate statistical tests. Tests can be one of SCORE_FUNC_CHOICES
    """
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
        elif score_func == "mutual_info_classif":
            self.score_func = partial(mutual_info_classif,
                                      random_state=random_state)
            # mutual info classif constantly crashes without mode percentile
            self.mode = "percentile"
        else:
            raise ValueError(f"score_func of {self.__class__.__name__} must be in {SCORE_FUNC_CHOICES}, "
                             "but is: {score_func}")

        super().__init__(random_state=random_state)
        self.add_fit_requirements([
            FitRequirement('issparse', (bool,), user_defined=True, dataset_property=True),
            FitRequirement('issigned', (bool,), user_defined=True, dataset_property=True)])

    def fit(self, X: Dict[str, Any], y: Any = None) -> BaseEstimator:

        self.check_requirements(X, y)

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
                                                                    value_range=('fpr', 'fdr', 'fwe', "percentile"),
                                                                    default_value='fpr',
                                                                    ),
        score_func: HyperparameterSearchSpace = HyperparameterSearchSpace(hyperparameter="score_func",
                                                                          value_range=SCORE_FUNC_CHOICES,
                                                                          default_value="chi2",
                                                                          ),
    ) -> ConfigurationSpace:

        score_func = filter_score_func_choices(class_name="SelectPercentileClassification",
                                               dataset_properties=dataset_properties,
                                               score_func=score_func)

        cs = ConfigurationSpace()

        score_func_hp = get_hyperparameter(score_func, CategoricalHyperparameter)
        add_hyperparameter(cs, alpha, UniformFloatHyperparameter)
        mode_hp = get_hyperparameter(mode, CategoricalHyperparameter)

        cs.add_hyperparameters([mode_hp, score_func_hp])
        # mutual_info_classif constantly crashes if mode is not percentile
        # as a WA, fix the mode for this score
        if "mutual_info_classif" in score_func_hp.choices:
            cond = NotEqualsCondition(mode_hp, score_func_hp, 'mutual_info_classif')
            cs.add_condition(cond)

        return cs

    @staticmethod
    def get_properties(dataset_properties: Optional[Dict[str, BaseDatasetPropertiesType]] = None) -> Dict[str, Any]:
        return {'shortname': 'SRC',
                'name': 'Select Rates Classification',
                'handles_sparse': True,
                'handles_regression': False,
                'handles_classification': True
                }
