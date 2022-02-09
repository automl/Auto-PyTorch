import warnings
from functools import partial
from typing import Any, Dict, Optional

from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import (
    CategoricalHyperparameter,
    UniformIntegerHyperparameter,
)

import numpy as np

from sklearn.base import BaseEstimator
from sklearn.feature_selection import SelectPercentile, chi2, f_classif, mutual_info_classif

from autoPyTorch.datasets.base_dataset import BaseDatasetPropertiesType
from autoPyTorch.pipeline.components.preprocessing.tabular_preprocessing.feature_preprocessing. \
    base_feature_preprocessor import autoPyTorchFeaturePreprocessingComponent
from autoPyTorch.utils.common import FitRequirement, HyperparameterSearchSpace, add_hyperparameter


class SelectPercentileClassification(autoPyTorchFeaturePreprocessingComponent):
    """
    Select features according to a percentile of the highest scores.
    Scores are calculated using one of 'chi2, 'f_classif', 'mutual_info_classif'
    """
    def __init__(self, score_func: str = "chi2",
                 percentile: int = 50,
                 random_state: Optional[np.random.RandomState] = None
                 ):
        self.percentile = percentile
        if score_func == "chi2":
            self.score_func = chi2
        elif score_func == "f_classif":
            self.score_func = f_classif
        elif score_func == "mutual_info_classif":
            self.score_func = partial(mutual_info_classif, random_state=random_state)
        else:
            raise ValueError("score_func must be in ('chi2, 'f_classif', 'mutual_info_classif'), "
                             "but is: %s" % score_func)

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
                                                                          value_range=("chi2",
                                                                                       "f_classif",
                                                                                       "mutual_info_classif"),
                                                                          default_value="chi2",
                                                                          ),
    ) -> ConfigurationSpace:
        value_range = list(score_func.value_range)
        if dataset_properties is not None:
            if (
                dataset_properties.get("issigned") is True
            ):
                value_range = [value for value in value_range if value not in ("chi2", "mutual_info_classif")]
            if dataset_properties.get("issparse") is True:
                value_range = [value for value in value_range if value != "f_classif"]

        if value_range != list(score_func.value_range):
            warnings.warn(f"Given choices for `score_func` are not compatible with the dataset. "
                          f"Updating choices to {value_range}")

        if len(value_range) == 0:
            raise TypeError("`SelectPercentileClassification` is not compatible with the"
                            " current dataset as it is both `signed` and `sparse`")
        default_value = score_func.default_value if score_func.default_value in value_range else value_range[-1]
        score_func = HyperparameterSearchSpace(hyperparameter="score_func",
                                               value_range=value_range,
                                               default_value=default_value,
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
