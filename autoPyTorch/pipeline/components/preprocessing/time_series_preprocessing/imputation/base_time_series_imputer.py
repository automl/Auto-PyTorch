from typing import Any, Dict, List, Optional, Union

from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import CategoricalHyperparameter

import numpy as np

from autoPyTorch.datasets.base_dataset import BaseDatasetPropertiesType
from autoPyTorch.utils.common import HyperparameterSearchSpace, add_hyperparameter


class BaseTimeSeriesImputer:
    def fit(self, X: Dict[str, Any], y: Optional[Any] = None) -> "BaseTimeSeriesImputer":
        raise NotImplementedError

    def transform(self, X: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError

    @staticmethod
    def get_hyperparameter_search_space(
        dataset_properties: Optional[Dict[str, BaseDatasetPropertiesType]] = None,
        imputation_strategy: HyperparameterSearchSpace = HyperparameterSearchSpace(
            hyperparameter='imputation_strategy',
            value_range=("drift", "linear", "nearest", "constant_zero", "mean", "median", "bfill", "ffill"),
            default_value="drift",
        ),
    ) -> ConfigurationSpace:
        """Get the hyperparameter search space for the Time Series Imputator

        Args:
            dataset_properties (Optional[Dict[str, BaseDatasetPropertiesType]])
                Properties that describe the dataset
            imputation_strategy (HyperparameterSearchSpace: default = ...)
                The strategy to use for imputation, its hyperparameters are defined by sktime

        Returns:
            ConfigurationSpace
                The space of possible configurations for a Time Series Imputor with the given
                `dataset_properties`
        """
        cs = ConfigurationSpace()
        add_hyperparameter(cs, imputation_strategy, CategoricalHyperparameter)
        return cs