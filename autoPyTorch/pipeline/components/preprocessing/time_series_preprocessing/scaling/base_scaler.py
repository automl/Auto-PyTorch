from typing import Any, Dict, List, Optional, Union

from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import CategoricalHyperparameter

import numpy as np

from autoPyTorch.datasets.base_dataset import BaseDatasetPropertiesType
from autoPyTorch.pipeline.components.preprocessing.time_series_preprocessing.base_time_series_preprocessing import \
    autoPyTorchTimeSeriesPreprocessingComponent
from autoPyTorch.pipeline.components.preprocessing.time_series_preprocessing.scaling.utils import \
    TimeSeriesScaler
from autoPyTorch.utils.common import (FitRequirement,
                                      HyperparameterSearchSpace,
                                      add_hyperparameter)


class BaseScaler(autoPyTorchTimeSeriesPreprocessingComponent):
    """
    Provides abstract class interface for time series scalers in AutoPytorch
    """

    def __init__(self,
                 random_state: Optional[Union[np.random.RandomState, int]] = None,
                 scaling_mode: str = 'standard'):
        super().__init__()
        self.add_fit_requirements([
            FitRequirement('numerical_columns', (List,), user_defined=True, dataset_property=True),
            FitRequirement('is_small_preprocess', (bool,), user_defined=True, dataset_property=True)
        ])
        self.random_state = random_state
        self.scaling_mode = scaling_mode

    def fit(self, X: Dict[str, Any], y: Any = None) -> 'BaseScaler':
        self.check_requirements(X, y)
        dataset_is_small_preprocess = X["dataset_properties"]["is_small_preprocess"]
        static_features = X['dataset_properties'].get('static_features', ())
        self.preprocessor['numerical'] = TimeSeriesScaler(mode=self.scaling_mode,
                                                          dataset_is_small_preprocess=dataset_is_small_preprocess,
                                                          static_features=static_features)
        return self

    def transform(self, X: Dict[str, Any]) -> Dict[str, Any]:
        """
        Adds the fitted scalar into the 'X' dictionary and returns it.

        Args:
            X (Dict[str, Any]): 'X' dictionary

        Returns:
            (Dict[str, Any]): the updated 'X' dictionary
        """
        if self.preprocessor['numerical'] is None and self.preprocessor['categorical'] is None:
            raise ValueError(f"can not call transform on {self.__class__.__name__} without fitting first.")
        X.update({'scaler': self.preprocessor})
        return X

    @staticmethod
    def get_hyperparameter_search_space(
            dataset_properties: Optional[Dict[str, BaseDatasetPropertiesType]] = None,
            scaling_mode: HyperparameterSearchSpace = HyperparameterSearchSpace(
                hyperparameter='scaling_mode',
                value_range=("standard", "min_max", "max_abs", "mean_abs", "none"),
                default_value="standard",
            ),
    ) -> ConfigurationSpace:
        """Get the hyperparameter search space for the Time Series Imputator

        Args:
            dataset_properties (Optional[Dict[str, BaseDatasetPropertiesType]])
                Properties that describe the dataset
            scaling_mode (HyperparameterSearchSpace: default = ...)
                The strategy to use for scaling, its hyperparameters are defined by sktime

        Returns:
            ConfigurationSpace
                The space of possible configurations for a Time Series Imputor with the given
                `dataset_properties`
        """
        cs = ConfigurationSpace()
        add_hyperparameter(cs, scaling_mode, CategoricalHyperparameter)
        return cs
