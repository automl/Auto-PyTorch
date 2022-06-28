from typing import Any, Dict, Optional, Union

from ConfigSpace import CategoricalHyperparameter, ConfigurationSpace

import numpy as np

from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline

import torch

from autoPyTorch.datasets.base_dataset import BaseDatasetPropertiesType
from autoPyTorch.pipeline.components.base_component import autoPyTorchComponent
from autoPyTorch.pipeline.components.setup.forecasting_target_scaling.utils import TargetScaler
from autoPyTorch.utils.common import HyperparameterSearchSpace, add_hyperparameter


class BaseTargetScaler(autoPyTorchComponent):
    def __init__(self,
                 random_state: Optional[Union[np.random.RandomState, int]] = None,
                 scaling_mode: str = 'none'):
        super().__init__()
        self.random_state = random_state
        self.scaling_mode = scaling_mode
        self.preprocessor: Optional[Pipeline] = None

    def fit(self, X: Dict[str, Any], y: Any = None) -> BaseEstimator:
        """
        Creates a column transformer for the chosen tabular
        preprocessors
        Args:
            X (Dict[str, Any]):
                fit dictionary

        Returns:
            "BaseEstimator":
                an instance of self
        """
        self.check_requirements(X, y)
        self.scaler = TargetScaler(mode=self.scaling_mode)
        return self

    def transform(self, X: Dict[str, Any]) -> Dict[str, Any]:
        """
        Adds the time series transformer to fit dictionary
        Args:
            X (Dict[str, Any]):
                fit dictionary

        Returns:
            X (Dict[str, Any]):
                updated fit dictionary
        """
        X.update({'target_scaler': self})
        return X

    def __call__(self,
                 past_target: Union[np.ndarray, torch.tensor],
                 past_observed_values: Optional[torch.BoolTensor] = None,
                 future_targets: Optional[Union[np.ndarray, torch.Tensor]] = None,
                 ) -> Union[np.ndarray, torch.tensor]:

        if self.scaler is None:
            raise ValueError("cant call {} without fitting the column transformer first."
                             .format(self.__class__.__name__))

        if len(past_target.shape) == 2:
            # expand batch dimension when called on a single record
            past_target = past_target[np.newaxis, ...]
        past_target, future_targets, loc, scale = self.scaler.transform(past_target,
                                                                        past_observed_values,
                                                                        future_targets)
        return past_target, future_targets, loc, scale

    @staticmethod
    def get_hyperparameter_search_space(
            dataset_properties: Optional[Dict[str, BaseDatasetPropertiesType]] = None,
            scaling_mode: HyperparameterSearchSpace = HyperparameterSearchSpace(
                hyperparameter='scaling_mode',
                value_range=("standard", "min_max", "max_abs", "mean_abs", "none"),
                default_value="standard",
            ),
    ) -> ConfigurationSpace:
        cs = ConfigurationSpace()
        add_hyperparameter(cs, scaling_mode, CategoricalHyperparameter)
        return cs
