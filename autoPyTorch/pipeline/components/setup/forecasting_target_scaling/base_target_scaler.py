from typing import Any, Dict, Optional, Union

import numpy as np

from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline

import torch

from ConfigSpace import ConfigurationSpace

from autoPyTorch.datasets.base_dataset import BaseDatasetPropertiesType
from autoPyTorch.pipeline.components.base_component import autoPyTorchComponent
from autoPyTorch.pipeline.components.setup.forecasting_target_scaling.utils import TargetScaler


class BaseTargetScaler(autoPyTorchComponent):
    def __init__(self, random_state: Optional[Union[np.random.RandomState, int]] = None):
        super().__init__()
        self.random_state = random_state
        self.preprocessor: Optional[Pipeline] = None

    def fit(self, X: Dict[str, Any], y: Any = None) -> BaseEstimator:
        """
        Creates a column transformer for the chosen tabular
        preprocessors
        Args:
            X (Dict[str, Any]): fit dictionary

        Returns:
            "TabularColumnTransformer": an instance of self
        """
        self.check_requirements(X, y)
        self.scaler = TargetScaler(mode=self.scaler_mode)
        return self

    @property
    def scaler_mode(self) -> str:
        raise NotImplementedError

    def transform(self, X: Dict[str, Any]) -> Dict[str, Any]:
        """
        Adds the time series transformer to fit dictionary
        Args:
            X (Dict[str, Any]): fit dictionary

        Returns:
            X (Dict[str, Any]): updated fit dictionary
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
            dataset_properties: Optional[Dict[str, BaseDatasetPropertiesType]] = None
    ) -> ConfigurationSpace:
        cs = ConfigurationSpace()
        return cs
