from typing import Any, Dict, List, Optional, Union, Tuple

import numpy as np

from sklearn.pipeline import Pipeline, make_pipeline
#from sktime.transformations.panel.compose import ColumnTransformer
from sklearn.compose import ColumnTransformer

import torch

from autoPyTorch.utils.common import FitRequirement, subsampler
from autoPyTorch.pipeline.components.preprocessing.time_series_preprocessing.base_time_series_preprocessing import (
    autoPyTorchTimeSeriesPreprocessingComponent
)
from autoPyTorch.pipeline.components.preprocessing.time_series_preprocessing.forecasting_target_scaling\
    .utils import TargetScaler


class BaseTargetScaler(autoPyTorchTimeSeriesPreprocessingComponent):
    def __init__(self, random_state: Optional[Union[np.random.RandomState, int]] = None):
        super().__init__()
        self.random_state = random_state
        self.preprocessor: Optional[Pipeline] = None
        self.add_fit_requirements([
            FitRequirement('target_columns', (Tuple,), user_defined=True, dataset_property=True),
        ])

    def fit(self, X: Dict[str, Any], y: Any = None) -> "BaseBatchScaler":
        """
        Creates a column transformer for the chosen tabular
        preprocessors
        Args:
            X (Dict[str, Any]): fit dictionary

        Returns:
            "TabularColumnTransformer": an instance of self
        """
        self.check_requirements(X, y)
        self.target_columns = X['dataset_properties']['target_columns']
        self.scaler = TargetScaler(mode=self.scaler_mode)
        return self

    @property
    def scaler_mode(self):
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
                 future_targets: Optional[Union[np.ndarray, torch.Tensor]]=None,
                 ) -> Union[np.ndarray, torch.tensor]:

        if self.scaler is None:
            raise ValueError("cant call {} without fitting the column transformer first."
                             .format(self.__class__.__name__))

        if len(past_target.shape) == 2:
            # expand batch dimension when called on a single record
            past_target = past_target[np.newaxis, ...]
        past_target, future_targets, loc, scale = self.scaler.transform(past_target, future_targets)
        return past_target, future_targets, loc, scale
