from typing import Any, Dict, Optional, Union

import numpy as np

from autoPyTorch.pipeline.components.preprocessing.time_series_preprocessing.scaling.base_scaler import BaseScaler
from autoPyTorch.pipeline.components.preprocessing.time_series_preprocessing.scaling.utils import TimeSeriesScaler


class NoScaler(BaseScaler):
    """
    No scaling performed
    """

    def __init__(self, random_state: Optional[Union[np.random.RandomState, int]] = None):
        """
        Args:
            random_state (Optional[Union[np.random.RandomState, int]]): Determines random number generation
        """
        super().__init__()
        self.random_state = random_state

    def fit(self, X: Dict[str, Any], y: Any = None) -> BaseScaler:
        """
        The fit function calls the fit function of the underlying model
        and returns the transformed array.
        Args:
            X (np.ndarray): input features
            y (Optional[np.ndarray]): input labels

        Returns:
            instance of self
        """
        self.check_requirements(X, y)

        self.preprocessor["numerical"] = TimeSeriesScaler(mode="none")
        return self

    @staticmethod
    def get_properties(dataset_properties: Optional[Dict[str, Any]] = None) -> Dict[str, Union[str, bool]]:
        return {
            'shortname': 'NoScaler',
            'name': 'NoScaler'
        }
