from typing import Any, Dict, Optional, Union

import numpy as np

from autoPyTorch.pipeline.components.preprocessing.time_series_preprocessing.scaling.base_scaler import BaseScaler
from autoPyTorch.pipeline.components.preprocessing.time_series_preprocessing.scaling.utils import TimeSeriesScaler


class MinMaxScaler(BaseScaler):
    """
    Scales numerical features into range [0, 1]
    """
    def __init__(self, random_state: Optional[Union[np.random.RandomState, int]] = None):
        """
        Args:
            random_state (Optional[Union[np.random.RandomState, int]]): Determines random number generation
        """
        super().__init__()
        self.random_state = random_state

    def fit(self, X: Dict[str, Any], y: Any = None) -> BaseScaler:
        self.check_requirements(X, y)

        sequence_lengths_train = X['dataset_properties']['sequence_lengths_train']

        self.preprocessor["numerical"] = TimeSeriesScaler(mode="min_max", sequence_lengths_train=sequence_lengths_train)
        return self

    @staticmethod
    def get_properties(dataset_properties: Optional[Dict[str, Any]] = None) -> Dict[str, Union[str, bool]]:
        return {
            'shortname': 'MinMaxScaler',
            'name': 'MinMaxScaler'
        }
