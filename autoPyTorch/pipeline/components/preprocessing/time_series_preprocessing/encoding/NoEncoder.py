from typing import Any, Dict, Optional, Union

import numpy as np

from autoPyTorch.datasets.base_dataset import BaseDatasetPropertiesType
from autoPyTorch.pipeline.components.preprocessing.tabular_preprocessing.encoding.NoEncoder import \
    NoEncoder
from autoPyTorch.pipeline.components.preprocessing.time_series_preprocessing.encoding.time_series_base_encoder import \
    TimeSeriesBaseEncoder


class TimeSeriesNoEncoder(TimeSeriesBaseEncoder):
    def __init__(self,
                 random_state: Optional[Union[np.random.RandomState, int]] = None
                 ):
        super().__init__()
        self.random_state = random_state

    def fit(self, X: Dict[str, Any], y: Any = None) -> "TimeSeriesBaseEncoder":
        NoEncoder.fit(self, X, y)
        self.feature_shapes = X['dataset_properties']['feature_shapes']
        return self

    @staticmethod
    def get_properties(
        dataset_properties: Optional[Dict[str, BaseDatasetPropertiesType]] = None
    ) -> Dict[str, Union[str, bool]]:
        return {
            'shortname': 'TimeSeriesNoEncoder',
            'name': 'Time Series No Encoder',
            'handles_sparse': True
        }

    def transform(self, X: Dict[str, Any]) -> Dict[str, Any]:
        """
        Adds the self into the 'X' dictionary and returns it.

        Args:
            X (Dict[str, Any]): 'X' dictionary

        Returns:
            (Dict[str, Any]): the updated 'X' dictionary
        """
        return NoEncoder.transform(self, X)
