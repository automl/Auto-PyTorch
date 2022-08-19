from typing import Any, Dict, Optional, Union

import numpy as np

from autoPyTorch.datasets.base_dataset import BaseDatasetPropertiesType
from autoPyTorch.pipeline.components.preprocessing.tabular_preprocessing.encoding.OneHotEncoder import \
    OneHotEncoder
from autoPyTorch.pipeline.components.preprocessing.time_series_preprocessing.encoding.time_series_base_encoder import \
    TimeSeriesBaseEncoder


class TimeSeriesOneHotEncoder(TimeSeriesBaseEncoder):
    def __init__(self,
                 random_state: Optional[Union[np.random.RandomState, int]] = None
                 ):
        super(TimeSeriesOneHotEncoder, self).__init__()
        self.random_state = random_state

    def fit(self, X: Dict[str, Any], y: Any = None) -> TimeSeriesBaseEncoder:
        OneHotEncoder.fit(self, X, y)
        categorical_columns = X['dataset_properties']['categorical_columns']
        if 'num_categories_per_col_encoded' in X['dataset_properties']:
            num_categories_per_col = X['dataset_properties']['num_categories_per_col_encoded']
        else:
            num_categories_per_col = X['dataset_properties']['num_categories_per_col']
        feature_names = X['dataset_properties']['feature_names']
        feature_shapes = X['dataset_properties']['feature_shapes']

        for i, cat_column in enumerate(categorical_columns):
            feature_shapes[feature_names[cat_column]] = num_categories_per_col[i]
        self.feature_shapes = feature_shapes
        return self

    @staticmethod
    def get_properties(
        dataset_properties: Optional[Dict[str, BaseDatasetPropertiesType]] = None
    ) -> Dict[str, Union[str, bool]]:
        return {
            'shortname': 'TimeSeriesOneHotEncoder',
            'name': 'Time Series One Hot Encoder',
            'handles_sparse': False
        }
