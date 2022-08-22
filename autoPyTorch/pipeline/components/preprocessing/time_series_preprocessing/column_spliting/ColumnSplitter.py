from typing import Any, Dict, Optional

import numpy as np

from autoPyTorch.pipeline.components.preprocessing.tabular_preprocessing.column_splitting.ColumnSplitter import (
    ColumnSplitter
)
from autoPyTorch.pipeline.components.preprocessing.time_series_preprocessing.base_time_series_preprocessing import \
    autoPyTorchTimeSeriesPreprocessingComponent


class TimeSeriesColumnSplitter(ColumnSplitter, autoPyTorchTimeSeriesPreprocessingComponent):
    """
    Splits categorical columns into embed or encode columns based on a hyperparameter.
    The splitter for time series is quite similar to the tabular splitter. However, we need to reserve the raw
    number of categorical features for later use
    """
    def __init__(
        self,
        min_categories_for_embedding: float = 5,
        random_state: Optional[np.random.RandomState] = None
    ):
        super(TimeSeriesColumnSplitter, self).__init__(min_categories_for_embedding, random_state)
        self.num_categories_per_col_encoded = None

    def fit(self, X: Dict[str, Any], y: Optional[Any] = None) -> 'TimeSeriesColumnSplitter':
        super(TimeSeriesColumnSplitter, self).fit(X, y)

        self.num_categories_per_col_encoded = X['dataset_properties']['num_categories_per_col']
        for i in range(len(self.num_categories_per_col_encoded)):
            if i in self.special_feature_types['embed_columns']:
                self.num_categories_per_col_encoded[i] = 1
        return self

    def transform(self, X: Dict[str, Any]) -> Dict[str, Any]:
        X = super(TimeSeriesColumnSplitter, self).transform(X)
        X['dataset_properties']['num_categories_per_col_encoded'] = self.num_categories_per_col_encoded
        return X
