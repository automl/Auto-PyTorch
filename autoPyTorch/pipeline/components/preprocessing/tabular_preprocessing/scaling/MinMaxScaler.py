from typing import Any, Dict, Optional, Tuple, Union

import numpy as np

from sklearn.preprocessing import MinMaxScaler as SklearnMinMaxScaler

from autoPyTorch.datasets.base_dataset import BaseDatasetPropertiesType
from autoPyTorch.pipeline.components.preprocessing.tabular_preprocessing.scaling.base_scaler import BaseScaler


class MinMaxScaler(BaseScaler):
    """
    Scale numerical columns/features into feature_range
    """
    def __init__(self,
                 random_state: Optional[Union[np.random.RandomState, int]] = None,
                 feature_range: Tuple[Union[int, float], Union[int, float]] = (0, 1)):
        super().__init__()
        self.random_state = random_state
        self.feature_range = feature_range

    def fit(self, X: Dict[str, Any], y: Any = None) -> BaseScaler:

        self.check_requirements(X, y)

        self.preprocessor['numerical'] = SklearnMinMaxScaler(feature_range=self.feature_range, copy=False)
        return self

    @staticmethod
    def get_properties(dataset_properties: Optional[Dict[str, BaseDatasetPropertiesType]] = None
                       ) -> Dict[str, Union[str, bool]]:
        return {
            'shortname': 'MinMaxScaler',
            'name': 'MinMaxScaler',
            'handles_sparse': True
        }
