from typing import Any, Dict, Optional, Union

import numpy as np

from sklearn.preprocessing import OneHotEncoder as OHE

from autoPyTorch.datasets.base_dataset import BaseDatasetPropertiesType
from autoPyTorch.pipeline.components.preprocessing.tabular_preprocessing.encoding.base_encoder import BaseEncoder


class OneHotEncoder(BaseEncoder):
    """
    Encode categorical features as a one-hot numerical array
    """
    def __init__(self, random_state: Optional[Union[np.random.RandomState, int]] = None):
        super().__init__()
        self.random_state = random_state

    def fit(self, X: Dict[str, Any], y: Any = None) -> BaseEncoder:

        self.check_requirements(X, y)

        self.preprocessor['categorical'] = OHE(
            # It is safer to have the OHE produce a 0 array than to crash a good configuration
            categories=X['dataset_properties']['categories']
            if len(X['dataset_properties']['categories']) > 0 else 'auto',
            sparse=False,
            handle_unknown='ignore')
        return self

    @staticmethod
    def get_properties(
        dataset_properties: Optional[Dict[str, BaseDatasetPropertiesType]] = None
    ) -> Dict[str, Union[str, bool]]:
        return {
            'shortname': 'OneHotEncoder',
            'name': 'One Hot Encoder',
            'handles_sparse': False
        }
