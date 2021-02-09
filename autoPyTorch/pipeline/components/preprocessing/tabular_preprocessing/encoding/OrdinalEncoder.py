from typing import Any, Dict, Optional, Union

import numpy as np

from sklearn.preprocessing import OrdinalEncoder as OE

from autoPyTorch.pipeline.components.preprocessing.tabular_preprocessing.encoding.base_encoder import BaseEncoder


class OrdinalEncoder(BaseEncoder):
    """
    Encode categorical features as a one-hot numerical array
    """
    def __init__(self, random_state: Optional[Union[np.random.RandomState, int]] = None):
        super().__init__()
        self.random_state = random_state

    def fit(self, X: Dict[str, Any], y: Any = None) -> BaseEncoder:

        self.check_requirements(X, y)

        self.preprocessor['categorical'] = OE()
        return self

    @staticmethod
    def get_properties(dataset_properties: Optional[Dict[str, Any]] = None) -> Dict[str, Union[str, bool]]:
        return {
            'shortname': 'OrdinalEncoder',
            'name': 'Ordinal Encoder',
            'handles_sparse': False
        }
