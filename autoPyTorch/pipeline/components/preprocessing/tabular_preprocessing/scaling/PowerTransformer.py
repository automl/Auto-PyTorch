from typing import Any, Dict, Optional, Union

import numpy as np

from sklearn.preprocessing import PowerTransformer as SklearnPowerTransformer

from autoPyTorch.datasets.base_dataset import BaseDatasetPropertiesType
from autoPyTorch.pipeline.components.preprocessing.tabular_preprocessing.scaling.base_scaler import BaseScaler


class PowerTransformer(BaseScaler):
    """
    Map data to as close to a Gaussian distribution as possible
    in order to reduce variance and minimize skewness.

    Uses `yeo-johnson` power transform method. Also, data is normalised
    to zero mean and unit variance.
    """
    def __init__(self,
                 random_state: Optional[np.random.RandomState] = None):
        super().__init__()
        self.random_state = random_state

    def fit(self, X: Dict[str, Any], y: Any = None) -> BaseScaler:

        self.check_requirements(X, y)

        self.preprocessor['numerical'] = SklearnPowerTransformer(method='yeo-johnson', copy=False)
        return self

    @staticmethod
    def get_properties(dataset_properties: Optional[Dict[str, BaseDatasetPropertiesType]] = None
                       ) -> Dict[str, Union[str, bool]]:
        return {
            'shortname': 'PowerTransformer',
            'name': 'PowerTransformer',
            'handles_sparse': False
        }
