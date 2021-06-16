from typing import Any, Dict, Optional, Union

import numpy as np

from sklearn.preprocessing import StandardScaler as SklearnStandardScaler

from autoPyTorch.datasets.base_dataset import BaseDatasetPropertiesType
from autoPyTorch.pipeline.components.preprocessing.tabular_preprocessing.scaling.base_scaler import BaseScaler
from autoPyTorch.utils.common import FitRequirement


class StandardScaler(BaseScaler):
    """
    Standardise numerical columns/features by removing mean and scaling to unit/variance
    """
    def __init__(self,
                 random_state: Optional[Union[np.random.RandomState, int]] = None
                 ):
        super().__init__()
        self.random_state = random_state
        self.add_fit_requirements([
            FitRequirement('issparse', (bool,), user_defined=True, dataset_property=True)
        ])

    def fit(self, X: Dict[str, Any], y: Any = None) -> BaseScaler:

        self.check_requirements(X, y)

        with_mean, with_std = (False, False) if X['dataset_properties']['issparse'] else (True, True)
        self.preprocessor['numerical'] = SklearnStandardScaler(with_mean=with_mean, with_std=with_std, copy=False)
        return self

    @staticmethod
    def get_properties(dataset_properties: Optional[Dict[str, BaseDatasetPropertiesType]] = None
                       ) -> Dict[str, Union[str, bool]]:
        return {
            'shortname': 'StandardScaler',
            'name': 'Standard Scaler',
            'handles_sparse': True
        }
