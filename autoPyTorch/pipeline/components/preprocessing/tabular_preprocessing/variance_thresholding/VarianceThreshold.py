from typing import Any, Dict, Optional, Union

import numpy as np

from sklearn.feature_selection import VarianceThreshold as SklearnVarianceThreshold

from autoPyTorch.datasets.base_dataset import BaseDatasetPropertiesType
from autoPyTorch.pipeline.components.preprocessing.tabular_preprocessing.base_tabular_preprocessing import \
    autoPyTorchTabularPreprocessingComponent


class VarianceThreshold(autoPyTorchTabularPreprocessingComponent):
    """
    Removes features that have the same value in the training data.
    """
    def __init__(self, random_state: Optional[np.random.RandomState] = None):
        super().__init__()

    def fit(self, X: Dict[str, Any], y: Optional[Any] = None) -> 'VarianceThreshold':

        self.check_requirements(X, y)

        self.preprocessor['numerical'] = SklearnVarianceThreshold(
            threshold=0.0
        )
        return self

    def transform(self, X: Dict[str, Any]) -> Dict[str, Any]:
        if self.preprocessor['numerical'] is None:
            raise ValueError("cannot call transform on {} without fitting first."
                             .format(self.__class__.__name__))
        X.update({'variance_threshold': self.preprocessor})
        return X

    @staticmethod
    def get_properties(
        dataset_properties: Optional[Dict[str, BaseDatasetPropertiesType]] = None
    ) -> Dict[str, Union[str, bool]]:

        return {
            'shortname': 'Variance Threshold',
            'name': 'Variance Threshold (constant feature removal)',
            'handles_sparse': True,
        }
