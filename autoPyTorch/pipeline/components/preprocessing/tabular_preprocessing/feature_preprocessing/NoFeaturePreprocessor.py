from typing import Any, Dict, Optional, Union

import numpy as np

from autoPyTorch.datasets.base_dataset import BaseDatasetPropertiesType
from autoPyTorch.pipeline.components.preprocessing.tabular_preprocessing.feature_preprocessing.\
    base_feature_preprocessor import autoPyTorchFeaturePreprocessingComponent


class NoFeaturePreprocessor(autoPyTorchFeaturePreprocessingComponent):
    """
    Don't perform feature preprocessing on categorical features
    """
    def __init__(self,
                 random_state: Optional[Union[np.random.RandomState, int]] = None
                 ):
        super().__init__()
        self.random_state = random_state

    def fit(self, X: Dict[str, Any], y: Any = None) -> autoPyTorchFeaturePreprocessingComponent:
        """
        The fit function calls the fit function of the underlying model
        and returns the transformed array.
        Args:
            X (np.ndarray): input features
            y (Optional[np.ndarray]): input labels

        Returns:
            instance of self
        """
        self.check_requirements(X, y)

        return self

    def transform(self, X: Dict[str, Any]) -> Dict[str, Any]:
        """
        Adds the self into the 'X' dictionary and returns it.
        Args:
            X (Dict[str, Any]): 'X' dictionary

        Returns:
            (Dict[str, Any]): the updated 'X' dictionary
        """
        X.update({'feature_preprocessor': self.preprocessor})
        return X

    @staticmethod
    def get_properties(dataset_properties: Optional[Dict[str, BaseDatasetPropertiesType]] = None
                       ) -> Dict[str, Union[str, bool]]:
        return {'shortname': 'NoFeaturePreprocessing',
                'name': 'No Feature Preprocessing',
                'handles_sparse': True,
                'handles_classification': True,
                'handles_regression': True
                }
