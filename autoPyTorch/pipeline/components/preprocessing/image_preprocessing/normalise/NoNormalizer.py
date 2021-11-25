from typing import Any, Dict, Optional, Union

import numpy as np

import torch

from autoPyTorch.datasets.base_dataset import BaseDatasetPropertiesType
from autoPyTorch.pipeline.components.preprocessing.image_preprocessing.normalise.base_normalizer import (
    BaseNormalizer
)


class NoNormalizer(BaseNormalizer):
    def __init__(self, random_state: Optional[Union[np.random.RandomState, int]] = None
                 ):
        super().__init__()
        self.random_state = random_state

    def fit(self, X: Dict[str, Any], y: Optional[Any] = None) -> "NoNormalizer":
        """
        Initialises early_preprocessor and returns self.
        Args:
            X (Dict[str, Any]): 'X' dictionary

        Returns:
            autoPyTorchImagePreprocessingComponent: self
        """
        self.check_requirements(X, y)

        return self

    def transform(self, X: Dict[str, Any]) -> Dict[str, Any]:

        X.update({'normalise': self})
        return X

    def __call__(self, X: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """
        Makes the autoPyTorchPreprocessingComponent Callable. Calling the component
        calls the transform function of the underlying early_preprocessor and
        returns the transformed array.
        Args:
            X (Union[np.ndarray, torch.Tensor]): input data tensor

        Returns:
            Union[np.ndarray, torch.Tensor]: Transformed data tensor
        """
        return X

    @staticmethod
    def get_properties(dataset_properties: Optional[Dict[str, BaseDatasetPropertiesType]] = None
                       ) -> Dict[str, Any]:
        return {
            'shortname': 'no-normalize',
            'name': 'No Normalizer Node',
        }
