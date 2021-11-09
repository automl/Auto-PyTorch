from typing import Any, Dict, Optional, Union

import numpy as np

import torch

from autoPyTorch.datasets.base_dataset import BaseDatasetPropertiesType
from autoPyTorch.pipeline.components.preprocessing.image_preprocessing.normalise.base_normalizer import BaseNormalizer


class ImageNormalizer(BaseNormalizer):

    def __init__(self, random_state: Optional[Union[np.random.RandomState, int]] = None
                 ):
        super().__init__()
        self.random_state = random_state
        self.mean: Optional[np.ndarray] = None
        self.std: Optional[np.ndarray] = None

    def fit(self, X: Dict[str, Any], y: Optional[Any] = None) -> "ImageNormalizer":
        """
        Initialises early_preprocessor and returns self.
        Args:
            X (Dict[str, Any]): 'X' dictionary

        Returns:
            autoPyTorchImagePreprocessingComponent: self
        """
        self.check_requirements(X, y)
        self.mean = X['dataset_properties']['mean']
        self.std = X['dataset_properties']['std']
        return self

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
        X = (X - self.mean) / self.std
        return X

    @staticmethod
    def get_properties(dataset_properties: Optional[Dict[str, BaseDatasetPropertiesType]] = None
                       ) -> Dict[str, Any]:
        return {
            'shortname': 'normalize',
            'name': 'Image Normalizer Node',
        }
