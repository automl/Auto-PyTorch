from typing import Any, Dict, Optional, Union

from ConfigSpace.configuration_space import ConfigurationSpace

import numpy as np

import pandas as pd

from scipy.sparse import csr_matrix

import torch

from autoPyTorch.pipeline.components.base_component import autoPyTorchComponent
from autoPyTorch.utils.backend import Backend
from autoPyTorch.utils.common import FitRequirement


class autoPyTorchPreprocessingComponent(autoPyTorchComponent):
    """
     Provides abstract interface for preprocessing algorithms in AutoPyTorch.
    """
    def __init__(self) -> None:
        super().__init__()
        self.add_fit_requirements([
            FitRequirement('X_train',
                           (np.ndarray, pd.DataFrame, csr_matrix),
                           user_defined=True, dataset_property=False),
            FitRequirement('backend',
                           (Backend, ),
                           user_defined=True, dataset_property=False)])

    def transform(self, X: Dict[str, Any]) -> Dict[str, Any]:
        """
        Adds the fitted early_preprocessor into the 'X' dictionary and returns it.
        Args:
            X (Dict[str, Any]): 'X' dictionary

        Returns:
            (Dict[str, Any]): the updated 'X' dictionary
        """
        raise NotImplementedError()

    def __call__(self, X: Union[np.ndarray, torch.tensor]) -> Union[np.ndarray, torch.tensor]:
        """
        Makes the autoPyTorchPreprocessingComponent Callable. Calling the component
        calls the transform function of the underlying early_preprocessor and
        returns the transformed array.
        Args:
            X (Union[np.ndarray, torch.tensor]): input data tensor

        Returns:
            Union[np.ndarray, torch.tensor]: Transformed data tensor
        """
        raise NotImplementedError()

    @staticmethod
    def get_hyperparameter_search_space(
        dataset_properties: Optional[Dict[str, str]] = None
    ) -> ConfigurationSpace:
        """Return the configuration space of this classification algorithm.

        Args:
            dataset_properties (Optional[Dict[str, Union[str, int]]): Describes the dataset
               to work on

        Returns:
            ConfigurationSpace: The configuration space of this algorithm.
        """
        return ConfigurationSpace()
