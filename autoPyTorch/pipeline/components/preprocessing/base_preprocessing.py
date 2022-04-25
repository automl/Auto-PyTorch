from typing import Any, Dict, Optional, Union

from ConfigSpace.configuration_space import ConfigurationSpace

import numpy as np

import pandas as pd

from scipy.sparse import spmatrix

import torch

from autoPyTorch.automl_common.common.utils.backend import Backend
from autoPyTorch.datasets.base_dataset import BaseDatasetPropertiesType
from autoPyTorch.pipeline.components.base_component import autoPyTorchComponent
from autoPyTorch.utils.common import FitRequirement


class autoPyTorchPreprocessingComponent(autoPyTorchComponent):
    """
     Provides abstract interface for preprocessing algorithms in AutoPyTorch.
    """
    def __init__(self) -> None:
        super().__init__()
        self.add_fit_requirements([
            FitRequirement('X_train',
                           (np.ndarray, pd.DataFrame, spmatrix),
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
        raise NotImplementedError()

    @staticmethod
    def get_hyperparameter_search_space(
        dataset_properties: Optional[Dict[str, BaseDatasetPropertiesType]] = None
    ) -> ConfigurationSpace:
        """Return the configuration space of this classification algorithm.

        Args:
            dataset_properties (Optional[Dict[str, BaseDatasetPropertiesType]): Describes the dataset
               to work on

        Returns:
            ConfigurationSpace: The configuration space of this algorithm.
        """
        return ConfigurationSpace()


class autoPyTorchTargetPreprocessingComponent(autoPyTorchComponent):
    """
     Provides abstract interface for target preprocessing algorithms in AutoPyTorch. Most methods defined in this class
     are the same as autoPyTorch.pipeline.components.preprocessing.base_preprocessing.autoPyTorchPreprocessingComponent
     However, they are defined as two different classes such that its subclasses will not be identified as feature
     preprocessor
    """
    def __init__(self) -> None:
        super().__init__()
        self.add_fit_requirements([
            FitRequirement('y_train',
                           (pd.DataFrame, ),
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
        raise NotImplementedError()

    @staticmethod
    def get_hyperparameter_search_space(
        dataset_properties: Optional[Dict[str, BaseDatasetPropertiesType]] = None
    ) -> ConfigurationSpace:
        """Return the configuration space of this classification algorithm.

        Args:
            dataset_properties (Optional[Dict[str, BaseDatasetPropertiesType]): Describes the dataset
               to work on

        Returns:
            ConfigurationSpace: The configuration space of this algorithm.
        """
        return ConfigurationSpace()
