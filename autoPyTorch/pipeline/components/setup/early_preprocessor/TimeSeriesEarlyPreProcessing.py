from typing import Any, Dict, Optional, Union

from ConfigSpace.configuration_space import ConfigurationSpace

import numpy as np

import pandas as pd

from scipy.sparse import spmatrix

from autoPyTorch.datasets.base_dataset import BaseDatasetPropertiesType
from autoPyTorch.pipeline.components.preprocessing.base_preprocessing import autoPyTorchTargetPreprocessingComponent
from autoPyTorch.pipeline.components.setup.early_preprocessor.EarlyPreprocessing import EarlyPreprocessing
from autoPyTorch.pipeline.components.setup.early_preprocessor.utils import (
    get_preprocess_transforms,
    time_series_preprocess
)
from autoPyTorch.utils.common import FitRequirement


class TimeSeriesEarlyPreprocessing(EarlyPreprocessing):

    def __init__(self, random_state: Optional[np.random.RandomState] = None) -> None:
        super().__init__()
        self.random_state = random_state
        self.add_fit_requirements([
            FitRequirement('is_small_preprocess', (bool,), user_defined=True, dataset_property=True),
            FitRequirement('X_train', (pd.DataFrame, ), user_defined=True,
                           dataset_property=False)])

    def transform(self, X: Dict[str, Any]) -> Dict[str, Any]:

        transforms = get_preprocess_transforms(X)
        if X['dataset_properties']['is_small_preprocess']:
            if 'X_train' in X:
                X_train = X['X_train']
            else:
                # Incorporate the transform to the dataset
                X_train = X['backend'].load_datamanager().train_tensors[0]

            X['X_train'] = time_series_preprocess(dataset=X_train, transforms=transforms)

        # We need to also save the preprocess transforms for inference
        X.update({'preprocess_transforms': transforms})
        return X

    @staticmethod
    def get_properties(dataset_properties: Optional[Dict[str, BaseDatasetPropertiesType]] = None
                       ) -> Dict[str, Union[str, bool]]:
        return {
            'shortname': 'TimeSeriesEarlyPreprocessing',
            'name': 'TIme Series Early Preprocessing Node',
        }


class TimeSeriesTargetEarlyPreprocessing(EarlyPreprocessing):

    def __init__(self, random_state: Optional[np.random.RandomState] = None) -> None:
        super().__init__()
        self.random_state = random_state
        self.add_fit_requirements([
            FitRequirement('is_small_preprocess', (bool,), user_defined=True, dataset_property=True),
            FitRequirement('y_train', (pd.DataFrame,), user_defined=True,
                           dataset_property=False)])

    def transform(self, X: Dict[str, Any]) -> Dict[str, Any]:
        # TODO consider inverse transformation
        transforms = get_preprocess_transforms(X, preprocess_type=autoPyTorchTargetPreprocessingComponent)
        if X['dataset_properties']['is_small_preprocess']:
            if 'y_train' in X:
                y_train = X['y_train']
            else:
                # Incorporate the transform to the dataset
                y_train = X['backend'].load_datamanager().train_tensors[1]

            X['y_train'] = time_series_preprocess(dataset=y_train, transforms=transforms)

        # We need to also save the preprocess transforms for inference
        X.update({'preprocess_target_transforms': transforms})
        return X

    @staticmethod
    def get_properties(dataset_properties: Optional[Dict[str, BaseDatasetPropertiesType]] = None
                       ) -> Dict[str, Union[str, bool]]:
        return {
            'shortname': 'TimeSeriesTargetEarlyPreprocessing',
            'name': 'TIme Series Target Early Preprocessing Node',
        }
