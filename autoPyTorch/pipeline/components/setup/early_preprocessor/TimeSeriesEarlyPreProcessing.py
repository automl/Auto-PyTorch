from typing import Any, Dict, List, Optional, Union

import numpy as np

import pandas as pd

from autoPyTorch.datasets.base_dataset import BaseDatasetPropertiesType
from autoPyTorch.pipeline.components.preprocessing.base_preprocessing import \
    autoPyTorchTargetPreprocessingComponent
from autoPyTorch.pipeline.components.setup.early_preprocessor.EarlyPreprocessing import \
    EarlyPreprocessing
from autoPyTorch.pipeline.components.setup.early_preprocessor.utils import (
    get_preprocess_transforms, get_preprocessed_dtype, time_series_preprocess)
from autoPyTorch.utils.common import FitRequirement


class TimeSeriesEarlyPreprocessing(EarlyPreprocessing):
    def __init__(self, random_state: Optional[np.random.RandomState] = None) -> None:
        super(EarlyPreprocessing, self).__init__()
        self.random_state = random_state
        self.add_fit_requirements([
            FitRequirement('X_train', (pd.DataFrame, ), user_defined=True,
                           dataset_property=False),
            FitRequirement('feature_names', (tuple,), user_defined=True, dataset_property=True),
            FitRequirement('numerical_columns', (List,), user_defined=True, dataset_property=True),
            FitRequirement('categorical_columns', (List,), user_defined=True, dataset_property=True),
        ])

    def transform(self, X: Dict[str, Any]) -> Dict[str, Any]:
        """
        if dataset is small process, we transform the entire dataset here.
        Before transformation, the order of the dataset is:
        [(unknown_columns), categorical_columns, numerical_columns]
        While after transformation, the order of the dataset is:
        [numerical_columns, categorical_columns, unknown_columns]
        we need to change feature_names and feature_shapes accordingly

        Args:
            X(Dict): fit dictionary

        Returns:
            X_transformed(Dict): transformed fit dictionary
        """

        transforms = get_preprocess_transforms(X)
        if 'X_train' in X:
            X_train = X['X_train']
        else:
            # Incorporate the transform to the dataset
            X_train = X['backend'].load_datamanager().train_tensors[0]

        X['X_train'] = time_series_preprocess(dataset=X_train, transforms=transforms)

        feature_names = X['dataset_properties']['feature_names']
        numerical_columns = X['dataset_properties']['numerical_columns']
        categorical_columns = X['dataset_properties']['categorical_columns']
        # encoding_columns = X['dataset_properties']['encoding_columns']
        encode_columns = X['encode_columns']
        import pdb
        pdb.set_trace()

        # resort feature_names
        # Previously, the categorical features are sorted before numerical features. However,
        # After the preprocessing. The numerical features are sorted at the first place.
        new_feature_names = [feature_names[num_col] for num_col in numerical_columns]
        new_feature_names += [feature_names[cat_col] for cat_col in categorical_columns]
        if set(feature_names) != set(new_feature_names):
            new_feature_names += list(set(feature_names) - set(new_feature_names))
        X['dataset_properties']['feature_names'] = tuple(new_feature_names)

        preprocessed_dtype = get_preprocessed_dtype(X['X_train'])
        # We need to also save the preprocess transforms for inference
        X.update({
            'preprocess_transforms': transforms,
            'shape_after_preprocessing': X['X_train'].shape[1:],
            'preprocessed_dtype': preprocessed_dtype
            })
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
        super(EarlyPreprocessing, self).__init__()
        self.random_state = random_state
        self.add_fit_requirements([
            FitRequirement('is_small_preprocess', (bool,), user_defined=True, dataset_property=True),
            FitRequirement('y_train', (pd.DataFrame,), user_defined=True,
                           dataset_property=False)])

    def transform(self, X: Dict[str, Any]) -> Dict[str, Any]:
        # TODO consider inverse transformation
        transforms = get_preprocess_transforms(X, preprocess_type=autoPyTorchTargetPreprocessingComponent)
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
