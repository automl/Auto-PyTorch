from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline

import torch

from autoPyTorch.pipeline.components.preprocessing.tabular_preprocessing.base_tabular_preprocessing import (
    autoPyTorchTabularPreprocessingComponent
)
from autoPyTorch.pipeline.components.preprocessing.tabular_preprocessing.utils import get_tabular_preprocessers
from autoPyTorch.utils.common import FitRequirement, subsampler


class TabularColumnTransformer(autoPyTorchTabularPreprocessingComponent):

    def __init__(self, random_state: Optional[Union[np.random.RandomState, int]] = None):
        super().__init__()
        self.random_state = random_state
        self.preprocessor: Optional[ColumnTransformer] = None
        self.add_fit_requirements([
            FitRequirement('numerical_columns', (List,), user_defined=True, dataset_property=True),
            FitRequirement('categorical_columns', (List,), user_defined=True, dataset_property=True)])

    def get_column_transformer(self) -> ColumnTransformer:
        """
        Get fitted column transformer that is wrapped around
        the sklearn early_preprocessor. Can only be called if fit()
        has been called on the object.
        Returns:
            BaseEstimator: Fitted sklearn column transformer
        """
        if self.preprocessor is None:
            raise AttributeError("{} can't return column transformer before transform is called"
                                 .format(self.__class__.__name__))
        return self.preprocessor

    def fit(self, X: Dict[str, Any], y: Any = None) -> "TabularColumnTransformer":
        """
        Creates a column transformer for the chosen tabular
        preprocessors
        Args:
            X (Dict[str, Any]): fit dictionary

        Returns:
            "TabularColumnTransformer": an instance of self
        """
        self.check_requirements(X, y)

        preprocessors = get_tabular_preprocessers(X)
        column_transformers: List[Tuple[str, BaseEstimator, List[int]]] = []
        if len(preprocessors['numerical']) > 0:
            numerical_pipeline = make_pipeline(*preprocessors['numerical'])
            column_transformers.append(
                ('numerical_pipeline', numerical_pipeline, X['dataset_properties']['numerical_columns'])
            )
        if len(preprocessors['categorical']) > 0:
            categorical_pipeline = make_pipeline(*preprocessors['categorical'])
            column_transformers.append(
                ('categorical_pipeline', categorical_pipeline, X['dataset_properties']['categorical_columns'])
            )

        # in case the preprocessing steps are disabled
        # i.e, NoEncoder for categorical, we want to
        # let the data in categorical columns pass through
        self.preprocessor = ColumnTransformer(
            column_transformers,
            remainder='passthrough'
        )

        # Where to get the data -- Prioritize X_train if any else
        # get from backend
        if 'X_train' in X:
            X_train = subsampler(X['X_train'], X['train_indices'])
        else:
            X_train = X['backend'].load_datamanager().train_tensors[0]

        if 'y_train' in X:
            y_train = subsampler(X['y_train'], X['train_indices'])
        else:
            y_train = X['backend'].load_datamanager().train_tensors[1]

        self.preprocessor.fit(X_train, y=y_train)

        return self

    def transform(self, X: Dict[str, Any]) -> Dict[str, Any]:
        """
        Adds the column transformer to fit dictionary
        Args:
            X (Dict[str, Any]): fit dictionary

        Returns:
            X (Dict[str, Any]): updated fit dictionary
        """
        X.update({'tabular_transformer': self})
        return X

    def __call__(self, X: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:

        if self.preprocessor is None:
            raise ValueError("cant call {} without fitting the column transformer first."
                             .format(self.__class__.__name__))

        if len(X.shape) == 1:
            # expand batch dimension when called on a single record
            X = X[np.newaxis, ...]

        return self.preprocessor.transform(X)
