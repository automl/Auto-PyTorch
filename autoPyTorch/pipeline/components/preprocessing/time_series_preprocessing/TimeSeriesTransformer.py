from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

import pandas as pd

from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline

from autoPyTorch.pipeline.components.preprocessing.time_series_preprocessing.base_time_series_preprocessing import (
    autoPyTorchTimeSeriesPreprocessingComponent,
    autoPyTorchTimeSeriesTargetPreprocessingComponent)
from autoPyTorch.pipeline.components.preprocessing.time_series_preprocessing.utils import (
    get_time_series_preprocessers, get_time_series_target_preprocessers)
from autoPyTorch.utils.common import FitRequirement


class TimeSeriesFeatureTransformer(autoPyTorchTimeSeriesPreprocessingComponent):
    def __init__(self, random_state: Optional[Union[np.random.RandomState, int]] = None):
        super().__init__()
        self.random_state = random_state
        self.preprocessor: Optional[ColumnTransformer] = None
        self.add_fit_requirements([
            FitRequirement('numerical_features', (List,), user_defined=True, dataset_property=True),
            FitRequirement('categorical_features', (List,), user_defined=True, dataset_property=True)])

    def fit(self, X: Dict[str, Any], y: Any = None) -> BaseEstimator:
        """
        Creates a column transformer for the chosen tabular
        preprocessors

        Args:
            X (Dict[str, Any]): fit dictionary

        Returns:
            "TimeSeriesFeatureTransformer": an instance of self
        """
        self.check_requirements(X, y)

        preprocessors = get_time_series_preprocessers(X)
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
            X_train = X['X_train']
        else:
            X_train = X['backend'].load_datamanager().train_tensors[0]

        self.preprocessor.fit(X_train)
        return self

    def transform(self, X: Dict[str, Any]) -> Dict[str, Any]:
        """
        Adds the time series transformer to fit dictionary

        Args:
            X (Dict[str, Any]): fit dictionary

        Returns:
            X (Dict[str, Any]): updated fit dictionary
        """
        X.update({'time_series_feature_transformer': self})
        return X

    def __call__(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.preprocessor is None:
            raise ValueError("cant call {} without fitting the column transformer first."
                             .format(self.__class__.__name__))

        return self.preprocessor.transform(X)

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


class TimeSeriesTargetTransformer(autoPyTorchTimeSeriesTargetPreprocessingComponent):
    def __init__(self, random_state: Optional[Union[np.random.RandomState, int]] = None):
        super().__init__()
        self.random_state = random_state
        self.preprocessor: Optional[ColumnTransformer] = None

    def fit(self, X: Dict[str, Any], y: Any = None) -> BaseEstimator:
        """
        Creates a column transformer for the chosen tabular
        preprocessors

        Args:
            X (Dict[str, Any]): fit dictionary

        Returns:
            "TimeSeriesTargetTransformer": an instance of self
        """
        self.check_requirements(X, y)

        preprocessors = get_time_series_target_preprocessers(X)
        column_transformers: List[Tuple[str, BaseEstimator, List[int]]] = []
        if len(preprocessors['target_numerical']) > 0:
            numerical_pipeline = make_pipeline(*preprocessors['target_numerical'])
            # TODO the last item needs to be adapted accordingly!
            column_transformers.append(
                ('target_numerical_pipeline', numerical_pipeline, list(range(len(preprocessors['target_numerical']))))
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
        if 'y_train' in X:
            y_train = X['y_train']
        else:
            y_train = X['backend'].load_datamanager().train_tensors[1]

        self.preprocessor.fit(y_train)
        return self

    def transform(self, X: Dict[str, Any]) -> Dict[str, Any]:
        """
        Adds the time series transformer to fit dictionary
        Args:
            X (Dict[str, Any]): fit dictionary

        Returns:
            X (Dict[str, Any]): updated fit dictionary
        """
        X.update({'time_series_target_transformer': self})
        return X

    def __call__(self, y: pd.DataFrame) -> pd.DataFrame:
        if self.preprocessor is None:
            raise ValueError("cant call {} without fitting the column transformer first."
                             .format(self.__class__.__name__))

        return self.preprocessor.transform(y)

    def get_target_transformer(self) -> ColumnTransformer:
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
