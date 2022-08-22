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
    get_time_series_preprocessors, get_time_series_target_preprocessers)
from autoPyTorch.utils.common import FitRequirement


class TimeSeriesFeatureTransformer(autoPyTorchTimeSeriesPreprocessingComponent):
    def __init__(self, random_state: Optional[Union[np.random.RandomState, int]] = None):
        super().__init__()
        self.random_state = random_state
        self.preprocessor: Optional[ColumnTransformer] = None
        self.add_fit_requirements([
            FitRequirement('numerical_features', (List,), user_defined=True, dataset_property=True),
            FitRequirement('categorical_features', (List,), user_defined=True, dataset_property=True)])
        self.output_feature_order = None

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

        preprocessors = get_time_series_preprocessors(X)
        column_transformers: List[Tuple[str, BaseEstimator, List[int]]] = []

        numerical_pipeline = 'passthrough'
        encode_pipeline = 'passthrough'

        if len(preprocessors['numerical']) > 0:
            numerical_pipeline = make_pipeline(*preprocessors['numerical'])

        column_transformers.append(
            ('numerical_pipeline', numerical_pipeline, X['dataset_properties']['numerical_columns'])
        )

        if len(preprocessors['encode']) > 0:
            encode_pipeline = make_pipeline(*preprocessors['encode'])

        column_transformers.append(
            ('encode_pipeline', encode_pipeline, X['encode_columns'])
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
        self.output_feature_order = self.get_output_column_orders(len(X['dataset_properties']['feature_names']))
        return self

    def transform(self, X: Dict[str, Any]) -> Dict[str, Any]:
        """
        Adds the time series transformer to fit dictionary

        Args:
            X (Dict[str, Any]): fit dictionary

        Returns:
            X (Dict[str, Any]): updated fit dictionary
        """
        X.update({'time_series_feature_transformer': self,
                  'feature_order_after_preprocessing': self.output_feature_order})
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

    def get_output_column_orders(self, n_input_columns: int) -> List[int]:
        """
        get the order of the output features transformed by self.preprocessor
        TODO: replace this function with self.preprocessor.get_feature_names_out() when switch to sklearn 1.0 !

        Args:
            n_input_columns (int): number of input columns that will be transformed

        Returns:
            np.ndarray: a list of index indicating the order of each columns after transformation. Its length should
                equal to n_input_columns
        """
        if self.preprocessor is None:
            raise ValueError("cant call {} without fitting the column transformer first."
                             .format(self.__class__.__name__))
        transformers = self.preprocessor.transformers

        n_reordered_input = np.arange(n_input_columns)
        processed_columns = np.asarray([], dtype=np.int)

        for tran in transformers:
            trans_columns = np.array(tran[-1], dtype=np.int)
            unprocessed_columns = np.setdiff1d(processed_columns, trans_columns)
            processed_columns = np.hstack([unprocessed_columns, trans_columns])
        unprocessed_columns = np.setdiff1d(n_reordered_input, processed_columns)
        return np.hstack([processed_columns, unprocessed_columns]).tolist()  # type: ignore[return-value]


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
