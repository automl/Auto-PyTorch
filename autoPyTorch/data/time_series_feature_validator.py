import logging
from typing import List, Optional, Tuple, Union


import numpy as np

import pandas as pd

from scipy.sparse import issparse

from sklearn.base import BaseEstimator
from sklearn.preprocessing import OrdinalEncoder


from autoPyTorch.data.tabular_feature_validator import TabularFeatureValidator
from autoPyTorch.utils.logging_ import PicklableClientLogger


def df2index(df: Union[pd.DataFrame, pd.Series]) -> np.ndarray:
    if isinstance(df, pd.Series):
        seq_lengths = df.value_counts().values
    else:
        seq_lengths = np.unique(
            OrdinalEncoder().fit_transform(df), axis=0, return_counts=True
        )[1]
    return np.arange(len(seq_lengths)).repeat(seq_lengths)


class TimeSeriesFeatureValidator(TabularFeatureValidator):
    def __init__(
        self,
        logger: Optional[Union[PicklableClientLogger, logging.Logger]] = None,
    ):
        super().__init__(logger)
        self.only_contain_series_idx = False
        self.static_features: Union[Tuple[()], Tuple[Union[int, str]]] = ()
        self.series_idx: Optional[List[Union[str, int]]] = None

    def get_reordered_columns(self) -> List[str]:
        return self.transformed_columns + [
            col for col in self.column_order if col not in set(self.transformed_columns)
        ]

    def fit(
        self,
        X_train: Union[pd.DataFrame, np.ndarray],
        X_test: Union[pd.DataFrame, np.ndarray] = None,
        series_idx: Optional[Union[List[Union[str, int]]]] = None,
        sequence_lengths: Optional[List[int]] = None,
    ) -> BaseEstimator:
        """

        Arguments:
            X_train (Union[pd.DataFrame, np.ndarray]):
                A set of data that are going to be validated (type and dimensionality
                checks) and used for fitting

            X_test (Union[pd.DataFrame, np.ndarray]):
                An optional set of data that is going to be validated

            series_idx (Optional[List[Union[str, int]]]):
                Series Index, to identify each individual series

            sequence_lengths (Optional[List[int]]):
                Length of each sequence

        Returns:
            self:
                The fitted base estimator
        """
        if issparse(X_train):
            raise NotImplementedError(
                "Sparse matrix is currently unsupported for Forecasting tasks"
            )
        index = None

        if series_idx is not None:
            self.series_idx = series_idx

            # remove series idx as they are not part of features
            # TODO consider them as static features?
            if isinstance(X_train, pd.DataFrame):
                for series_id in series_idx:
                    if series_id not in X_train.columns:
                        raise ValueError(
                            f"All Series ID must be contained in the training column, however, {series_id}"
                            f"is not part of {X_train.columns.tolist()}"
                        )
                if X_train[list(series_idx)].isnull().values.any():
                    raise ValueError("NaN should not exit in Series ID!")
                index = df2index(df=X_train[series_idx])

                self.only_contain_series_idx = len(X_train.columns) == len(series_idx)

                if self.only_contain_series_idx:
                    self._is_fitted = True

                    self.num_features = 0
                    self.numerical_columns = []
                    self.categorical_columns = []
                    return self

                X_train = X_train.drop(series_idx, axis=1)
                X_train.index = index

                if X_test is not None:
                    assert isinstance(X_test, pd.DataFrame)
                    index = df2index(df=X_test[series_idx])
                    X_test = X_test.drop(series_idx, axis=1)
                    X_test.index = index

                super().fit(X_train, X_test)
            else:
                raise NotImplementedError(
                    f"series idx only works with pandas.DataFrame but the type of "
                    f"X_train is {type(X_train)} "
                )
        else:
            super().fit(X_train, X_test)

        X_train_has_idx = isinstance(X_train, pd.DataFrame)
        X_train = pd.DataFrame(X_train)
        if index is None:
            if sequence_lengths is None:
                if not X_train_has_idx:
                    index = np.zeros(len(X_train))
                else:
                    index = X_train.index
            else:
                if np.sum(sequence_lengths) != len(X_train):
                    raise ValueError(
                        "The Sum of Sequence length must equal to the length of hte dataset"
                    )
                index = np.arange(len(sequence_lengths)).repeat(sequence_lengths)
        X_train.index = index

        static_features: pd.Series = (
            X_train.groupby(X_train.index).nunique() <= 1
        ).all()
        self.static_features = tuple(   # type: ignore[assignment]
            idx for idx in static_features.index if static_features[idx]
        )
        return self

    def transform(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        index: Optional[Union[pd.Index, np.ndarray]] = None,
    ) -> Optional[pd.DataFrame]:
        if self.only_contain_series_idx:
            return None
        if self.series_idx is not None:
            if isinstance(X, pd.DataFrame):
                X = X.drop(self.series_idx, axis=1)
            else:
                raise NotImplementedError(
                    f"series idx only works with pandas.DataFrame but the type of "
                    f"X_train is {type(X)} "
                )
        X_has_idx = isinstance(X, pd.DataFrame)
        if X_has_idx and index is None:
            index = X.index
        X = super(TimeSeriesFeatureValidator, self).transform(X)
        if X.ndim == 1:
            X = np.expand_dims(X, -1)  # type:ignore[no-redef]
        X: pd.DataFrame = pd.DataFrame(X, columns=self.get_reordered_columns())  # type:ignore[no-redef]
        if index is None:
            if not X_has_idx:
                index = np.array([0] * len(X))
        else:
            if len(index) != X.shape[0]:
                raise ValueError("Given index must have length as the input features!")
        X.index = index
        return X
