import logging
from typing import Optional, Union, Tuple, Sequence
import pandas as pd
import numpy as np

import sklearn.utils
from sklearn.base import BaseEstimator
from sklearn.exceptions import NotFittedError
from autoPyTorch.data.tabular_feature_validator import TabularFeatureValidator
from autoPyTorch.utils.logging_ import PicklableClientLogger


class TimeSeriesFeatureValidator(TabularFeatureValidator):
    def __init__(
        self,
        logger: Optional[Union[PicklableClientLogger, logging.Logger]] = None,
    ):
        super().__init__(logger)
        self.only_contain_series_idx = True

    def fit(self,
            X_train: Union[pd.DataFrame, np.ndarray],
            X_test: Union[pd.DataFrame, np.ndarray] = None,
            series_idx: Optional[Union[Tuple[Union[str, int]]]] = None) -> BaseEstimator:
        """

        Arguments:
            X_train (Union[pd.DataFrame, np.ndarray]):
                A set of data that are going to be validated (type and dimensionality
                checks) and used for fitting

            X_test (Union[pd.DataFrame, np.ndarray]):
                An optional set of data that is going to be validated

            series_idx (Optional[Union[str, int]]):
                Series Index, to identify each individual series

        Returns:
            self:
                The fitted base estimator
        """
        if series_idx is not None:
            # remove series idx as they are not part of features
            if isinstance(X_train, pd.DataFrame):
                for series_id in series_idx:
                    if series_id not in X_train.columns:
                        raise ValueError(f"All Series ID must be contained in the training column, however, {series_id}"
                                         f"is not part of {X_train.columns.tolist()}")
                self.only_contain_series_idx = len(X_train.columns) == series_idx
                if self.only_contain_series_idx:
                    self._is_fitted = True

                    self.num_features = 0
                    self.numerical_columns = []
                    self.categorical_columns = []

                X_train_ = X_train.drop(series_idx, axis=1)

                X_test_ = X_test.drop(series_idx, axis=1) if X_test is not None else None

                super().fit(X_train_, X_test_)
            else:
                raise NotImplementedError(f"series idx only works with pandas.DataFrame but the type of "
                                          f"X_train is {type(X_train)} ")
        else:
            super().fit(X_train, X_test)

        return self

