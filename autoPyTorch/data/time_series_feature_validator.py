import logging
from typing import Optional, Union

import numpy as np

import sklearn.utils
from sklearn.base import BaseEstimator

from autoPyTorch.utils.logging_ import PicklableClientLogger


class TimeSeriesFeatureValidator(BaseEstimator):
    def __init__(self,
                 logger: Optional[Union[PicklableClientLogger, logging.Logger]] = None) -> None:
        self.logger = logger

    def fit(self,
            X_train: np.ndarray,
            X_test: Optional[np.ndarray] = None) -> BaseEstimator:
        """

        Arguments:
            X_train (np.ndarray):
                A set of data that are going to be validated (type and dimensionality
                checks) and used for fitting

            X_test (Optional[np.ndarray]):
                An optional set of data that is going to be validated

        Returns:
            self:
                The fitted base estimator
        """

        if not isinstance(X_train, np.ndarray):
            raise ValueError("Time series train data must be given as a numpy array")

        if X_train.ndim != 3:
            raise ValueError(f"Invalid number of dimensions for time series train data, "
                             f"expected 3 but got {X_train.ndim}. "
                             f"Time series data has to be of shape [B, T, F] where B is the "
                             f"batch dimension, T is the time dimension and F are the number of features.")

        _ = sklearn.utils.check_array(
            X_train,
            force_all_finite=True,
            ensure_2d=False,
            allow_nd=True,
            accept_sparse=False,
            accept_large_sparse=False
        )

        if X_test is not None:
            if not isinstance(X_test, np.ndarray):
                raise ValueError("Time series test data must be given as a numpy array")

            if not X_test.ndim == 3:
                raise ValueError(f"Invalid number of dimensions for time series test data, "
                                 f"expected 3 but got {X_train.ndim}. "
                                 f"Time series data has to be of shape [B, T, F] where B is the "
                                 f"batch dimension, T is the time dimension and F are the number of features")

            if X_train.shape[1:] != X_test.shape[1:]:
                raise ValueError(f"Time series train and test data are expected to have the same shape except for "
                                 f"the batch dimension, but got {X_train.shape} for train data and "
                                 f"{X_test.shape} for test data")

            _ = sklearn.utils.check_array(
                X_test,
                force_all_finite=True,
                ensure_2d=False,
                allow_nd=True,
                accept_sparse=False,
                accept_large_sparse=False
            )

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """

        Arguments:
            X (np.ndarray):
                A set of data, that is going to be transformed

        Return:
            np.ndarray:
                The transformed array
        """

        return sklearn.utils.check_array(
            X,
            force_all_finite=True,
            ensure_2d=False,
            allow_nd=True,
            accept_sparse=False,
            accept_large_sparse=False
        )
