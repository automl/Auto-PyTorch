from typing import Optional, Union, List
import logging
import copy
import sklearn.utils


from autoPyTorch.utils.logging_ import PicklableClientLogger

import numpy as np

import sklearn.utils
from sklearn.base import BaseEstimator
from sklearn.exceptions import NotFittedError

from autoPyTorch.data.tabular_validator import TabularFeatureValidator


class TimeSeriesForecastingFeatureValidator(TabularFeatureValidator):
    def __init__(self,
                 logger: Optional[Union[PicklableClientLogger, logging.Logger
                 ]] = None,
                 ) -> None:
        TabularFeatureValidator.__init__(self, logger)
        self._extend_feature_dims = False

    def fit(self,
            X_train: Union[np.ndarray, List[np.ndarray]],
            X_test: Optional[np.ndarray] = None) -> BaseEstimator:
        """
        We expect a time series dataset stored in the form :[population, time_series, features]

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
        # TODO only allow np.ndarray(3D) and List of np.ndarray(2D) or array of array (2D) to reduce complexity!!!!!
        if isinstance(X_train, np.ndarray):
            if X_train.ndim > 3:
                raise ValueError(f"Number of dimensions too large for time series train data.")
            if X_train.ndim == 1:
                self.validate_ts_data(X_train)
        elif isinstance(X_train, list):
            self.validate_ts_data(X_train)
        else:
            raise ValueError(f"Time series train data must be given as a numpy array or nested list,"
                             f" but got {type(X_train)}")
        """
        _ = sklearn.utils.check_array(
            X_train,
            force_all_finite=True,
            ensure_2d=False,
            allow_nd=True,
            accept_sparse=False,
            accept_large_sparse=False
        )
        """
        if X_test is not None:
            if isinstance(X_test, np.ndarray):
                if X_test.ndim > 3:
                    raise ValueError(f"Number of dimensions too large for time series train data.")
                if X_test.ndim == 1:
                    self.validate_ts_data(X_test)
            elif isinstance(X_test, list):
                self.validate_ts_data(X_test)
            else:
                raise ValueError(f"Time series train data must be given as a numpy array or nested list,"
                                 f" but got {type(X_test)}")
            """
            _ = sklearn.utils.check_array(
                X_test,
                force_all_finite=True,
                ensure_2d=False,
                allow_nd=True,
                accept_sparse=False,
                accept_large_sparse=False
            )
            """
        first_sequence = np.array(X_train[0])

        if self._extend_feature_dims:
            first_sequence = np.expand_dims(first_sequence, axis=-1)
            self.n_feature_dims = 1
        self._fit(first_sequence)

        self._is_fitted = True

        return self

    def validate_ts_data(self, X, is_train_set=True):
        n_feature_dims = [0] * len(X)
        seq_ndims = [0] * len(X)
        for idx_seq, x in enumerate(X):
            x_array_shape = np.array(x).shape
            x_array_n_dims = len(x_array_shape)
            seq_ndims[idx_seq] = x_array_n_dims

            if x_array_n_dims == 1:
                # As lots of time series prediction tasks only have one sequence feature, we will not raise an error here
                #self.logger.warning(f"For each piece of time series data, we will automatically convert 1D vector to"
                #                    f"2D matrix!")
                self._extend_feature_dims = True
                n_feature_dims[idx_seq] = 1
            elif x_array_n_dims > 2:
                raise ValueError(f"Invalid number of dimensions for time series train data")
            else:
                n_feature_dims[idx_seq] = x_array_shape[-1]


        if not np.all(np.asarray(seq_ndims) == seq_ndims[0]):
            raise ValueError(f"All the sequence needs to have the same shape!")
        if not np.all(np.asarray(n_feature_dims) == n_feature_dims[0]):
            raise ValueError(f"Number of features does not match for all the sequence")

        if is_train_set:
            self.n_feature_dims = n_feature_dims[0]
            self.seq_ndims = seq_ndims[0]
        else:
            if seq_ndims[0] != self.seq_ndims:
                raise ValueError("number of sequence dimensions does not match for training and test sets!")
            if n_feature_dims[0] != self.n_feature_dims:
                raise ValueError("number of feature dimensions does not match for training and test sets!")

    def transform(self, X: np.ndarray) -> np.ndarray:
        """

        Arguments:
            X (np.ndarray):
                A set of data, that is going to be transformed

        Return:
            np.ndarray:
                The transformed array
        """
        if not self._is_fitted:
            raise NotFittedError("Cannot call transform on a validator that is not fitted")

        if self._extend_feature_dims:
            for seq_idx in range(len(X)):
                X[seq_idx] = np.expand_dims(X[seq_idx], axis=-1)
        return X
        """
        return sklearn.utils.check_array(
            X,
            force_all_finite=True,
            ensure_2d=False,
            allow_nd=True,
            accept_sparse=False,
            accept_large_sparse=False
        )
        """

