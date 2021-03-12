from autoPyTorch.data.tabular_target_validator import TabularTargetValidator

import typing

import numpy as np

import pandas as pd
from pandas.api.types import is_numeric_dtype

import scipy.sparse

import sklearn.utils
from sklearn import preprocessing
from sklearn.base import BaseEstimator
from sklearn.exceptions import NotFittedError
from sklearn.utils.multiclass import type_of_target

from autoPyTorch.data.base_target_validator import BaseTargetValidator, SUPPORTED_TARGET_TYPES


class TimeSeriesForecastingTargetValidator(TabularTargetValidator):
    def transform(
            self,
            y: typing.Union[SUPPORTED_TARGET_TYPES],
    ) -> np.ndarray:
        if not self._is_fitted:
            raise NotFittedError("Cannot call transform on a validator that is not fitted")

        # Check the data here so we catch problems on new test data
        self._check_data(y)

        # sklearn check array will make sure we have the
        # correct numerical features for the array
        # Also, a numpy array will be created
        y = sklearn.utils.check_array(
            y,
            force_all_finite=True,
            ensure_2d=False,
            allow_nd=True,
            accept_sparse=False,
            accept_large_sparse=False
        )
        return y

    """
    Validator for time series forecasting, currently only consider regression tasks
    TODO: Considering Classification Validator
    """
    def _check_data(
        self,
        y: SUPPORTED_TARGET_TYPES,
    ) -> None:
        """
        Perform dimensionality and data type checks on the targets

        Arguments:
            y (typing.Union[np.ndarray, pd.DataFrame, pd.Series]):
                A set of features whose dimensionality and data type is going to be checked
        """

        if not isinstance(
                y, (np.ndarray, pd.DataFrame, list, pd.Series)) and not scipy.sparse.issparse(y):
            raise ValueError("AutoPyTorch only supports Numpy arrays, Pandas DataFrames,"
                             " pd.Series, sparse data and Python Lists as targets, yet, "
                             "the provided input is of type {}".format(
                                 type(y)
                             ))

        # Sparse data muss be numerical
        # Type ignore on attribute because sparse targets have a dtype
        if scipy.sparse.issparse(y) and not np.issubdtype(y.dtype.type,  # type: ignore[union-attr]
                                                          np.number):
            raise ValueError("When providing a sparse matrix as targets, the only supported "
                             "values are numerical. Please consider using a dense"
                             " instead."
                             )

        if self.data_type is None:
            self.data_type = type(y)
        if self.data_type != type(y):
            self.logger.warning("AutoPyTorch previously received targets of type %s "
                                "yet the current features have type %s. Changing the dtype "
                                "of inputs to an estimator might cause problems" % (
                                    str(self.data_type),
                                    str(type(y)),
                                ),
                                )

        # No Nan is supported
        has_nan_values = False
        if hasattr(y, 'iloc'):
            has_nan_values = typing.cast(pd.DataFrame, y).isnull().values.any()
        if scipy.sparse.issparse(y):
            y = typing.cast(scipy.sparse.spmatrix, y)
            has_nan_values = not np.array_equal(y.data, y.data)
        else:
            # List and array like values are considered here
            # np.isnan cannot work on strings, so we have to check for every element
            # but NaN, are not equal to themselves:
            has_nan_values = not np.array_equal(y, y)
        if has_nan_values:
            raise ValueError("Target values cannot contain missing/NaN values. "
                             "This is not supported by scikit-learn. "
                             )

        # Pandas Series is not supported for multi-label indicator
        # This format checks are done by type of target
        try:
            self.type_of_target = type_of_target(y[0])
        except Exception as e:
            raise ValueError("The provided data could not be interpreted by AutoPyTorch. "
                             "While determining the type of the targets via type_of_target "
                             "run into exception: {}.".format(e))

        supported_output_types = ('binary',
                                  'continuous',
                                  'continuous-multioutput',
                                  'multiclass',
                                  'multilabel-indicator',
                                  # Notice unknown/multiclass-multioutput are not supported
                                  # This can only happen during testing only as estimators
                                  # should filter out unsupported types.
                                  )
        if self.type_of_target not in supported_output_types:
            raise ValueError("Provided targets are not supported by AutoPyTorch. "
                             "Provided type is {} whereas supported types are {}.".format(
                                 self.type_of_target,
                                 supported_output_types
                             ))


