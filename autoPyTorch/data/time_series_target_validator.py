from typing import List, Optional, Union, cast

import numpy as np

import pandas as pd
from pandas.api.types import is_numeric_dtype

from scipy.sparse import issparse, spmatrix

import sklearn.utils
from sklearn import preprocessing
from sklearn.base import BaseEstimator
from sklearn.exceptions import NotFittedError
from sklearn.utils.multiclass import type_of_target

from autoPyTorch.data.base_target_validator import BaseTargetValidator, SupportedTargetTypes
from autoPyTorch.utils.common import ispandas
from autoPyTorch.data.tabular_target_validator import TabularTargetValidator, ArrayType


def _check_and_to_array(y: SupportedTargetTypes) -> ArrayType:
    """ sklearn check array will make sure we have the correct numerical features for the array """
    return sklearn.utils.check_array(y, force_all_finite=False, accept_sparse='csr', ensure_2d=False)


def _modify_regression_target(y: ArrayType) -> ArrayType:
    # Regression targets must have numbers after a decimal point.
    # Ref: https://github.com/scikit-learn/scikit-learn/issues/8952
    y_min = np.abs(np.nan_to_num(y, 1e12)).min()
    offset = max(y_min, 1e-13) * 1e-13  # Sufficiently small number
    if y_min > 1e12:
        raise ValueError(
            "The minimum value for the target labels of regression tasks must be smaller than "
            f"1e12 to avoid errors caused by an overflow, but got {y_min}"
        )

    # Since it is all integer, we can just add a random small number
    if isinstance(y, np.ndarray):
        y = y.astype(dtype=np.float64) + offset
    else:
        y.data = y.data.astype(dtype=np.float64) + offset

    return y


class TimeSeriesTargetValidator(TabularTargetValidator):
    def transform(self,
                  y: SupportedTargetTypes,
                  index: Optional[Union[pd.Index, np.ndarray]] = None,
                  ) ->pd.DataFrame:
        """
        Validates and fit a categorical encoder (if needed) to the features.
        The supported data types are List, numpy arrays and pandas DataFrames.

        Args:
            y (SupportedTargetTypes)
                A set of targets that are going to be encoded if the current task
                is classification
            index (Optional[Union[pd.Index], np.ndarray]):
                index indentifying which series the data belongs to

        Returns:
            pd.DataFrame:
                The transformed array
        """
        if not self._is_fitted:
            raise NotFittedError("Cannot call transform on a validator that is not fitted")

        # Check the data here so we catch problems on new test data
        self._check_data(y)
        y = self._transform_by_encoder(y)

        # When translating a dataframe to numpy, make sure we honor the ravel requirement
        if y.ndim == 2 and y.shape[1] == 1:
            y = np.ravel(y)

        if not self.is_classification and "continuous" not in type_of_target(np.nan_to_num(y)):
            y = _modify_regression_target(y)

        if index is None:
            index = np.array([0.] * len(y))
        if y.ndim == 1:
            y = np.expand_dims(y, -1)

        y: pd.DataFrame = pd.DataFrame(y)
        y.index = index

        return y


    def _check_data(self, y: SupportedTargetTypes) -> None:
        """
        Perform dimensionality and data type checks on the targets, This is nearly the same as
        TabularTargetValidator._check_data, however, we allow NAN values in target

        Args:
            y (SupportedTargetTypes):
                A set of features whose dimensionality and data type is going to be checked
        """
        if not isinstance(y, (np.ndarray, pd.DataFrame,
                              List, pd.Series)) \
                and not issparse(y):  # type: ignore[misc]
            raise ValueError("AutoPyTorch only supports Numpy arrays, Pandas DataFrames,"
                             " pd.Series, sparse data and Python Lists as targets, yet, "
                             "the provided input is of type {}".format(
                                 type(y)
                             ))

        # Sparse data muss be numerical
        # Type ignore on attribute because sparse targets have a dtype
        if issparse(y) and not np.issubdtype(y.dtype.type,  # type: ignore[union-attr]
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
        if ispandas(y):
            has_nan_values = cast(pd.DataFrame, y).isnull().values.any()
            if has_nan_values:
                y = cast(pd.DataFrame, y).fillna(method='pad')
        if issparse(y):
            y = cast(spmatrix, y)
            has_nan_values = not np.array_equal(y.data, y.data)
            if has_nan_values:
                type_y = type(y)
                y = type_y(np.nan_to_num(y.todense()))
        else:
            # List and array like values are considered here
            # np.isnan cannot work on strings, so we have to check for every element
            # but NaN, are not equal to themselves:
            has_nan_values = not np.array_equal(y, y)
            if has_nan_values:
                y = np.nan_to_num(y)

        # Pandas Series is not supported for multi-label indicator
        # This format checks are done by type of target
        try:
            self.type_of_target = type_of_target(y)
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

    def _transform_by_encoder(self, y: SupportedTargetTypes) -> np.ndarray:
        if self.encoder is None:
            return _check_and_to_array(y)

        # remove ravel warning from pandas Series
        shape = np.shape(y)
        if len(shape) > 1:
            y = self.encoder.transform(y)
        elif ispandas(y):
            # The Ordinal encoder expects a 2 dimensional input.
            # The targets are 1 dimensional, so reshape to match the expected shape
            y = cast(pd.DataFrame, y)
            y = self.encoder.transform(y.to_numpy().reshape(-1, 1)).reshape(-1)
        else:
            y = self.encoder.transform(np.array(y).reshape(-1, 1)).reshape(-1)

        return _check_and_to_array(y)