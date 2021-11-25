from typing import List, Optional, Union, cast

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


class TabularTargetValidator(BaseTargetValidator):
    def _fit(
        self,
        y_train: SUPPORTED_TARGET_TYPES,
        y_test: Optional[SUPPORTED_TARGET_TYPES] = None,
    ) -> BaseEstimator:
        """
        If dealing with classification, this utility encodes the targets.

        It does so by also using the classes from the test data, to prevent encoding
        errors

        Args:
            y_train (SUPPORTED_TARGET_TYPES)
                The labels of the current task. They are going to be encoded in case
                of classification
            y_test (Optional[SUPPORTED_TARGET_TYPES])
                A holdout set of labels
        """
        if not self.is_classification or self.type_of_target == 'multilabel-indicator':
            # Only fit an encoder for classification tasks
            # Also, encoding multilabel indicator data makes the data multiclass
            # Let the user employ a MultiLabelBinarizer if needed
            return self

        if y_test is not None:
            if hasattr(y_train, "iloc"):
                y_train = pd.concat([y_train, y_test], ignore_index=True, sort=False)
            elif isinstance(y_train, list):
                y_train = y_train + y_test
            elif isinstance(y_train, np.ndarray):
                y_train = np.concatenate((y_train, y_test))

        ndim = len(np.shape(y_train))
        if ndim == 1 or (ndim > 1 and np.shape(y_train)[1] == 1):
            # The label encoder makes sure data is, and remains
            # 1 dimensional
            self.encoder = preprocessing.OrdinalEncoder(handle_unknown='use_encoded_value',
                                                        unknown_value=-1)
        else:
            # We should not reach this if statement as we check for type of targets before
            raise ValueError("Multi-dimensional classification is not yet supported. "
                             "Encoding multidimensional data converts multiple columns "
                             "to a 1 dimensional encoding. Data involved = {}/{}".format(
                                 np.shape(y_train),
                                 self.type_of_target
                             ))

        # Mypy redefinition
        assert self.encoder is not None

        # remove ravel warning from pandas Series
        if ndim > 1:
            self.encoder.fit(y_train)
        else:
            if hasattr(y_train, 'iloc'):
                y_train = cast(pd.DataFrame, y_train)
                self.encoder.fit(y_train.to_numpy().reshape(-1, 1))
            else:
                self.encoder.fit(np.array(y_train).reshape(-1, 1))

        # we leave objects unchanged, so no need to store dtype in this case
        if hasattr(y_train, 'dtype'):
            # Series and numpy arrays are checked here
            # Cast is as numpy for mypy checks
            y_train = cast(np.ndarray, y_train)
            if is_numeric_dtype(y_train.dtype):
                self.dtype = y_train.dtype
        elif (
            hasattr(y_train, 'dtypes')
            and is_numeric_dtype(cast(pd.DataFrame, y_train).dtypes[0])
        ):
            # This case is for pandas array with a single column
            y_train = cast(pd.DataFrame, y_train)
            self.dtype = y_train.dtypes[0]

        return self

    def transform(
        self,
        y: Union[SUPPORTED_TARGET_TYPES],
    ) -> np.ndarray:
        """
        Validates and fit a categorical encoder (if needed) to the features.
        The supported data types are List, numpy arrays and pandas DataFrames.

        Args:
            y (SUPPORTED_TARGET_TYPES)
                A set of targets that are going to be encoded if the current task
                is classification

        Returns:
            np.ndarray:
                The transformed array
        """
        if not self._is_fitted:
            raise NotFittedError("Cannot call transform on a validator that is not fitted")

        # Check the data here so we catch problems on new test data
        self._check_data(y)

        if self.encoder is not None:
            # remove ravel warning from pandas Series
            shape = np.shape(y)
            if len(shape) > 1:
                y = self.encoder.transform(y)
            else:
                # The Ordinal encoder expects a 2 dimensional input.
                # The targets are 1 dimensional, so reshape to match the expected shape
                if hasattr(y, 'iloc'):
                    y = cast(pd.DataFrame, y)
                    y = self.encoder.transform(y.to_numpy().reshape(-1, 1)).reshape(-1)
                else:
                    y = self.encoder.transform(np.array(y).reshape(-1, 1)).reshape(-1)

        # sklearn check array will make sure we have the
        # correct numerical features for the array
        # Also, a numpy array will be created
        y = sklearn.utils.check_array(
            y,
            force_all_finite=True,
            accept_sparse='csr',
            ensure_2d=False,
        )

        # When translating a dataframe to numpy, make sure we
        # honor the ravel requirement
        if y.ndim == 2 and y.shape[1] == 1:
            y = np.ravel(y)

        return y

    def inverse_transform(
        self,
        y: SUPPORTED_TARGET_TYPES,
    ) -> np.ndarray:
        """
        Revert any encoding transformation done on a target array

        Args:
            y (Union[np.ndarray, pd.DataFrame, pd.Series]):
                Target array to be transformed back to original form before encoding
        Returns:
            np.ndarray:
                The transformed array
        """
        if not self._is_fitted:
            raise NotFittedError("Cannot call inverse_transform on a validator that is not fitted")

        if self.encoder is None:
            return y
        shape = np.shape(y)
        if len(shape) > 1:
            y = self.encoder.inverse_transform(y)
        else:
            # The targets should be a flattened array, hence reshape with -1
            if hasattr(y, 'iloc'):
                y = cast(pd.DataFrame, y)
                y = self.encoder.inverse_transform(y.to_numpy().reshape(-1, 1)).reshape(-1)
            else:
                y = self.encoder.inverse_transform(np.array(y).reshape(-1, 1)).reshape(-1)

        # Inverse transform returns a numpy array of type object
        # This breaks certain metrics as accuracy, which makes type_of_target be unknown
        # If while fit a dtype was observed, we try to honor that dtype
        if self.dtype is not None:
            y = y.astype(self.dtype)
        return y

    def _check_data(
        self,
        y: SUPPORTED_TARGET_TYPES,
    ) -> None:
        """
        Perform dimensionality and data type checks on the targets

        Args:
            y (Union[np.ndarray, pd.DataFrame, pd.Series]):
                A set of features whose dimensionality and data type is going to be checked
        """

        if not isinstance(y, (np.ndarray, pd.DataFrame,
                              List, pd.Series)) \
                and not scipy.sparse.issparse(y):  # type: ignore[misc]
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
            has_nan_values = cast(pd.DataFrame, y).isnull().values.any()
        if scipy.sparse.issparse(y):
            y = cast(scipy.sparse.spmatrix, y)
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
