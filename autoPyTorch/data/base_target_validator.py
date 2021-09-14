"""Base class for the target (or label) validator given a task
* A wrapper class of the sklearn.base.BaseEstimator
* The target validator for each task inherits this class
* Check if the provided targets (or labels) are compatible in both
  training and test

TODO:
    * SUPPORTED_FEAT_TYPES --> Enumerator
    * Describe the shape of y
    * typing.<type> --> <type>
    * logging.Logger --> Logger
    * Rename classes_ --> get_classes
    * Check the return of classes_
    * is_single_column_target --> is_target_scalar
"""

import logging
import typing

import numpy as np

import pandas as pd

import scipy.sparse

from sklearn.base import BaseEstimator

from autoPyTorch.utils.logging_ import PicklableClientLogger


SUPPORTED_TARGET_TYPES = typing.Union[
    typing.List,
    pd.Series,
    pd.DataFrame,
    np.ndarray,
    scipy.sparse.bsr_matrix,
    scipy.sparse.coo_matrix,
    scipy.sparse.csc_matrix,
    scipy.sparse.csr_matrix,
    scipy.sparse.dia_matrix,
    scipy.sparse.dok_matrix,
    scipy.sparse.lil_matrix,
]


class BaseTargetValidator(BaseEstimator):
    """
    A class to pre-process targets. It validates the data provided during fit (to make sure
    it matches AutoPyTorch expectation) as well as encoding the targets in case of classification

    Attributes:
        is_classification (bool):
            A bool that indicates if the validator should operate in classification mode.
            During classification, the targets are encoded.
        encoder (typing.Optional[BaseEstimator]):
            Host an encoder object if the data requires transformation (for example,
            if provided a categorical column in a pandas DataFrame)
        enc_columns (typing.List[str])
            List of columns that where encoded
    """
    def __init__(self,
                 is_classification: bool = False,
                 logger: typing.Optional[typing.Union[PicklableClientLogger, logging.Logger
                                                      ]] = None,
                 ) -> None:
        self.is_classification = is_classification

        self.data_type = None  # type: typing.Optional[type]

        self.encoder = None  # type: typing.Optional[BaseEstimator]

        self.out_dimensionality = None  # type: typing.Optional[int]
        self.type_of_target = None  # type: typing.Optional[str]

        self.logger: typing.Union[
            PicklableClientLogger, logging.Logger
        ] = logger if logger is not None else logging.getLogger(__name__)

        # Store the dtype for remapping to correct type
        self.dtype = None  # type: typing.Optional[type]

        self._is_fitted = False

    def fit(
        self,
        y_train: SUPPORTED_TARGET_TYPES,
        y_test: typing.Optional[SUPPORTED_TARGET_TYPES] = None,
    ) -> BaseEstimator:
        """
        Validates and fit a categorical encoder (if needed) to the targets
        The supported data types are List, numpy arrays and pandas DataFrames.

        Arguments:
            y_train (SUPPORTED_TARGET_TYPES)
                A set of targets set aside for training
            y_test (typing.Union[SUPPORTED_TARGET_TYPES])
                A hold out set of data used of the targets. It is also used to fit the
                categories of the encoder.
        """
        # Check that the data is valid
        self._check_data(y_train)

        shape = np.shape(y_train)
        if y_test is not None:
            self._check_data(y_test)

            if len(shape) != len(np.shape(y_test)) or (
                    len(shape) > 1 and (shape[1] != np.shape(y_test)[1])):
                raise ValueError("The dimensionality of the train and test targets "
                                 "does not match train({}) != test({})".format(
                                     np.shape(y_train),
                                     np.shape(y_test)
                                 ))
            if isinstance(y_train, pd.DataFrame):
                y_test = typing.cast(pd.DataFrame, y_test)
                if y_train.columns.tolist() != y_test.columns.tolist():
                    raise ValueError(
                        "Train and test targets must both have the same columns, yet "
                        "y={} and y_test={} ".format(
                            y_train.columns,
                            y_test.columns
                        )
                    )

                if list(y_train.dtypes) != list(y_test.dtypes):
                    raise ValueError("Train and test targets must both have the same dtypes")

        if self.out_dimensionality is None:
            self.out_dimensionality = 1 if len(shape) == 1 else shape[1]
        else:
            _n_outputs = 1 if len(shape) == 1 else shape[1]
            if self.out_dimensionality != _n_outputs:
                raise ValueError('Number of outputs changed from %d to %d!' %
                                 (self.out_dimensionality, _n_outputs))

        # Fit on the training data
        self._fit(y_train, y_test)

        self._is_fitted = True

        return self

    def _fit(
        self,
        y_train: SUPPORTED_TARGET_TYPES,
        y_test: typing.Optional[SUPPORTED_TARGET_TYPES] = None,
    ) -> BaseEstimator:
        """
        Arguments:
            y_train (SUPPORTED_TARGET_TYPES)
                The labels of the current task. They are going to be encoded in case
                of classification
            y_test (typing.Optional[SUPPORTED_TARGET_TYPES])
                A holdout set of labels
        """
        raise NotImplementedError()

    def transform(
        self,
        y: typing.Union[SUPPORTED_TARGET_TYPES],
    ) -> np.ndarray:
        """
        Arguments:
            y (SUPPORTED_TARGET_TYPES)
                A set of targets that are going to be encoded if the current task
                is classification
        Returns:
            np.ndarray:
                The transformed array
        """
        raise NotImplementedError()

    def inverse_transform(
        self,
        y: SUPPORTED_TARGET_TYPES,
    ) -> np.ndarray:
        """
        Revert any encoding transformation done on a target array

        Arguments:
            y (typing.Union[np.ndarray, pd.DataFrame, pd.Series]):
                Target array to be transformed back to original form before encoding
        Returns:
            np.ndarray:
                The transformed array
        """
        raise NotImplementedError()

    @property
    def classes_(self) -> np.ndarray:
        """
        Complies with scikit learn classes_ attribute,
        which consist of a ndarray of shape (n_classes,)
        where n_classes are the number of classes seen while fitting
        an encoder to the targets.
        Returns:
            classes_: np.ndarray
                The unique classes seen during encoding of a classifier
        """
        if self.encoder is None:
            return np.array([])
        else:
            return self.encoder.categories_[0]

    def is_single_column_target(self) -> bool:
        """
        Output is encoded with a single column encoding
        """
        return self.out_dimensionality == 1
