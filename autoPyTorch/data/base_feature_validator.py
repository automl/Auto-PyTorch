import logging
from typing import List, Optional, Union

import numpy as np

import pandas as pd

from scipy.sparse import spmatrix

from sklearn.base import BaseEstimator

from autoPyTorch.utils.logging_ import PicklableClientLogger


SupportedFeatTypes = Union[List, pd.DataFrame, np.ndarray, spmatrix]


class BaseFeatureValidator(BaseEstimator):
    """
    A class to pre-process features. In this regards, the format of the data is checked,
    and if applicable, features are encoded.
    Attributes:
        feat_type (List[str]):
            List of the column types found by this estimator during fit.
        data_type (str):
            Class name of the data type provided during fit.
        column_transformer (Optional[BaseEstimator])
            Host a encoder object if the data requires transformation (for example,
            if provided a categorical column in a pandas DataFrame)
        transformed_columns (List[str])
            List of columns that were encoded.
    """
    def __init__(
        self,
        logger: Optional[Union[PicklableClientLogger, logging.Logger]] = None,
    ):
        # Register types to detect unsupported data format changes
        self.feat_types: Optional[List[str]] = None
        self.data_type: Optional[type] = None
        self.dtypes: List[str] = []
        self.column_order: List[str] = []

        self.column_transformer: Optional[BaseEstimator] = None
        self.transformed_columns: List[str] = []

        self.logger: Union[
            PicklableClientLogger, logging.Logger
        ] = logger if logger is not None else logging.getLogger(__name__)

        # Required for dataset properties
        self.num_features: Optional[int] = None
        self.categories: List[List[int]] = []
        self.categorical_columns: List[int] = []
        self.numerical_columns: List[int] = []

        self._is_fitted = False

    def fit(
        self,
        X_train: SupportedFeatTypes,
        X_test: Optional[SupportedFeatTypes] = None,
    ) -> BaseEstimator:
        """
        Validates and fit a categorical encoder (if needed) to the features.
        The supported data types are List, numpy arrays and pandas DataFrames.
        CSR sparse data types are also supported

        Args:
            X_train (SupportedFeatTypes):
                A set of features that are going to be validated (type and dimensionality
                checks) and a encoder fitted in the case the data needs encoding
            X_test (Optional[SupportedFeatTypes]):
                A hold out set of data used for checking
        """

        # If a list was provided, it will be converted to pandas
        if isinstance(X_train, list):
            X_train, X_test = self.list_to_dataframe(X_train, X_test)

        self._check_data(X_train)

        if X_test is not None:
            self._check_data(X_test)

            if np.shape(X_train)[1] != np.shape(X_test)[1]:
                raise ValueError("The feature dimensionality of the train and test "
                                 "data does not match train({}) != test({})".format(
                                     np.shape(X_train)[1],
                                     np.shape(X_test)[1]
                                 ))

        # Fit on the training data
        self._fit(X_train)

        self._is_fitted = True

        return self

    def _fit(
        self,
        X: SupportedFeatTypes,
    ) -> BaseEstimator:
        """
        Args:
            X (SupportedFeatTypes):
                A set of features that are going to be validated (type and dimensionality
                checks) and a encoder fitted in the case the data needs encoding
        Returns:
            self:
                The fitted base estimator
        """
        raise NotImplementedError()

    def transform(
        self,
        X: SupportedFeatTypes,
    ) -> np.ndarray:
        """
        Args:
            X_train (SupportedFeatTypes):
                A set of features, whose categorical features are going to be
                transformed

        Return:
            np.ndarray:
                The transformed array
        """
        raise NotImplementedError()
