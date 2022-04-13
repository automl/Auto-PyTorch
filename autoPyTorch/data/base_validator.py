# -*- encoding: utf-8 -*-
import logging.handlers
from typing import Optional, Tuple

import numpy as np

from sklearn.base import BaseEstimator
from sklearn.exceptions import NotFittedError

from autoPyTorch.data.base_feature_validator import SupportedFeatTypes
from autoPyTorch.data.base_target_validator import SupportedTargetTypes


class BaseInputValidator(BaseEstimator):
    """
    Makes sure the input data complies with Auto-PyTorch requirements.
    Categorical inputs are encoded via an Encoder, if the input
    is a dataframe. This allow us to nicely predict string targets

    This class also perform checks for data integrity and flags the user
    via informative errors.

    Attributes:
        is_classification (bool):
            For classification task, this flag indicates that the target data
            should be encoded
        feature_validator (FeatureValidator):
            A FeatureValidator instance used to validate and encode feature columns to match
            sklearn expectations on the data
        target_validator (TargetValidator):
            A TargetValidator instance used to validate and encode (in case of classification)
            the target values
    """
    def __init__(
        self,
        is_classification: bool = False,
        logger_port: Optional[int] = logging.handlers.DEFAULT_TCP_LOGGING_PORT,
    ) -> None:
        raise NotImplementedError()

    def fit(
        self,
        X_train: SupportedFeatTypes,
        y_train: SupportedTargetTypes,
        X_test: Optional[SupportedFeatTypes] = None,
        y_test: Optional[SupportedTargetTypes] = None,
    ) -> BaseEstimator:
        """
        Validates and fit a categorical encoder (if needed) to the features, and
        a encoder for targets in the case of classification. Specifically:

        For features:
            + Valid data types are enforced (List, np.ndarray, pd.DataFrame, pd.Series, scipy
              sparse) as well as dimensionality checks
            + If the provided data is a pandas DataFrame with categorical/boolean/int columns,
              such columns will be encoded using an Ordinal Encoder
        For targets:
            + Checks for dimensionality as well as missing values are performed.
            + If performing a classification task, the data is going to be encoded

        Args:
            X_train (SupportedFeatTypes):
                A set of features that are going to be validated (type and dimensionality
                checks). If this data contains categorical columns, an encoder is going to
                be instantiated and trained with this data.
            y_train (SupportedTargetTypes):
                A set of targets that are going to be encoded if the task is for classification
            X_test (Optional[SupportedFeatTypes]):
                A hold out set of features used for checking
            y_test (SupportedTargetTypes):
                A hold out set of targets used for checking. Additionally, if the current task
                is a classification task, this y_test categories are also going to be used to
                fit a pre-processing encoding (to prevent errors on unseen classes).
        Returns:
            self
        """
        # Check that the data is valid
        if np.shape(X_train)[0] != np.shape(y_train)[0]:
            raise ValueError("Inconsistent number of train datapoints for features and targets,"
                             " {} for features and {} for targets".format(
                                 np.shape(X_train)[0],
                                 np.shape(y_train)[0],
                             ))
        if X_test is not None and np.shape(X_test)[0] != np.shape(y_test)[0]:
            raise ValueError("Inconsistent number of test datapoints for features and targets,"
                             " {} for features and {} for targets".format(
                                 np.shape(X_test)[0],
                                 np.shape(y_test)[0],
                             ))

        self.feature_validator.fit(X_train, X_test)
        self.target_validator.fit(y_train, y_test)
        self._is_fitted = True

        return self

    def transform(
        self,
        X: SupportedFeatTypes,
        y: Optional[SupportedTargetTypes] = None,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Transform the given target or features to a numpy array

        Args:
            X (SupportedFeatTypes):
                A set of features to transform
            y (Optional[SupportedTargetTypes]):
                A set of targets to transform

        Returns:
            np.ndarray:
                The transformed features array
            np.ndarray:
                The transformed targets array
        """
        if not self._is_fitted:
            raise NotFittedError("Cannot call transform on a validator that is not fitted")
        X_transformed = self.feature_validator.transform(X)
        if y is not None:
            return X_transformed, self.target_validator.transform(y)
        else:
            return X_transformed, y
