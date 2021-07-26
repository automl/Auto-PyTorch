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
from autoPyTorch.data.base_feature_validator import BaseFeatureValidator, SUPPORTED_FEAT_TYPES


class TimeSeriesForecastingFeatureValidator(TabularFeatureValidator):
    def __init__(self,
                 logger: Optional[Union[PicklableClientLogger, logging.Logger]] = None,
                 ) -> None:
        super(TimeSeriesForecastingFeatureValidator, self).__init__(logger)
        self.feature_validators = None # type: Optional[List[TabularFeatureValidator]]

    def fit(self,
            X_train: Union[np.ndarray, List[SUPPORTED_FEAT_TYPES]],
            X_test: Optional[Union[np.ndarray, List[SUPPORTED_FEAT_TYPES]]] = None) -> BaseEstimator:
        """
        We expect a time series dataset stored in the form :[time_series_sequences]
        TODO can we directly read X_train and X_test from panda DataFrame

        Arguments:
            X_train (np.ndarray):
                A set of data that are going to be validated (type and dimensionality
                checks) and used for fitting, it is composed of multiple time series sequences which might have
                different length

            X_test (Optional[np.ndarray]):
                An optional set of data that is going to be validated

        Returns:
            self:
                The fitted base estimator
        """
        categorical_columns = [[] for _ in range(len(X_train))]
        numerical_columns = [[] for _ in range(len(X_train))]
        categories = [[] for _ in range(len(X_train))]
        num_features = [0] * len(X_train)

        if X_test is not None:
            if len(X_train) != len(X_test):
                raise ValueError(f"Training data needs to have the same number sequences as the test data")

        self.feature_validators = [TabularFeatureValidator(self.logger) for _ in range(len(X_train))]
        if X_test is not None:
            for seq_idx, (X_train_seq, X_test_seq)  in enumerate(zip(X_train, X_test)):
                self.feature_validators[seq_idx].fit(X_train_seq, X_test_seq)

                categorical_columns[seq_idx] = self.feature_validators[seq_idx].categorical_columns
                numerical_columns[seq_idx] = self.feature_validators[seq_idx].numerical_columns
                categories[seq_idx] = self.feature_validators[seq_idx].categories
                num_features[seq_idx] = self.feature_validators[seq_idx].num_features
        else:
            for seq_idx, X_train_seq in enumerate(X_train):
                self.feature_validators[seq_idx].fit(X_train_seq)

                categorical_columns[seq_idx] = self.feature_validators[seq_idx].categorical_columns
                numerical_columns[seq_idx] = self.feature_validators[seq_idx].numerical_columns
                categories[seq_idx] = self.feature_validators[seq_idx].categories
                num_features[seq_idx] = self.feature_validators[seq_idx].num_features

        if not np.all(np.asarray(categorical_columns) == categorical_columns[0]):
            raise ValueError(f"All the sequence needs to have the same categorical columns!")
        if not np.all(np.asarray(categories) == categories[0]):
            raise ValueError(f"All the sequence needs to have the same categories!")
        if not np.all(np.asarray(numerical_columns) == numerical_columns[0]):
            raise ValueError(f"All the sequence needs to have the same Numerical columns!")
        if not np.all(np.asarray(num_features) == num_features[0]):
            raise ValueError(f"All the sequence needs to have the same number of features!")

        self.categories = categories[0]
        self.num_features = num_features[0]
        self.categorical_columns = categorical_columns[0]
        self.numerical_columns = numerical_columns[0]

        self.feat_type = self.feature_validators[0].feat_type
        self.data_type = self.feature_validators[0].data_type
        self.dtypes = self.feature_validators[0].dtypes
        self.column_order = self.feature_validators[0].column_order
        self._is_fitted = True

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
        if not self._is_fitted:
            raise NotFittedError("Cannot call transform on a validator that is not fitted")

        for seq_idx in range(len(X)):
            X[seq_idx] = self.feature_validators[seq_idx].transform(X[seq_idx])
        return X
