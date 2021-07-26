from autoPyTorch.data.tabular_target_validator import TabularTargetValidator

import typing
import logging
import numpy as np

import pandas as pd

import scipy.sparse

import sklearn
from sklearn.base import BaseEstimator
from sklearn.exceptions import NotFittedError
from sklearn.utils.multiclass import type_of_target

from autoPyTorch.data.base_target_validator import BaseTargetValidator, SUPPORTED_TARGET_TYPES
from autoPyTorch.utils.logging_ import PicklableClientLogger


class TimeSeriesForecastingTargetValidator(TabularTargetValidator):
    def __init__(self,
                 is_classification: bool = False,
                 logger: typing.Optional[typing.Union[PicklableClientLogger, logging.Logger
                 ]] = None,
                 ) -> None:
        TabularTargetValidator.__init__(self, is_classification, logger)
        self.target_validators = None

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
        if y_test is not None:
            if len(y_train) != len(y_test):
                raise ValueError(f"Training target needs to have the same number sequences as the test target")

        self.target_validators = [TabularTargetValidator(self.is_classification, self.logger) for _ in
                                  range(len(y_train))]

        out_dimensionality = [[] for _ in range(len(y_train))]
        type_of_target = [""] * len(y_train)

        if y_test is not None:
            for seq_idx, (y_train_seq, y_test_seq)  in enumerate(zip(y_train, y_test)):
                self.target_validators[seq_idx].fit(y_train_seq, y_test_seq)

                out_dimensionality[seq_idx] = self.target_validators[seq_idx].out_dimensionality
                type_of_target[seq_idx] = self.target_validators[seq_idx].type_of_target

        else:
            for seq_idx, y_train_seq in enumerate(y_train):
                self.target_validators[seq_idx].fit(y_train_seq)

                out_dimensionality[seq_idx] = self.target_validators[seq_idx].out_dimensionality
                type_of_target[seq_idx] = self.target_validators[seq_idx].type_of_target

        if not np.all(np.asarray(out_dimensionality) == out_dimensionality[0]):
            raise ValueError(f"All the sequence needs to have the same out_dimensionality!")
        # TODO consider how to handle "continuous" and "multiple_classes" data type
        """
        if not np.all(np.asarray(type_of_target) == type_of_target[0]):
            raise ValueError(f"All the sequence needs to have the same type_of_target!")
        """

        self.out_dimensionality = out_dimensionality[0]
        self.type_of_target = type_of_target[0]

        self.data_type = self.target_validators[0].data_type
        self.dtype = self.target_validators[0].dtype

        self._is_fitted = True

        return self

    def transform(
            self,
            y: typing.Union[SUPPORTED_TARGET_TYPES],
    ) -> np.ndarray:
        if not self._is_fitted:
            raise NotFittedError("Cannot call transform on a validator that is not fitted")
        for seq_idx in range(len(y)):
            y[seq_idx] = self.target_validators[seq_idx].transform(y[seq_idx])
        return y

    """
    Validator for time series forecasting, currently only consider regression tasks
    TODO: Considering Classification Validator
    """