from autoPyTorch.data.time_series_validator import TimeSeriesInputValidator

# -*- encoding: utf-8 -*-
import logging
from typing import Optional, Tuple, List, Union
import numpy as np

from sklearn.base import BaseEstimator
from sklearn.exceptions import NotFittedError

from autoPyTorch.data.base_feature_validator import SUPPORTED_FEAT_TYPES
from autoPyTorch.data.base_target_validator import SUPPORTED_TARGET_TYPES
from autoPyTorch.data.tabular_validator import TabularInputValidator
from autoPyTorch.data.tabular_feature_validator import TabularFeatureValidator


class TimeSeriesForecastingInputValidator(TabularInputValidator):
    def __init__(self,
                 is_classification: bool = False,
                 logger_port: Optional[int] = None,
                 ) -> None:
        super(TimeSeriesForecastingInputValidator, self).__init__(is_classification, logger_port)
        self._is_uni_variant = False
        self.known_future_features = None
        self.n_prediction_steps = 1

    """
    A validator designed for a time series forecasting dataset.
    As a time series forecasting dataset might contain several time sequnces with
    """

    def fit(
            self,
            X_train: Optional[SUPPORTED_FEAT_TYPES],
            y_train: SUPPORTED_TARGET_TYPES,
            X_test: Optional[SUPPORTED_FEAT_TYPES] = None,
            y_test: Optional[SUPPORTED_TARGET_TYPES] = None,
            n_prediction_steps: int = 1,
            known_future_features: Optional[List[Union[int, str]]] = None,
    ) -> BaseEstimator:
        self.n_prediction_steps = n_prediction_steps
        if X_train is None:
            self._is_uni_variant = True
        if self._is_uni_variant:
            self.feature_validator.num_features = 0
            self.feature_validator.numerical_columns = []
            self.feature_validator.categorical_columns = []

            if y_test is not None:
                self.target_validator.fit(y_train[0], y_test[0])
            else:
                self.target_validator.fit(y_train[0])
            self._is_fitted = True
        else:
            self.known_future_features = known_future_features
            # Check that the data is valid
            if len(X_train) != len(y_train):
                raise ValueError("Inconsistent number of sequences for features and targets,"
                                 " {} for features and {} for targets".format(len(X_train), len(y_train), ))

            if X_test is not None:
                if len(X_test) != len(y_test):
                    raise ValueError("Inconsistent number of test datapoints for features and targets,"
                                     " {} for features and {} for targets".format(len(X_test), len(y_test), ))
                # TODO write a feature input validator to check X_test for known_future_features
                super().fit(X_train[0], y_train[0], X_test[0], y_test[0])
            else:
                super().fit(X_train[0], y_train[0])

            self.check_input_shapes(X_train, y_train, is_training=True)

            if X_test is not None:
                self.check_input_shapes(X_test, y_test, is_training=False)
            return self

    @staticmethod
    def get_num_features(X):
        X_shape = np.shape(X)
        return 1 if len(X_shape) == 1 else X_shape[1]

    @staticmethod
    def check_input_shapes(X, y, is_training: bool = True):
        num_features = [0] * len(X)
        out_dimensionality = [0] * len(y)

        for i in range(len(X)):
            num_features[i] = TimeSeriesForecastingInputValidator.get_num_features(X[i])
            out_dimensionality[i] = TimeSeriesForecastingInputValidator.get_num_features(y[i])

        if not np.all(np.asarray(num_features) == num_features[0]):
            raise ValueError(f"All the sequences need to have the same number of features in "
                             f"{'train' if is_training else 'test'} set!")

        if not np.all(np.asarray(out_dimensionality) == out_dimensionality[0]):
            raise ValueError(f"All the sequences need to have the same number of targets in "
                             f"{'train' if is_training else 'test'} set!")

    def transform(
            self,
            X: Optional[SUPPORTED_FEAT_TYPES],
            y: Optional[SUPPORTED_TARGET_TYPES] = None,
    ) -> Tuple[Union[np.ndarray], List[int], Optional[np.ndarray]]:
        if not self._is_fitted:
            raise NotFittedError("Cannot call transform on a validator that is not fitted")

        if y is None:
            raise ValueError('Targets must be given!')

        num_sequences = len(y)
        sequence_lengths = [0] * num_sequences
        if self._is_uni_variant:
            num_features = 0
        else:
            if X is None:
                raise ValueError('Multi Variant dataset requires X as input!')
            num_features = self.feature_validator.num_features

        for seq_idx in range(num_sequences):
            sequence_lengths[seq_idx] = len(y[seq_idx])
        sequence_lengths = np.asarray(sequence_lengths)

        num_targets = self.target_validator.out_dimensionality

        num_data = np.sum(sequence_lengths)

        start_idx = 0

        if self._is_uni_variant:
            y_flat = np.empty([num_data, num_targets])
            for seq_idx, seq_length in enumerate(sequence_lengths):
                end_idx = start_idx + seq_length
                y_flat[start_idx: end_idx] = np.array(y[seq_idx]).reshape([-1, num_targets])
                start_idx = end_idx
            y_transformed = self.target_validator.transform(y_flat)  # type:np.ndarray
            if y_transformed.ndim == 1:
                y_transformed = np.expand_dims(y_transformed, -1)
            return np.asarray([]), y_transformed, sequence_lengths

        # a matrix that is concatenated by all the time series sequences
        X_flat = np.empty([num_data, num_features])
        y_flat = np.empty([num_data, num_targets])

        start_idx = 0
        for seq_idx, seq_length in enumerate(sequence_lengths):
            end_idx = start_idx + seq_length
            X_flat[start_idx: end_idx] = np.array(X[seq_idx]).reshape([-1, num_features])
            y_flat[start_idx: end_idx] = np.array(y[seq_idx]).reshape([-1, num_targets])
            start_idx = end_idx

        X_transformed = self.feature_validator.transform(X_flat)  # type:np.ndarray
        y_transformed = self.target_validator.transform(y_flat)  # type:np.ndarray
        if y_transformed.ndim == 1:
            y_transformed = np.expand_dims(y_transformed, -1)
        return X_transformed, y_transformed, sequence_lengths

