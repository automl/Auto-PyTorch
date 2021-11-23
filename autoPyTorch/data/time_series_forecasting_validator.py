from autoPyTorch.data.time_series_validator import TimeSeriesInputValidator

# -*- encoding: utf-8 -*-
import logging
import typing
import numpy as np

from sklearn.base import BaseEstimator
from sklearn.exceptions import NotFittedError

from autoPyTorch.data.base_feature_validator import SUPPORTED_FEAT_TYPES
from autoPyTorch.data.base_target_validator import SUPPORTED_TARGET_TYPES
from autoPyTorch.data.tabular_validator import TabularInputValidator


class TimeSeriesForecastingInputValidator(TabularInputValidator):
    """
    A validator designed for a time series forecasting dataset.
    As a time series forecasting dataset might contain several time sequnces with
    """

    def fit(
            self,
            X_train: SUPPORTED_FEAT_TYPES,
            y_train: SUPPORTED_TARGET_TYPES,
            X_test: typing.Optional[SUPPORTED_FEAT_TYPES] = None,
            y_test: typing.Optional[SUPPORTED_TARGET_TYPES] = None,
    ) -> BaseEstimator:
        # Check that the data is valid
        if len(X_train) != len(y_train):
            raise ValueError("Inconsistent number of sequences for features and targets,"
                             " {} for features and {} for targets".format(len(X_train), len(y_train), ))

        if X_test is not None:
            if len(X_test) != len(y_test):
                raise ValueError("Inconsistent number of test datapoints for features and targets,"
                                 " {} for features and {} for targets".format(len(X_test), len(y_test), ))
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
            X: SUPPORTED_FEAT_TYPES,
            y: typing.Optional[SUPPORTED_TARGET_TYPES] = None,
            shift_input_data: bool = True,
            n_prediction_steps: int = 1
    ) -> typing.Tuple[np.ndarray, typing.List[int], typing.Optional[np.ndarray]]:
        if not self._is_fitted:
            raise NotFittedError("Cannot call transform on a validator that is not fitted")

        num_sequences = len(X)
        sequence_lengths = [0] * num_sequences
        num_features = self.feature_validator.num_features

        if shift_input_data:
            for seq_idx in range(num_sequences):
                X[seq_idx] = X[seq_idx][:-n_prediction_steps]
                # y[seq_idx] = y[seq_idx][n_prediction_steps:]
                sequence_lengths[seq_idx] = len(X[seq_idx])
        else:
            for seq_idx in range(num_sequences):
                sequence_lengths[seq_idx] = len(X[seq_idx])

        if y is not None:
            num_targets = self.target_validator.out_dimensionality

            num_train_data = np.sum(sequence_lengths)

            # a matrix that is concatenated by all the time series sequences
            X_flat = np.empty([num_train_data, num_features])
            y_flat = np.empty([num_train_data + n_prediction_steps * num_sequences, num_targets])

            start_idx = 0
            for seq_idx, seq_length in enumerate(sequence_lengths):
                end_idx = start_idx + seq_length
                X_flat[start_idx: end_idx] = np.array(X[seq_idx]).reshape([-1, num_features])
                if shift_input_data:
                    y_flat[
                    start_idx + n_prediction_steps * seq_idx: end_idx + n_prediction_steps * (seq_idx + 1)] = np.array(
                        y[seq_idx]).reshape([-1, num_targets])
                else:
                    y_flat[start_idx: end_idx] = np.array(y[seq_idx]).reshape([-1, num_targets])
                start_idx = end_idx

            X_transformed = self.feature_validator.transform(X_flat)
            y_transformed = self.target_validator.transform(y_flat)
            return X_transformed, sequence_lengths, y_transformed

            num_train_data = np.sum(sequence_lengths)

            # a matrix that is concatenated by all the time series sequences
            X_flat = np.empty([num_train_data, num_features])

            start_idx = 0
            # TODO make it parallel with large number of sequences
            for seq_idx, seq_length in enumerate(sequence_lengths):
                end_idx = start_idx + seq_length
                X_flat[start_idx: end_idx] = np.array(X[seq_idx]).reshape([-1, num_features])
                start_idx = end_idx

            X_transformed = self.feature_validator.transform(X_flat)

            return X_transformed, sequence_lengths
