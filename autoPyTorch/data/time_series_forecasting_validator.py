from autoPyTorch.data.time_series_validator import TimeSeriesInputValidator

# -*- encoding: utf-8 -*-
import logging
import warnings
from typing import Optional, Tuple, List, Union, Dict
import numpy as np
import pandas as pd
from pandas.core.groupby.generic import DataFrameGroupBy
from sklearn.base import BaseEstimator
from sklearn.exceptions import NotFittedError

from autoPyTorch.data.utils import  DatasetCompressionSpec
from autoPyTorch.data.tabular_validator import TabularInputValidator
from autoPyTorch.data.tabular_feature_validator import TabularFeatureValidator
from autoPyTorch.data.time_series_feature_validator import TimeSeriesFeatureValidator


class TimeSeriesForecastingInputValidator(TabularInputValidator):
    """
    A validator designed for a time series forecasting dataset.
    As a time series forecasting dataset might contain several time sequence with different length, we will transform
    all the data to DataFrameGroupBy whereas each group represents a series
    """
    def __init__(self,
                 is_classification: bool = False,
                 logger_port: Optional[int] = None,
                 dataset_compression: Optional[DatasetCompressionSpec] = None,
                 ) -> None:
        super(TimeSeriesForecastingInputValidator, self).__init__(is_classification, logger_port, dataset_compression)
        self.feature_validator = TimeSeriesFeatureValidator(logger=self.logger)
        self._is_uni_variant = False
        self.known_future_features = None
        self.n_prediction_steps = 1
        self.start_times_train = None
        self.start_times_test = None
        self.feature_shapes = {}
        self.feature_names = []
        self.series_idx = None

    def fit(
            self,
            X_train: Optional[Union[List, pd.DataFrame]],
            y_train: Union[List, pd.DataFrame],
            series_idx: Optional[Union[List[Union[str, int]], str, int]] = None,
            X_test: Optional[Union[List, pd.DataFrame]] = None,
            y_test: Optional[Union[List, pd.DataFrame]] = None,
            start_times_train: Optional[List[pd.DatetimeIndex]] = None,
            start_times_test: Optional[List[pd.DatetimeIndex]] = None,
            freq: str = '1Y',
            n_prediction_steps: int = 1,
            known_future_features: Optional[List[Union[int, str]]] = None,
            use_time_features: bool = False
    ) -> BaseEstimator:
        """
        fit the validator with the training data, (optionally) start times and other information
        Args:
            X_train (Optional[Union[List, pd.DataFrame]]): training features, could be None for "pure" forecasting tasks
            y_train (Union[List, pd.DataFrame]), training targets
            series_idx (Optional[Union[List[Union[str, int]], str, int]]): which columns of the data are considered as

        """
        if isinstance(series_idx, (str, int)):
            series_idx = [series_idx]
        self.series_idx = series_idx
        self.n_prediction_steps = n_prediction_steps

        if y_test is not None and bool(start_times_test) != bool(start_times_train):
            warnings.warn('One of start_times_test or start_times_train is missing! This might result in the '
                          'risk of not proper evaluated ')

        if start_times_train is None:
            start_times_train = [pd.DatetimeIndex(pd.to_datetime(['2000-01-01']), freq=freq)] * len(y_train)
        else:
            assert len(start_times_train) == len(y_train), 'start_times_train must have the same length as y_train!'

        if y_test is not None:
            if start_times_test is None:
                start_times_test = [pd.DatetimeIndex(pd.to_datetime(['1900-01-01']), freq=freq)] * len(y_test)
            else:
                assert len(start_times_train) == len(y_train), 'start_times_train must have the same length as y_train!'

        self.start_times_train = start_times_train
        self.start_times_test = start_times_test

        if X_train is None:
            self._is_uni_variant = True
        if isinstance(y_train, List):
            # X_train and y_train are stored as lists
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
                self.feature_validator.fit(X_train[0], None if X_test is None else X_test[0], series_idx=series_idx)
                self.target_validator.fit(y_train[0], None if y_test is None else y_test[0])

                if self.feature_validator.only_contain_series_idx:
                    self._is_uni_variant = True

                self._is_fitted = True

                # In this case we don't assign series index to the data, we manually assigne

                self.check_input_shapes(X_train, y_train, is_training=True)

                if X_test is not None:
                    self.check_input_shapes(X_test, y_test, is_training=False)
                if hasattr(X_train[0], 'columns'):
                    features = X_train[0].columns.values.tolist()
                else:
                    features = list(map(str, range(len(X_train[0]))))
                for feature in features:
                    self.feature_names.append(feature)
                    self.feature_shapes[feature] = 1
        else:
            # TODO X_train and y_train are pd.DataFrame
            raise NotImplementedError

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
            X: Optional[Union[List, pd.DataFrame]],
            y: Optional[Union[List, pd.DataFrame]] = None,
    ) -> Tuple[Optional[DataFrameGroupBy], DataFrameGroupBy, np.ndarray]:
        if not self._is_fitted:
            raise NotFittedError("Cannot call transform on a validator that is not fitted")

        if y is None:
            raise ValueError('Targets must be given!')

        if isinstance(y, List):
            num_sequences = len(y)
            sequence_lengths = [0] * num_sequences
            if self._is_uni_variant:
                num_features = 0
            else:
                if X is None:
                    raise ValueError('Multi Variant dataset requires X as input!')
                num_features = self.feature_validator.num_features
            assert len(X) == len(y), "Length of features must equal to length of targets!"

            for seq_idx in range(num_sequences):
                sequence_lengths[seq_idx] = len(y[seq_idx])
            sequence_lengths = np.asarray(sequence_lengths)

            num_targets = self.target_validator.out_dimensionality

            num_data = np.sum(sequence_lengths)

            start_idx = 0

            y_flat = np.empty([num_data, num_targets])

            for seq_idx, seq_length in enumerate(sequence_lengths):
                end_idx = start_idx + seq_length
                y_flat[start_idx: end_idx] = np.array(y[seq_idx]).reshape([-1, num_targets])
                start_idx = end_idx

            y_transformed = self.target_validator.transform(y_flat)
            if y_transformed.ndim == 1:
                y_transformed = np.expand_dims(y_transformed, -1)

            if self.series_idx is None:
                series_number = np.arange(len(sequence_lengths)).repeat(sequence_lengths)
                if not self._is_uni_variant:
                    if isinstance(X[0], np.ndarray):
                        x_flat: pd.DataFrame = pd.DataFrame(np.vstack(X))
                    elif isinstance(X[0], pd.DataFrame):
                        x_flat: pd.DataFrame = pd.concat(X)
                    else:
                        raise NotImplementedError(f'Cannot transform a List of {type(X[0])}')
                    x_transformed = self.feature_validator.transform(x_flat)

            else:
                # In this case X can only contain pd.DataFrame, see ```time_series_feature_validator.py```
                x_flat = pd.concat(X)
                x_columns = x_flat.columns
                for ser_id in self.series_idx:
                    if ser_id not in x_columns:
                        raise ValueError(f'{ser_id} does not exist in input feature X')

                series_number = pd.MultiIndex.from_frame(x_flat[self.series_idx])
                if not self._is_uni_variant:
                    x_transformed = self.feature_validator.transform(x_flat.drop[self.series_idx])
            y_transformed: pd.DataFrame = pd.DataFrame(y_transformed,
                                                       index=pd.Index(series_number))
            y_transformed: DataFrameGroupBy = y_transformed.groupby(y_transformed.index)
            if self._is_uni_variant:
                return None, y_transformed, sequence_lengths
            return x_transformed.groupby(x_transformed.index), y_transformed, sequence_lengths
        else:
            raise NotImplementedError
