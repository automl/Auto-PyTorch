# -*- encoding: utf-8 -*-
from typing import Optional, Tuple, List, Union, Dict
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.exceptions import NotFittedError

from scipy import sparse

from autoPyTorch.data.utils import DatasetCompressionSpec
from autoPyTorch.data.tabular_validator import TabularInputValidator
from autoPyTorch.data.time_series_feature_validator import TimeSeriesFeatureValidator
from autoPyTorch.data.time_series_target_validator import TimeSeriesTargetValidator
from autoPyTorch.data.base_feature_validator import SupportedFeatTypes


class TimeSeriesForecastingInputValidator(TabularInputValidator):
    """
    A validator designed for a time series forecasting dataset.
    As a time series forecasting dataset might contain several time sequence with different length, we will transform
    all the data to DataFrameGroupBy whereas each group represents a series
    TODO for multiple output: target names and shapes
    """

    def __init__(self,
                 is_classification: bool = False,
                 logger_port: Optional[int] = None,
                 dataset_compression: Optional[DatasetCompressionSpec] = None,
                 ) -> None:
        super(TimeSeriesForecastingInputValidator, self).__init__(is_classification, logger_port, dataset_compression)
        self.feature_validator = TimeSeriesFeatureValidator(logger=self.logger)
        self.target_validator = TimeSeriesTargetValidator(is_classification=self.is_classification,
                                                          logger=self.logger)
        self._is_uni_variant = False
        self.known_future_features = None
        self.start_times = None
        self.feature_shapes: Dict[str, int] = {}
        self.feature_names: List[str] = []
        self.series_idx = None

    def fit(
            self,
            X_train: Optional[Union[List, pd.DataFrame]],
            y_train: Union[List, pd.DataFrame],
            series_idx: Optional[Union[List[Union[str, int]], str, int]] = None,
            X_test: Optional[Union[List, pd.DataFrame]] = None,
            y_test: Optional[Union[List, pd.DataFrame]] = None,
            start_times: Optional[List[pd.DatetimeIndex]] = None,
            known_future_features: Optional[List[Union[int, str]]] = None,
    ) -> BaseEstimator:
        """
        fit the validator with the training data, (optionally) start times and other information
        Args:
            X_train (Optional[Union[List, pd.DataFrame]]): training features, could be None for "pure" forecasting tasks
            y_train (Union[List, pd.DataFrame]), training targets
            series_idx (Optional[Union[List[Union[str, int]], str, int]]): which columns of features are applied to
                identify the series
            X_test (Optional[Union[List, pd.DataFrame]]): test features. For forecasting tasks, test features indicates
                known future features after the forecasting timestep\
            y_test (Optional[Union[List, pd.DataFrame]]): target in the future
            start_times (Optional[List[pd.DatetimeIndex]]): start times on which the first element of each series is
                sampled
            known_future_features (Optional[List[Union[int, str]]]): which features are known even in the future

        """
        self.series_idx = series_idx

        if start_times is None:
            start_times = [pd.Timestamp('1900-01-01')] * len(y_train)
        else:
            assert len(start_times) == len(y_train), 'start_times_train must have the same length as y_train!'

        self.start_times = start_times

        if X_train is None:
            self._is_uni_variant = True
        if isinstance(y_train, List):
            # X_train and y_train are stored as lists
            y_train_stacked = self.join_series(y_train)
            if y_test is not None:
                y_test_stacked, seq_y_test_length = self.join_series(y_test, return_seq_lengths=True)
            else:
                y_test_stacked = None

            if self._is_uni_variant:
                self.feature_validator.num_features = 0
                self.feature_validator.numerical_columns = []
                self.feature_validator.categorical_columns = []

                self.target_validator.fit(y_train_stacked, y_test_stacked)

                self._is_fitted = True
            else:
                self.known_future_features = known_future_features
                # Check that the data is valid
                if len(X_train) != len(y_train):
                    raise ValueError("Inconsistent number of sequences for features and targets,"
                                     " {} for features and {} for targets".format(len(X_train), len(y_train), ))
                X_train_stacked, sequence_lengths = self.join_series(X_train, return_seq_lengths=True)
                X_test_stacked = self.join_series(X_test) if X_test is not None else None
                if X_test_stacked is not None and y_test_stacked is not None:
                    if len(X_test_stacked) != len(y_test_stacked):
                        raise ValueError("Inconsistent number of test datapoints for features and targets,"
                                         " {} for features and {} for targets".format(len(X_test), len(y_test), ))

                self.feature_validator.fit(X_train_stacked, X_test_stacked,
                                           series_idx=series_idx, sequence_lengths=sequence_lengths)
                self.target_validator.fit(y_train_stacked, y_test_stacked)

                if self.feature_validator.only_contain_series_idx:
                    self._is_uni_variant = True

                self._is_fitted = True

                self.feature_names = self.feature_validator.get_reordered_columns()
                self.feature_shapes = {feature_name: 1 for feature_name in self.feature_names}
        else:
            # TODO X_train and y_train are pd.DataFrame
            raise NotImplementedError

        return self

    def transform(
            self,
            X: Optional[Union[List, pd.DataFrame]],
            y: Optional[Union[List, pd.DataFrame]] = None,
            validate_for_future_features: bool = False
    ) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], List[int]]:
        """
        transform the data with the fitted validator
        Args:
            validate_for_future_features: bool
                if the validator is applied to transform future features (for test sets), in this case we only validate
                X
        """
        if not self._is_fitted:
            raise NotFittedError("Cannot call transform on a validator that is not fitted")
        if validate_for_future_features and y is None:
            if X is None:
                return None, None, []
            if isinstance(X, List):
                num_sequences = len(X)
                sequence_lengths = [0] * num_sequences
                for seq_idx in range(num_sequences):
                    sequence_lengths[seq_idx] = len(X[seq_idx])
                sequence_lengths = np.asarray(sequence_lengths)
                x_transformed, _ = self._transform_X(X, sequence_lengths)
                return x_transformed, None, sequence_lengths
            else:
                raise NotImplementedError

        else:
            if y is None:
                raise ValueError('Targets must be given!')

            if isinstance(y, List):
                num_sequences = len(y)
                sequence_lengths = [0] * num_sequences
                if not self._is_uni_variant:
                    if X is None:
                        raise ValueError('Multi Variant dataset requires X as input!')
                    assert len(X) == len(y), "Length of features must equal to length of targets!"
                if self.series_idx is not None and X is None:
                    raise ValueError('X must be given as series_idx!')

                for seq_idx in range(num_sequences):
                    sequence_lengths[seq_idx] = len(y[seq_idx])
                sequence_lengths = np.asarray(sequence_lengths)

                y_stacked = self.join_series(y)

                x_transformed, series_number = self._transform_X(X, sequence_lengths)
                y_transformed: pd.DataFrame = self.target_validator.transform(y_stacked, index=series_number)

                if self._is_uni_variant:
                    return None, y_transformed, sequence_lengths

                return x_transformed, y_transformed, sequence_lengths
            else:
                raise NotImplementedError

    def _transform_X(self,
                     X: Optional[Union[List, pd.DataFrame]],
                     sequence_lengths: np.ndarray) -> Tuple[pd.DataFrame, Union[np.ndarray, pd.Index]]:
        if self.series_idx is None:
            series_number = np.arange(len(sequence_lengths)).repeat(sequence_lengths)
            if not self._is_uni_variant:
                x_stacked = self.join_series(X)
                x_transformed = self.feature_validator.transform(x_stacked,
                                                                 index=series_number)
            else:
                x_transformed = None
        else:
            # In this case X can only contain pd.DataFrame, see ```time_series_feature_validator.py```
            x_stacked = pd.concat(X)

            series_number = pd.MultiIndex.from_frame(pd.DataFrame(x_stacked[self.series_idx]))

            if not self._is_uni_variant:
                x_transformed = self.feature_validator.transform(x_stacked,
                                                                 index=series_number)
            else:
                x_transformed = None

        return x_transformed, series_number

    @staticmethod
    def join_series(X: List[Union[pd.DataFrame, np.ndarray]],
                    return_seq_lengths: bool = False) -> Union[pd.DataFrame,
                                                               Tuple[pd.DataFrame, List[int]]]:
        """
        join the series into one single value
        """
        num_sequences = len(X)
        sequence_lengths = [0] * num_sequences
        for seq_idx in range(num_sequences):
            sequence_lengths[seq_idx] = len(X[seq_idx])
        series_number = np.arange(len(sequence_lengths)).repeat(sequence_lengths)
        if not isinstance(X, List):
            raise ValueError(f'Input must be a list, but it is {type(X)}')
        if isinstance(X[0], pd.DataFrame):
            joint_input = pd.concat(X)
        elif isinstance(X[0], (List, np.ndarray)):
            joint_input = np.concatenate(X)
        else:
            raise NotImplementedError(f'Unsupported input type: List[{type(X[0])}]')
        joint_input = pd.DataFrame(joint_input)
        joint_input.index = series_number

        if return_seq_lengths:
            return joint_input, sequence_lengths
        else:
            return joint_input
