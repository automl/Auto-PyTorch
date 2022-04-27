import os
from typing import Any, Dict, List, Optional, Tuple, Union, cast, Set
import uuid
import bisect
import copy
import warnings

import numpy as np

import pandas as pd
from pandas._libs.tslibs import to_offset
from pandas._libs.tslibs.np_datetime import OutOfBoundsDatetime
from scipy.sparse import issparse

import torch
from torch.utils.data.dataset import Dataset, Subset, ConcatDataset

import torchvision.transforms

from autoPyTorch.constants import (
    CLASSIFICATION_OUTPUTS,
    STRING_TO_OUTPUT_TYPES,
    TASK_TYPES_TO_STRING,
    TIMESERIES_FORECASTING,
)
from autoPyTorch.datasets.base_dataset import BaseDataset, BaseDatasetInputType, type_of_target
from autoPyTorch.datasets.resampling_strategy import (
    CrossValFuncs,
    CrossValTypes,
    DEFAULT_RESAMPLING_PARAMETERS,
    HoldOutFuncs,
    HoldoutValTypes
)

from gluonts.time_feature.lag import get_lags_for_frequency
from gluonts.time_feature import (
    Constant as ConstantTransform,
    TimeFeature,
    time_features_from_frequency_str,
)
from autoPyTorch.utils.forecasting_time_features import FREQUENCY_MAP

from autoPyTorch.data.time_series_forecasting_validator import TimeSeriesForecastingInputValidator
from autoPyTorch.pipeline.components.preprocessing.time_series_preprocessing.TimeSeriesTransformer import \
    TimeSeriesTransformer
from autoPyTorch.utils.common import FitRequirement
from autoPyTorch.constants_forecasting import SEASONALITY_MAP, MAX_WINDOW_SIZE_BASE
from autoPyTorch.pipeline.components.training.metrics.metrics import compute_mase_coefficient

TIME_SERIES_FORECASTING_INPUT = Tuple[np.ndarray, np.ndarray]  # currently only numpy arrays are supported
TIME_SERIES_REGRESSION_INPUT = Tuple[np.ndarray, np.ndarray]
TIME_SERIES_CLASSIFICATION_INPUT = Tuple[np.ndarray, np.ndarray]


class TimeSeriesSequence(Dataset):
    def __init__(self,
                 X: Optional[Union[np.ndarray, pd.DataFrame]],
                 Y: Union[np.ndarray, pd.Series],
                 start_time_train: Optional[pd.DatetimeIndex] = None,
                 freq: str = '1Y',
                 time_feature_transform: List[TimeFeature] = [],
                 X_test: Optional[Union[np.ndarray, pd.DataFrame]] = None,
                 Y_test: Optional[Union[np.ndarray, pd.DataFrame]] = None,
                 start_time_test: Optional[pd.DatetimeIndex] = None,
                 train_transforms: Optional[torchvision.transforms.Compose] = None,
                 val_transforms: Optional[torchvision.transforms.Compose] = None,
                 static_features: Tuple[Union[int, str]] = None,
                 n_prediction_steps: int = 0,
                 sp: int = 1,
                 known_future_features: Optional[Tuple[str]] = None,
                 only_has_past_targets: bool = False,
                 compute_mase_coefficient_value: bool = True,
                 time_features=None,
                 time_features_test=None,
                 is_test_set=False,
                 ):
        """
        A dataset representing a time series sequence.
        Args:
            seed:
            train_transforms:
            val_transforms:
            n_prediction_steps: int, how many steps need to be predicted in advance
        """
        self.n_prediction_steps = n_prediction_steps

        self.X = X
        self.Y = Y

        self.observed_target = ~np.isnan(self.Y)
        if start_time_train is None:
            start_time_train = pd.DatetimeIndex(pd.to_datetime(['1900-01-01']), freq=freq)
        self.start_time_train = start_time_train

        self.X_val = None
        self.Y_val = None

        self.X_test = X_test
        self.Y_tet = Y_test
        self.start_time_test = start_time_test

        self.time_feature_transform = time_feature_transform
        self.static_features = static_features

        self.freq = freq

        # We also need to be able to transform the data, be it for pre-processing
        # or for augmentation
        self.train_transform = train_transforms
        self.val_transform = val_transforms
        self.sp = sp
        if compute_mase_coefficient_value:
            if only_has_past_targets:
                self.mase_coefficient = compute_mase_coefficient(self.Y, sp=self.sp,
                                                                 n_prediction_steps=n_prediction_steps)
            else:
                self.mase_coefficient = compute_mase_coefficient(self.Y[:-n_prediction_steps], sp=self.sp,
                                                                 n_prediction_steps=n_prediction_steps)

        else:
            self.mase_coefficient = 1.0
        self.only_has_past_targets = only_has_past_targets
        self.known_future_features = known_future_features

        self.transform_time_features = False
        self._cached_time_features: Optional[np.ndarray] = time_features
        self.is_test_set = is_test_set

    def __getitem__(self, index: int, train: bool = True) \
            -> Tuple[Dict[str, torch.Tensor], Optional[Dict[str, torch.Tensor]]]:
        """
        get a subsequent of time series data, unlike vanilla tabular dataset, we obtain all the previous sequences
        until the given index, this allows us to do further transformation.
        (When fed to the neural network, the data is arranged as follows:
        [past_targets, time_features, X_features])

        Args:
            index (int): what element to yield from all the train/test tensors
            train (bool): Whether to apply a train or test transformation, if any

        Returns:
            features from past, targets from past and future
        """
        if index < 0:
            index = self.__len__() + index

        if self.X is not None:
            if hasattr(self.X, 'loc'):
                past_features = self.X.iloc[:index + 1]
            else:
                past_features = self.X[:index + 1]

            if self.known_future_features:
                future_features = self.X.iloc[
                                  index + 1: index + self.n_prediction_steps + 1, self.known_future_features
                                  ]
            else:
                future_features = None
        else:
            past_features = None
            future_features = None

        if self.transform_time_features:
            if self.time_feature_transform:
                self.compute_time_features()

                if past_features:
                    past_features = np.hstack(past_features, [self._cached_time_features[:index + 1]])
                else:
                    past_features = self._cached_time_features[:index + 1]
                if future_features:
                    future_features = np.hstack([
                        past_features,
                        self._cached_time_features[index + 1:index + self.n_prediction_steps + 1]
                    ])
                else:
                    future_features = self._cached_time_features[index + 1:index + self.n_prediction_steps + 1]

        if future_features is not None and future_features.shape[0] == 0:
            future_features = None

        if self.train_transform is not None and train and past_features is not None:
            past_features = self.train_transform(past_features)
            if future_features is not None:
                future_features = self.train_transform(future_features)
        elif self.val_transform is not None and not train and past_features is not None:
            past_features = self.val_transform(past_features)
            if future_features is not None:
                future_features = self.val_transform(future_features)

        # In case of prediction, the targets are not provided
        targets = self.Y
        if self.only_has_past_targets:
            future_targets = None
        else:
            future_targets = targets[index + 1: index + self.n_prediction_steps + 1]
            future_targets = torch.from_numpy(future_targets)
            future_targets = {
                'future_targets': future_targets,
                'future_observed_targets': torch.from_numpy(
                    self.observed_target[index + 1: index + self.n_prediction_steps + 1]
                )
            }

        if isinstance(past_features, np.ndarray):
            past_features = torch.from_numpy(past_features)

        if isinstance(future_features, np.ndarray):
            future_features = torch.from_numpy(future_features)

        past_target = targets[:index + 1]
        past_target = torch.from_numpy(past_target)

        return {"past_targets": past_target,
                "past_features": past_features,
                "future_features": future_features,
                "mase_coefficient": self.mase_coefficient,
                'past_observed_targets': torch.from_numpy(self.observed_target[:index + 1]),
                'decoder_lengths': 0 if future_targets is None else future_targets['future_targets'].shape[
                    0]}, future_targets

    def __len__(self) -> int:
        return self.Y.shape[0] if self.only_has_past_targets else self.Y.shape[0] - self.n_prediction_steps

    def compute_time_features(self, ):
        if self._cached_time_features is None:
            periods = self.Y.shape[0]
            if self.is_test_set:
                periods += self.n_prediction_steps

            date_info = pd.date_range(start=self.start_time_train,
                                      periods=periods,
                                      freq=self.freq)

            self._cached_time_features = np.vstack(
                [transform(date_info).to_numpy(float) for transform in self.time_feature_transform]
            ).T
        else:
            if self.is_test_set:
                if self._cached_time_features.shape[0] == self.Y.shape[0]:
                    try:
                        date_info = pd.date_range(start=self.start_time_train,
                                                  periods=self.n_prediction_steps + self.Y.shape[0],
                                                  freq=self.freq)
                        time_feature_future = np.vstack(
                            [transform(date_info).to_numpy(float)
                             if not isinstance(transform, ConstantTransform) else transform(date_info)
                             for transform in self.time_feature_transform]
                        ).T
                    except OutOfBoundsDatetime:
                        # This is only a temporal solution TODO consider how to solve this!
                        time_feature_future = np.zeros([self.n_prediction_steps, len(self.time_feature_transform)])

                    self._cached_time_features = np.concatenate([self._cached_time_features, time_feature_future])

    def update_transform(self, transform: Optional[torchvision.transforms.Compose],
                         train: bool = True,
                         ) -> 'BaseDataset':
        """
        During the pipeline execution, the pipeline object might propose transformations
        as a product of the current pipeline configuration being tested.

        This utility allows to return a self with the updated transformation, so that
        a dataloader can yield this dataset with the desired transformations

        Args:
            transform (torchvision.transforms.Compose): The transformations proposed
                by the current pipeline
            train (bool): Whether to update the train or validation transform

        Returns:
            self: A copy of the update pipeline
        """
        if train:
            self.train_transform = transform
        else:
            self.val_transform = transform
        return self

    def get_val_seq_set(self, index: int) -> "TimeSeriesSequence":
        if self.only_has_past_targets:
            raise ValueError("get_val_seq_set is not supported for the sequence that only has past targets!")
        if index < 0:
            index = self.__len__() + index
        if index == self.__len__() - 1:
            return copy.copy(self)
        else:
            if self.X is not None:
                X = self.X[:index + 1 + self.n_prediction_steps]
            else:
                X = None
            if self._cached_time_features is None:
                cached_time_features = None
            else:
                cached_time_features = self._cached_time_features[:index + 1 + self.n_prediction_steps]

            return TimeSeriesSequence(X=X,
                                      Y=self.Y[:index + 1],
                                      start_time_train=self.start_time_train,
                                      freq=self.freq,
                                      time_feature_transform=self.time_feature_transform,
                                      train_transforms=self.train_transform,
                                      val_transforms=self.val_transform,
                                      n_prediction_steps=self.n_prediction_steps,
                                      static_features=self.static_features,
                                      known_future_features=self.known_future_features,
                                      sp=self.sp,
                                      only_has_past_targets=True,
                                      compute_mase_coefficient_value=False,
                                      time_features=cached_time_features)

    def get_test_target(self, test_idx: int):
        if self.only_has_past_targets:
            raise ValueError("get_test_target is not supported for the sequence that only has past targets!")
        if test_idx < 0:
            test_idx = self.__len__() + test_idx
        Y_future = self.Y[test_idx + 1: test_idx + self.n_prediction_steps + 1]
        return Y_future

    def update_attribute(self, **kwargs):
        for key, value in kwargs.items():
            if not hasattr(self, key):
                raise ValueError('Trying to update invalid attribute for TimeSeriesSequence!!!')
            setattr(self, key, value)


class TimeSeriesForecastingDataset(BaseDataset, ConcatDataset):
    datasets: List[TimeSeriesSequence]
    cumulative_sizes: List[int]

    def __init__(self,
                 X: Optional[Union[np.ndarray, List[List]]],
                 Y: Union[np.ndarray, pd.Series],
                 X_test: Optional[Union[np.ndarray, pd.DataFrame]] = None,
                 Y_test: Optional[Union[np.ndarray, pd.DataFrame]] = None,
                 start_times_train: Optional[List[pd.DatetimeIndex]] = None,
                 start_times_test: Optional[List[pd.DatetimeIndex]] = None,
                 known_future_features: Optional[Tuple[str]] = None,
                 time_feature_transform: Optional[List[TimeFeature]] = None,
                 freq: Optional[Union[str, int, List[int]]] = None,
                 resampling_strategy: Optional[Union[
                     CrossValTypes, HoldoutValTypes]] = HoldoutValTypes.time_series_hold_out_validation,
                 resampling_strategy_args: Optional[Dict[str, Any]] = None,
                 shuffle: Optional[bool] = True,
                 seed: Optional[int] = 42,
                 train_transforms: Optional[torchvision.transforms.Compose] = None,
                 val_transforms: Optional[torchvision.transforms.Compose] = None,
                 validator: Optional[TimeSeriesForecastingInputValidator] = None,
                 lagged_value: Optional[List[int]] = None,
                 n_prediction_steps: int = 1,
                 dataset_name: Optional[str] = None,
                 normalize_y: bool = False,
                 ):
        """
        :param target_variables:  Optional[Union[Tuple[int], int]] used for multi-variant forecasting
        tasks, the target_variables indicates which values in X corresponds to Y.
        TODO add supports on X for pandas and target variables can be str or Tuple[str]
        :param freq: Optional[Union[str, int]] frequency of the series sequences, used to determine the (possible)
        period
        :param lagged_value: lagged values applied to RNN and Transformer that allows them to use previous data
        :param n_prediction_steps: The number of steps you want to forecast into the future
        :param shift_input_data: bool
        if the input X and targets needs to be shifted to be aligned:
        such that the data until X[t] is applied to predict the value y[t+n_prediction_steps]
        :param normalize_y: bool
        if y values needs to be normalized with mean 0 and variance 1
        if the dataset is trained with log_prob losses, this needs to be specified in the very beginning such that the
        header's configspace can be built beforehand.
        :param static_features: statistic features, invariant across different
        """
        assert X is not Y, "Training and Test data needs to belong two different object!!!"

        if freq is None:
            self.freq = None
            self.freq_value = None

        if isinstance(freq, str):
            if freq not in SEASONALITY_MAP:
                Warning("The given freq name is not supported by our dataset, we will use the default "
                        "configuration space on the hyperparameter window_size, if you want to adapt this value"
                        "you could pass freq with a numerical value")
            freq_value = SEASONALITY_MAP.get(freq, None)
        else:
            freq_value = freq

        if isinstance(freq_value, list):
            min_base_size = min(n_prediction_steps, MAX_WINDOW_SIZE_BASE)
            if np.max(freq_value) < min_base_size:
                tmp_freq = max(freq_value)
            else:
                tmp_freq = min([freq_value_item for
                                freq_value_item in freq_value if freq_value_item >= min_base_size])
            freq_value = tmp_freq

        seasonality = SEASONALITY_MAP.get(freq, 1)
        if isinstance(seasonality, list):
            seasonality = min(seasonality)  # Use to calculate MASE
        self.seasonality = int(seasonality)

        self.freq: Optional[str] = freq
        self.freq_value: Optional[int] = freq_value

        self.dataset_name = dataset_name

        if self.dataset_name is None:
            self.dataset_name = str(uuid.uuid1(clock_seq=os.getpid()))

        self.n_prediction_steps = n_prediction_steps
        if validator is None:
            validator = TimeSeriesForecastingInputValidator(is_classification=False)
        self.validator: TimeSeriesForecastingInputValidator = validator

        if not isinstance(validator, TimeSeriesForecastingInputValidator):
            raise ValueError(f"This dataset only support TimeSeriesForecastingInputValidator "
                             f"but receive {type(validator)}")

        if not self.validator._is_fitted:
            self.validator.fit(X_train=X, y_train=Y, X_test=X_test, y_test=Y_test,
                               start_times_train=start_times_train, start_times_test=start_times_test,
                               n_prediction_steps=n_prediction_steps)

        self.is_uni_variant = self.validator._is_uni_variant

        self.numerical_columns = self.validator.feature_validator.numerical_columns
        self.categorical_columns = self.validator.feature_validator.categorical_columns

        self.num_features = self.validator.feature_validator.num_features  # type: int
        self.num_target = self.validator.target_validator.out_dimensionality  # type: int

        self.categories = self.validator.feature_validator.categories

        self.feature_shapes = self.validator.feature_shapes
        self.feature_names = tuple(self.validator.feature_names)

        self.start_times_train = self.validator.start_times_train
        self.start_times_test = self.validator.start_times_test

        self.static_features = self.validator.feature_validator.static_features

        self._transform_time_feature = False
        if not time_feature_transform:
            time_feature_transform = time_features_from_frequency_str(self.freq)
            if not time_feature_transform:
                # If time features are empty (as for yearly data), we add a
                # constant feature of 0
                time_feature_transform = [ConstantTransform()]

        self.time_feature_transform = time_feature_transform
        self.time_feature_names = tuple([f'time_feature_{t.__class__.__name__}' for t in self.time_feature_transform])

        # Time features are lazily generated, we do not count them as either numerical_columns or categorical columns

        X, Y, sequence_lengths = self.validator.transform(X, Y)
        time_features_train = self.compute_time_features(self.start_times_train, sequence_lengths)

        if Y_test is not None:
            X_test, Y_test, self.sequence_lengths_tests = self.validator.transform(X_test, Y_test)
            time_features_test = self.compute_time_features(self.start_times_test, self.sequence_lengths_tests)
        else:
            self.sequence_lengths_tests = None
            time_features_test = None

        y_groups = Y.groupby(Y.index)
        if normalize_y:
            mean = y_groups.transform("mean")
            std = y_groups.transform("std")
            std[std == 0] = 1.
            Y = (Y[mean.columns] - mean) / std
            if Y_test is not None:
                y_groups_test = Y_test.groupby(Y.index)

                mean = y_groups_test.transform("mean")
                std = y_groups_test.transform("std")
                std[std == 0] = 1.
                Y_test = (Y_test[mean.columns] - mean) / std

        self.shuffle = shuffle
        self.random_state = np.random.RandomState(seed=seed)

        # check if dataset could be split with cross validation
        minimal_seq_length = np.min(sequence_lengths) - n_prediction_steps
        if isinstance(resampling_strategy, CrossValTypes):
            num_splits = DEFAULT_RESAMPLING_PARAMETERS[resampling_strategy].get(
                'num_splits', None)
            if resampling_strategy_args is not None:
                num_splits = resampling_strategy_args.get('num_split', num_splits)

            if resampling_strategy != CrossValTypes.time_series_ts_cross_validation:
                while minimal_seq_length - n_prediction_steps * num_splits <= 0:
                    num_splits -= 1

                if num_splits >= 2:
                    resampling_strategy = CrossValTypes.time_series_cross_validation
                    if resampling_strategy_args is None:
                        resampling_strategy_args = {'num_splits': num_splits}
                    else:
                        resampling_strategy_args.update({'num_splits': num_splits})
                else:
                    warnings.warn('The dataset is not suitable for cross validation, we will apply holdout instead')

                    resampling_strategy = HoldoutValTypes.time_series_hold_out_validation
                    resampling_strategy_args = None
            else:
                seasonality_h_value = int(
                    np.round((self.n_prediction_steps // int(self.freq_value) + 1) * self.freq_value))

                while minimal_seq_length < (num_splits - 1) * freq_value + seasonality_h_value - n_prediction_steps:
                    if num_splits <= 2:
                        break
                    num_splits -= 1
                if resampling_strategy_args is None:
                    resampling_strategy_args = {'num_splits': num_splits}
                else:
                    resampling_strategy_args.update({'num_splits': num_splits})

        num_seqs = len(sequence_lengths)

        if resampling_strategy_args is not None and "n_repeat" not in resampling_strategy_args:
            n_repeat = resampling_strategy_args["n_repeat"]
        else:
            n_repeat = None
        if (num_seqs < 100 and minimal_seq_length > 10 * n_prediction_steps) or \
                minimal_seq_length > 50 * n_prediction_steps:
            if n_repeat is None:
                if num_seqs < 100:
                    n_repeat = int(np.ceil(100.0 / num_seqs))
                else:
                    n_repeat = int(np.round(minimal_seq_length / (50 * n_prediction_steps)))

            if resampling_strategy == CrossValTypes.time_series_cross_validation:
                n_repeat = min(n_repeat, minimal_seq_length // (5 * n_prediction_steps * num_splits))
            elif resampling_strategy == CrossValTypes.time_series_ts_cross_validation:
                seasonality_h_value = int(np.round(
                    (self.n_prediction_steps * n_repeat // int(self.freq_value) + 1) * self.freq_value)
                )

                while minimal_seq_length // 5 < (num_splits - 1) * n_repeat * freq_value \
                        + seasonality_h_value - n_repeat * n_prediction_steps:
                    n_repeat -= 1
                    seasonality_h_value = int(np.round(
                        (self.n_prediction_steps * n_repeat // int(self.freq_value) + 1) * self.freq_value)
                    )
            elif resampling_strategy == HoldoutValTypes.time_series_hold_out_validation:
                n_repeat = min(n_repeat, minimal_seq_length // (5 * n_prediction_steps) - 1)

            else:
                n_repeat = 1

            n_repeat = max(n_repeat, 1)
        if n_repeat is None:
            n_repeat = 1

        if resampling_strategy_args is None:
            resampling_strategy_args = {'n_repeat': n_repeat}
        else:
            resampling_strategy_args.update({'n_repeat': n_repeat})

        self.resampling_strategy = resampling_strategy
        self.resampling_strategy_args = resampling_strategy_args

        # We also need to be able to transform the data, be it for pre-processing
        # or for augmentation
        self.train_transform = train_transforms
        self.val_transform = val_transforms

        self.num_sequences = len(Y)
        self.sequence_lengths_train = np.asarray(sequence_lengths) - n_prediction_steps

        if known_future_features is None:
            known_future_features = tuple()

        # initialize datasets
        sequences_kwargs = {"freq": self.freq,
                            "time_feature_transform": self.time_feature_transform,
                            "train_transforms": self.train_transform,
                            "val_transforms": self.val_transform,
                            "n_prediction_steps": n_prediction_steps,
                            "sp": self.seasonality,
                            "known_future_features": known_future_features,
                            "static_features": self.static_features}

        sequence_datasets, train_tensors, test_tensors = self.make_sequences_datasets(
            X=X, Y=Y,
            X_test=X_test, Y_test=Y_test,
            start_times_train=self.start_times_train,
            start_times_test=self.start_times_test,
            time_features_train=time_features_train,
            time_features_test=time_features_test,
            **sequences_kwargs)
        self.normalize_y = normalize_y

        ConcatDataset.__init__(self, datasets=sequence_datasets)
        self.known_future_features = known_future_features

        self.seq_length_min = int(np.min(self.sequence_lengths_train))
        self.seq_length_median = int(np.median(self.sequence_lengths_train))
        self.seq_length_max = int(np.max(self.sequence_lengths_train))

        if freq_value > self.seq_length_median:
            self.base_window_size = self.seq_length_median
        else:
            self.base_window_size = freq_value

        self.train_tensors = train_tensors

        self.test_tensors = test_tensors
        self.val_tensors = None

        self.task_type: Optional[str] = None
        self.issparse: bool = issparse(self.train_tensors[0])
        # TODO find a way to edit input shape!
        self.input_shape: Tuple[int, int] = (self.seq_length_min, self.num_features)

        if known_future_features is None:
            self.future_feature_shapes: Tuple[int, int] = (self.seq_length_min, 0)
        else:
            self.future_feature_shapes: Tuple[int, int] = (self.seq_length_min, len(known_future_features))

        if len(self.train_tensors) == 2 and self.train_tensors[1] is not None:
            self.output_type: str = type_of_target(self.train_tensors[1][0].fillna(method="pad"))

            if self.output_type in ["binary", "multiclass"]:
                self.output_type = "continuous"

            if STRING_TO_OUTPUT_TYPES[self.output_type] in CLASSIFICATION_OUTPUTS:
                num_target = len(np.unique(Y))
                # self.output_shape = len(np.unique(Y))
            else:
                # self.output_shape = self.train_tensors[1].shape[-1] if self.train_tensors[1].ndim > 1 else 1
                num_target = Y.shape[-1] if Y.ndim > 1 else 1
            self.output_shape = [self.n_prediction_steps, num_target]
        else:
            raise ValueError('Forecasting dataset must contain target values!')

        # TODO: Look for a criteria to define small enough to preprocess
        self.is_small_preprocess = True

        self.task_type = TASK_TYPES_TO_STRING[TIMESERIES_FORECASTING]

        self.numerical_features: List[int] = self.numerical_columns
        self.categorical_features: List[int] = self.categorical_columns

        if isinstance(resampling_strategy, CrossValTypes):
            self.cross_validators = CrossValFuncs.get_cross_validators(resampling_strategy)
        else:
            self.cross_validators = CrossValFuncs.get_cross_validators(CrossValTypes.time_series_cross_validation)
        if isinstance(resampling_strategy, HoldoutValTypes):
            self.holdout_validators = HoldOutFuncs.get_holdout_validators(resampling_strategy)

        else:
            self.holdout_validators = HoldOutFuncs.get_holdout_validators(
                HoldoutValTypes.time_series_hold_out_validation)

        self.splits = self.get_splits_from_resampling_strategy()

        # TODO doing experiments to give the most proper way of defining these two values
        if lagged_value is None:
            try:
                lagged_value = [0] + get_lags_for_frequency(freq)
            except Exception:
                lagged_value = list(range(8))

        self.lagged_value = lagged_value

    def compute_time_features(self,
                              start_times: List[pd.DatetimeIndex],
                              seq_lengths: List[int]) -> Dict[pd.DatetimeIndex, np.ndarray]:
        """
        compute the max series length for each start_time and compute their corresponding time_features. As lots of
        series in a dataset share the same start time, we could only compute the features for longest possible series
        and reuse them
        """
        series_lengths_max = {}
        for start_t, seq_l in zip(start_times, seq_lengths):
            if start_t not in series_lengths_max or seq_l > series_lengths_max[start_t]:
                series_lengths_max[start_t] = seq_l
        series_time_features = {}
        for start_t, max_l in series_lengths_max.items():
            try:
                date_info = pd.date_range(start=start_t,
                                          periods=max_l,
                                          freq=self.freq)
                series_time_features[start_t] = np.vstack(
                    [transform(date_info).to_numpy(float)
                     if not isinstance(transform, ConstantTransform) else transform(date_info)
                     for transform in self.time_feature_transform]
                ).T
            except OutOfBoundsDatetime as e:
                series_time_features[start_t] = np.zeros([max_l, len(self.time_feature_transform)])
        return series_time_features

    def _get_dataset_indices(self, idx: int, only_dataset_idx: bool = False) -> Union[int, Tuple[int, int]]:
        if idx < 0:
            if -idx > len(self):
                raise ValueError("absolute value of index should not exceed dataset length")
            idx = len(self) + idx
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if only_dataset_idx:
            return dataset_idx
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        return dataset_idx, sample_idx

    def __getitem__(self, idx, train=True):
        dataset_idx, sample_idx = self._get_dataset_indices(idx)
        return self.datasets[dataset_idx].__getitem__(sample_idx, train)

    def get_validation_set(self, idx):
        dataset_idx, sample_idx = self._get_dataset_indices(idx)
        return self.datasets[dataset_idx].get_val_seq_set(sample_idx)

    def get_time_series_seq(self, idx) -> TimeSeriesSequence:
        dataset_idx = self._get_dataset_indices(idx, True)
        return self.datasets[dataset_idx]

    def get_test_target(self, test_indices: np.ndarray) -> np.ndarray:
        test_indices = np.where(test_indices < 0, test_indices + len(self), test_indices)
        y_test = np.ones([len(test_indices), self.n_prediction_steps, self.num_target])
        y_test_argsort = np.argsort(test_indices)
        dataset_idx = self._get_dataset_indices(test_indices[y_test_argsort[0]], only_dataset_idx=True)

        for y_i in y_test_argsort:
            test_idx = test_indices[y_i]
            while test_idx > self.cumulative_sizes[dataset_idx]:
                dataset_idx += 1
            if dataset_idx != 0:
                test_idx = test_idx - self.cumulative_sizes[dataset_idx - 1]
            y_test[y_i] = self.datasets[dataset_idx].get_test_target(test_idx)

        return y_test.reshape([-1, self.num_target])

    def make_sequences_datasets(self,
                                X: pd.DataFrame,
                                Y: pd.DataFrame,
                                start_times_train: List[pd.DatetimeIndex],
                                time_features_train: Optional[Dict[pd.Timestamp, np.ndarray]] = None,
                                X_test: Optional[pd.DataFrame] = None,
                                Y_test: Optional[pd.DataFrame] = None,
                                start_times_test: Optional[List[pd.DatetimeIndex]] = None,
                                time_features_test: Optional[Dict[pd.Timestamp, np.ndarray]] = None,
                                **sequences_kwargs: Optional[Dict]) -> Tuple[
        List[TimeSeriesSequence],
        Tuple[Optional[pd.DataFrame], pd.DataFrame],
        Optional[Tuple[pd.DataFrame, pd.DataFrame]]
    ]:
        """
        build a series time sequence datasets
        Args:
            X: pd.DataFrame (N_all, N_feature)
                flattened train feature DataFrame with size N_all (the sum of all the series sequences) and N_feature,
                number of features, X's index should contain the information identifying its series number
            Y: pd.DataFrame (N_all, N_target)
                flattened train target array with size N_all (the sum of all the series sequences) and number of targets
            start_times_train: List[pd.DatetimeIndex]
                start time of each training series
            time_features_train: Dict[pd.Timestamp, np.ndarray]:
                time features for each possible start training times
            X_test: Optional[np.ndarray (N_all_test, N_feature)]
                flattened test feature array with size N_all_test (the sum of all the series sequences) and N_feature,
                number of features
            Y_test: np.ndarray (N_all_test, N_target)
                flattened test target array with size N_all (the sum of all the series sequences) and number of targets
            start_times_test: Optional[List[pd.DatetimeIndex]]
                start time for each test series
            time_features_test:Optional[Dict[pd.Timestamp, np.ndarray]]
                time features for each possible start test times.
            sequences_kwargs: Dict
                additional arguments for test sets
        Returns:
            sequence_datasets : List[TimeSeriesSequence]
                a
            train_tensors: Tuple[List[np.ndarray], List[np.ndarray]]
                training tensors
            test_tensors: Option[Tuple List[np.ndarray, List[np.ndarray]]
                test tensors

        """
        sequence_datasets = []

        y_group = Y.groupby(Y.index)
        if X is not None:
            x_group = X.groupby(X.index)
        if Y_test is not None:
            y_test_group = Y_test.groupby(Y_test.index)
        if X_test is not None:
            x_test_group = X_test.groupby(X_test.index)

        for i_ser, (start_train, y) in enumerate(zip(start_times_train, y_group)):
            ser_id = y[0]
            y_ser = y[1].transform(np.array).values
            x_ser = x_group.get_group(ser_id).transform(np.array).values if X is not None else None

            y_test_ser = y_test_group.get_group(ser_id).transform(np.array).values if Y_test is not None else None
            x_test_ser = x_test_group.get_group(ser_id).transform(np.array).values if X_test is not None else None

            start_test = None if start_times_test is None else start_times_test[i_ser]
            time_feature_test = None if time_features_test is None else time_features_test[start_test][:len(y_test_ser)]

            sequence = TimeSeriesSequence(
                X=x_ser,
                Y=y_ser,
                start_time_train=start_train,
                X_test=x_test_ser,
                Y_test=y_test_ser,
                start_time_test=start_test,
                time_features=time_features_train[start_train][:len(y_ser)],
                time_features_test=time_feature_test,
                **sequences_kwargs)
            sequence_datasets.append(sequence)

        train_tensors = (X, Y)
        if Y_test is None:
            test_tensors = None
        else:
            # test_tensors = (X_test_seq_all, Y_test_seq_all)
            test_tensors = (X_test, Y_test)

        return sequence_datasets, train_tensors, test_tensors

    def replace_data(self, X_train: pd.DataFrame, X_test: Optional[pd.DataFrame]) -> 'BaseDataset':
        super(TimeSeriesForecastingDataset, self).replace_data(X_train=X_train, X_test=X_test)
        if X_train is None:
            return self
        if X_test is not None:
            X_test_group = X_test.groupby(X_test.index)
        for seq, x in zip(self.datasets, X_train.groupby(X_train.index)):
            ser_id = x[0]
            x_ser = x[1].transform(np.array).values
            seq.X = x_ser
            if X_test is not None:
                seq.X_test = X_test_group.get_group(ser_id).transform(np.array).values

        return self

    def update_transform(self, transform: Optional[torchvision.transforms.Compose],
                         train: bool = True,
                         ) -> 'BaseDataset':
        """
        During the pipeline execution, the pipeline object might propose transformations
        as a product of the current pipeline configuration being tested.

        This utility allows to return a self with the updated transformation, so that
        a dataloader can yield this dataset with the desired transformations

        Args:
            transform (torchvision.transforms.Compose): The transformations proposed
                by the current pipeline
            train (bool): Whether to update the train or validation transform

        Returns:
            self: A copy of the update pipeline
        """
        if train:
            self.train_transform = transform
        else:
            self.val_transform = transform
        for seq in self.datasets:
            seq = seq.update_transform(transform, train)
        return self

    @property
    def transform_time_features(self):
        return self._transform_time_features

    @transform_time_features.setter
    def transform_time_features(self, value: bool):
        for seq in self.datasets:
            seq.transform_time_features = value

    def get_splits_from_resampling_strategy(self) -> List[Tuple[List[int], List[int]]]:
        """
        Creates a set of splits based on a resampling strategy provided, apart from the
        'get_splits_from_resampling_strategy' implemented in base_dataset, here we will get self.upper_sequence_length
        with the given value

        Returns
            (List[Tuple[List[int], List[int]]]): splits in the [train_indices, val_indices] format
        """
        splits = []
        if isinstance(self.resampling_strategy, HoldoutValTypes):
            val_share = DEFAULT_RESAMPLING_PARAMETERS[self.resampling_strategy].get(
                'val_share', None)
            if self.resampling_strategy_args is not None:
                val_share = self.resampling_strategy_args.get('val_share', val_share)
                n_repeat = self.resampling_strategy_args.get("n_repeat", 1)
            else:
                n_repeat = 1
            splits.append(self.create_holdout_val_split(holdout_val_type=self.resampling_strategy,
                                                        val_share=val_share,
                                                        n_repeat=n_repeat))

        elif isinstance(self.resampling_strategy, CrossValTypes):
            num_splits = DEFAULT_RESAMPLING_PARAMETERS[self.resampling_strategy].get(
                'num_splits', None)
            if self.resampling_strategy_args is not None:
                num_splits = self.resampling_strategy_args.get('num_splits', num_splits)
                n_repeat = self.resampling_strategy_args.get("n_repeat", 1)
            else:
                n_repeat = 1
            # Create the split if it was not created before
            splits.extend(self.create_cross_val_splits(
                cross_val_type=self.resampling_strategy,
                num_splits=cast(int, num_splits),
                n_repeat=n_repeat
            ))
        elif self.resampling_strategy is None:
            splits.append(self.create_refit_split())
        else:
            raise ValueError(f"Unsupported resampling strategy={self.resampling_strategy}")
        return splits

    def get_required_dataset_info(self) -> Dict[str, Any]:
        """
        Returns a dictionary containing required dataset properties to instantiate a pipeline,
        """
        info = super().get_required_dataset_info()
        info.update({
            'task_type': self.task_type,
            'numerical_features': self.numerical_features,
            'categorical_features': self.categorical_features,
            'numerical_columns': self.numerical_columns,
            'categorical_columns': self.categorical_columns,
            'categories': self.categories,
        })
        return info

    def get_dataset_properties(self, dataset_requirements: List[FitRequirement]) -> Dict[str, Any]:
        dataset_properties = super().get_dataset_properties(dataset_requirements=dataset_requirements)
        dataset_properties.update({'n_prediction_steps': self.n_prediction_steps,
                                   'sp': self.seasonality,  # For metric computation,
                                   'freq': self.freq,
                                   'sequence_lengths_train': self.sequence_lengths_train,
                                   'seq_length_max': self.seq_length_max,
                                   'input_shape': self.input_shape,
                                   'lagged_value': self.lagged_value,
                                   'feature_names': self.feature_names,
                                   'feature_shapes': self.feature_shapes,
                                   'known_future_features': self.known_future_features,
                                   'static_features': self.static_features,
                                   'time_feature_transform': self.time_feature_transform,
                                   'time_feature_names': self.time_feature_names,
                                   'future_feature_shapes': self.future_feature_shapes,
                                   'uni_variant': self.is_uni_variant,
                                   'targets_have_missing_values': self.train_tensors[1].isnull().values.any(),
                                   'features_have_missing_values': False if self.train_tensors[0] is None
                                   else self.train_tensors[0].isnull().values.any()})
        return dataset_properties

    def create_cross_val_splits(
            self,
            cross_val_type: CrossValTypes,
            num_splits: int,
            n_repeat=1,
    ) -> List[Tuple[Union[List[int], np.ndarray], Union[List[int], np.ndarray]]]:
        """
        This function creates the cross validation split for the given task.

        It is done once per dataset to have comparable results among pipelines
        Args:
            cross_val_type (CrossValTypes):
            num_splits (int): number of splits to be created
            n_repeat (int): how many n_prediction_steps to repeat in the validation set

        Returns:
            (List[Tuple[Union[List[int], np.ndarray], Union[List[int], np.ndarray]]]):
                list containing 'num_splits' splits.
        """
        # Create just the split once
        # This is gonna be called multiple times, because the current dataset
        # is being used for multiple pipelines. That is, to be efficient with memory
        # we dump the dataset to memory and read it on a need basis. So this function
        # should be robust against multiple calls, and it does so by remembering the splits

        if not isinstance(cross_val_type, CrossValTypes):
            raise NotImplementedError(f'The selected `cross_val_type` "{cross_val_type}" is not implemented.')
        idx_start = 0

        kwargs = {"n_prediction_steps": self.n_prediction_steps}
        if cross_val_type == CrossValTypes.time_series_ts_cross_validation:
            seasonality_h_value = int(np.round((self.n_prediction_steps // int(self.freq_value) + 1) * self.freq_value))
            kwargs.update({'seasonality_h_value': seasonality_h_value,
                           'freq_value': self.freq_value})
        kwargs["n_repeat"] = n_repeat

        splits = [[() for _ in range(len(self.datasets))] for _ in range(num_splits)]

        for idx_seq, dataset in enumerate(self.datasets):
            split = self.cross_validators[cross_val_type.name](self.random_state,
                                                               num_splits,
                                                               indices=idx_start + np.arange(len(dataset)),
                                                               **kwargs)

            for idx_split in range(num_splits):
                splits[idx_split][idx_seq] = split[idx_split]
            idx_start += self.sequence_lengths_train[idx_seq]
        # in this case, splits is stored as :
        #  [ first split, second_split ...]
        #  first_split = [([0], [1]), ([2], [3])] ....
        splits_merged = []
        for i in range(num_splits):
            split = splits[i]
            train_indices = np.hstack([sp[0] for sp in split])
            test_indices = np.hstack([sp[1] for sp in split])
            splits_merged.append((train_indices, test_indices))
        return splits_merged

    def create_holdout_val_split(
            self,
            holdout_val_type: HoldoutValTypes,
            val_share: float,
            n_repeat: int = 1,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        This function creates the holdout split for the given task.

        It is done once per dataset to have comparable results among pipelines
        Args:
            holdout_val_type (HoldoutValTypes):
            val_share (float): share of the validation data
            n_repeat (int): how many n_prediction_steps to repeat in the validation set

        Returns:
            (Tuple[np.ndarray, np.ndarray]): Tuple containing (train_indices, val_indices)
        """
        if holdout_val_type is None:
            raise ValueError(
                '`val_share` specified, but `holdout_val_type` not specified.'
            )

        if val_share < 0 or val_share > 1:
            raise ValueError(f"`val_share` must be between 0 and 1, got {val_share}.")
        if not isinstance(holdout_val_type, HoldoutValTypes):
            raise NotImplementedError(f'The specified `holdout_val_type` "{holdout_val_type}" is not supported.')
        kwargs = {"n_prediction_steps": self.n_prediction_steps,
                  "n_repeat": n_repeat}

        splits = [[() for _ in range(len(self.datasets))] for _ in range(2)]
        idx_start = 0
        for idx_seq, dataset in enumerate(self.datasets):

            split = self.holdout_validators[holdout_val_type.name](self.random_state,
                                                                   val_share,
                                                                   indices=np.arange(len(dataset)) + idx_start,
                                                                   **kwargs)
            for idx_split in range(2):
                splits[idx_split][idx_seq] = split[idx_split]
            idx_start += self.sequence_lengths_train[idx_seq]

        train_indices = np.hstack([sp for sp in splits[0]])
        test_indices = np.hstack([sp for sp in splits[1]])

        return train_indices, test_indices

    def create_refit_split(
            self,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        This function creates the refit split for the given task. All the data in the dataset will be considered as
        training sets
        Args:
            holdout_val_type (HoldoutValTypes):
            val_share (float): share of the validation data

        Returns:
            (Tuple[np.ndarray, np.ndarray]): Tuple containing (train_indices, val_indices)
        """
        kwargs = {"n_prediction_steps": self.n_prediction_steps}

        splits = [[() for _ in range(len(self.datasets))] for _ in range(2)]
        idx_start = 0
        for idx_seq, dataset in enumerate(self.datasets):
            split = [np.arange(len(dataset)), np.array([len(dataset) - 1])]

            for idx_split in range(2):
                splits[idx_split][idx_seq] = idx_start + split[idx_split]
            idx_start += self.sequence_lengths_train[idx_seq]

        train_indices = np.hstack([sp for sp in splits[0]])
        test_indices = np.hstack([sp for sp in splits[1]])

        return train_indices, test_indices

    def create_refit_set(self) -> "TimeSeriesForecastingDataset":
        refit_set: TimeSeriesForecastingDataset = copy.deepcopy(self)
        refit_set.resampling_strategy = None
        refit_set.splits = refit_set.get_splits_from_resampling_strategy()
        return refit_set

    def generate_test_seqs(self) -> List[TimeSeriesSequence]:
        test_sets = copy.deepcopy(self.datasets)
        for test_seq in test_sets:
            test_seq.is_test_set = True
            test_seq.only_has_past_targets = True
        return test_sets
