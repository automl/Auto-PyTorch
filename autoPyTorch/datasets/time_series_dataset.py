import bisect
import copy
import os
import uuid
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union, cast

from gluonts.time_feature import Constant as ConstantTransform
from gluonts.time_feature import TimeFeature, time_features_from_frequency_str
from gluonts.time_feature.lag import get_lags_for_frequency

import numpy as np

import pandas as pd
from pandas._libs.tslibs.np_datetime import OutOfBoundsDatetime

from scipy.sparse import issparse

import torch
from torch.utils.data.dataset import ConcatDataset, Dataset

import torchvision.transforms

from autoPyTorch.constants import (
    CLASSIFICATION_OUTPUTS,
    MAX_WINDOW_SIZE_BASE,
    SEASONALITY_MAP,
    STRING_TO_OUTPUT_TYPES,
    TASK_TYPES_TO_STRING,
    TIMESERIES_FORECASTING
)
from autoPyTorch.data.time_series_forecasting_validator import TimeSeriesForecastingInputValidator
from autoPyTorch.datasets.base_dataset import BaseDataset, type_of_target
from autoPyTorch.datasets.resampling_strategy import (
    CrossValFuncs,
    CrossValTypes,
    DEFAULT_RESAMPLING_PARAMETERS,
    HoldOutFuncs,
    HoldoutValTypes,
    NoResamplingStrategyTypes,
    ResamplingStrategies
)
from autoPyTorch.pipeline.components.training.metrics.metrics import compute_mase_coefficient
from autoPyTorch.utils.common import FitRequirement


def extract_feature_index(feature_shapes: Dict[str, int],
                          feature_names: Tuple[str],
                          queried_features: Union[Tuple[Union[str, int]], Tuple[()]]) -> Tuple[int]:
    """
    extract the index of a set of queried_features from the extracted feature_shapes

    Args:
        feature_shapes (dict):
            feature_shapes recoding the shape of each features
        feature_names (List[str]):
            names of the features
        queried_features (Tuple[str]):
            names of the features that we expect their index

    Returns:
        feature_index (Tuple[int]):
            indices of the corresponding features
    """
    df_range = pd.DataFrame(feature_shapes, columns=feature_names, index=[0])
    df_range_end = df_range.cumsum(axis=1)
    df_range = pd.concat([df_range_end - df_range, df_range_end])
    value_ranges = df_range[list(queried_features)].T.values
    feature_index: List[int] = sum([list(range(*value_r)) for value_r in value_ranges], [])
    feature_index.sort()
    return tuple(feature_index)  # type: ignore[return-value]


def compute_time_features(start_time: pd.DatetimeIndex,
                          date_period_length: int,
                          time_feature_length: int,
                          freq: str,
                          time_feature_transforms: List[TimeFeature]) -> np.ndarray:
    date_info = pd.date_range(start=start_time,
                              periods=date_period_length,
                              freq=freq)[-time_feature_length:]
    try:
        time_features = np.vstack(
            [transform(date_info) for transform in time_feature_transforms]
        ).T
    except OutOfBoundsDatetime:
        # This is only a temporal solution TODO consider how to solve this!
        time_features = np.zeros([time_feature_length, len(time_feature_transforms)])
    return time_features


class TimeSeriesSequence(Dataset):
    """
    A dataset representing a time series sequence. It returns all the previous observations once it is asked for an item

    Args:
        X (Optional[np.ndarray]):
            past features
        Y (np.ndarray):
            past targets
        start_time (Optional[pd.DatetimeIndex]):
            times of the first timestep of the series
        freq (str):
            frequency that the data is sampled
        time_feature_transform (List[TimeFeature]):
            available time features applied to the series
        X_test (Optional[np.ndarray]):
            known future features
        Y_test (Optional[np.ndarray]):
            future targets
        train_transforms (Optional[torchvision.transforms.Compose]):
            training transforms, used to transform training features
        val_transforms (Optional[torchvision.transforms.Compose]):
            validation transforms, used to transform training features
        n_prediction_steps (int):
            how many steps need to be predicted in advance
        known_future_features_index (int):
            indices of the known future index
        compute_mase_coefficient_value (bool):
            if the mase coefficient for this series is pre-computed
        time_features (Optional[np.ndarray]):
            pre-computed time features
        is_test_set (bool):
            if this dataset is test sets. Test sequence will simply make X_test and Y_test as future features and
            future targets
    """
    _is_test_set = False
    is_pre_processed = False

    def __init__(self,
                 X: Optional[np.ndarray],
                 Y: np.ndarray,
                 start_time: Optional[pd.DatetimeIndex] = None,
                 freq: str = '1Y',
                 time_feature_transform: List[TimeFeature] = [ConstantTransform],
                 X_test: Optional[np.ndarray] = None,
                 Y_test: Optional[np.ndarray] = None,
                 train_transforms: Optional[torchvision.transforms.Compose] = None,
                 val_transforms: Optional[torchvision.transforms.Compose] = None,
                 n_prediction_steps: int = 1,
                 sp: int = 1,
                 known_future_features_index: Optional[Tuple[int]] = None,
                 compute_mase_coefficient_value: bool = True,
                 time_features: Optional[np.ndarray] = None,
                 is_test_set: bool = False,
                 ) -> None:
        self.n_prediction_steps = n_prediction_steps

        if X is not None and X.ndim == 1:
            X = X[:, np.newaxis]
        self.X = X
        self.Y = Y

        self.observed_target = ~np.isnan(self.Y)
        if start_time is None:
            start_time = pd.Timestamp('1900-01-01')
        self.start_time = start_time

        self.X_val = None
        self.Y_val = None

        if X_test is not None and X_test.ndim == 1:
            X_test = X_test[:, np.newaxis]

        self.X_test = X_test
        self.Y_test = Y_test

        self.time_feature_transform = time_feature_transform

        self.freq = freq

        # We also need to be able to transform the data, be it for pre-processing
        # or for augmentation
        self.train_transform = train_transforms
        self.val_transform = val_transforms
        self.sp = sp

        if compute_mase_coefficient_value:
            if is_test_set:
                self.mase_coefficient = compute_mase_coefficient(self.Y, sp=self.sp)
            else:
                self.mase_coefficient = compute_mase_coefficient(self.Y[:-n_prediction_steps], sp=self.sp)

        else:
            self.mase_coefficient = np.asarray([1.0])
        self.known_future_features_index = known_future_features_index

        self.transform_time_features = False
        self._cached_time_features: Optional[np.ndarray] = time_features

        self.future_observed_target = None
        self.is_test_set = is_test_set

    @property
    def is_test_set(self) -> bool:
        return self._is_test_set

    @is_test_set.setter
    def is_test_set(self, value: bool) -> None:
        if value and value != self._is_test_set:
            if self.known_future_features_index:
                if self.X_test is None:
                    raise ValueError('When future features are known, X_test '
                                     'for Time Series Sequences must be given!')
        if self.Y_test is not None:
            self.future_observed_target = ~np.isnan(self.Y_test)
        self._is_test_set = value

    def __getitem__(self, index: int, train: bool = True) \
            -> Tuple[Dict[str, torch.Tensor], Optional[Dict[str, torch.Tensor]]]:
        """
        get a subsequent of time series data, unlike vanilla tabular dataset, we obtain all the previous observations
        until the given index

        Args:
            index (int):
                what element to yield from the series
            train (bool):
                Whether a train or test transformation is applied

        Returns:
            past_information (Dict[str, torch.Tensor]):
                a dict contains all the required information required for future forecasting
                past_targets (torch.Tensor), past_features(Optional[torch.Tensor]),
                future_features(Optional[torch.Tensor]),
                mase_coefficient (np.array, cached value to compute MASE scores),
                past_observed_targets(torch.BoolTensor), if the past targets are observed.
                decoder_lengths(int), length of decoder output
            future_information (Optional[Dict[str, torch.Tensor]]):
                a dict contains all the future information that are required to predict, including
                future_targets: (torch.Tensor) and future_observed_targets (torch.BoolTensor)
        """
        if index < 0:
            index = self.__len__() + index

        if self.X is not None:
            past_features = self.X[:index + 1]

            if self.known_future_features_index:
                if not self.is_test_set:
                    future_features = \
                        self.X[index + 1: index + self.n_prediction_steps + 1, self.known_future_features_index]
                else:
                    if index < self.__len__() - 1:
                        raise ValueError('Test Sequence is only allowed to be accessed with the last index!')
                    future_features = self.X_test[:, self.known_future_features_index]  # type: ignore[index]
            else:
                future_features = None
        else:
            past_features = None
            future_features = None

        if self.train_transform is not None and train and past_features is not None:
            past_features = self.train_transform(past_features)
            if future_features is not None:
                future_features = self.train_transform(future_features)
        elif self.val_transform is not None and not train and past_features is not None:
            past_features = self.val_transform(past_features)
            if future_features is not None:
                future_features = self.val_transform(future_features)

        if self.transform_time_features:
            if self.time_feature_transform:
                self.cache_time_features()

                if past_features is not None:
                    past_features = np.hstack(
                        [past_features, self._cached_time_features[:index + 1]]  # type: ignore[index]
                    )
                else:
                    past_features = self._cached_time_features[:index + 1]  # type: ignore[index]
                if future_features is not None:
                    future_features = np.hstack([
                        future_features,
                        self._cached_time_features[index + 1:index + self.n_prediction_steps + 1]  # type: ignore[index]
                    ])
                else:
                    future_features = self._cached_time_features[index + 1:  # type: ignore[index]
                                                                 index + self.n_prediction_steps + 1]

        if future_features is not None and future_features.shape[0] == 0:
            future_features = None

        # In case of prediction, the targets are not provided
        targets = self.Y
        if self.is_test_set:
            if self.Y_test is not None:
                future_targets: Optional[Dict[str, torch.Tensor]] = {
                    'future_targets': torch.from_numpy(self.Y_test),
                    'future_observed_targets': torch.from_numpy(self.future_observed_target)
                }
            else:
                future_targets = None
        else:
            future_targets_np = targets[index + 1: index + self.n_prediction_steps + 1]
            future_targets_tt = torch.from_numpy(future_targets_np)
            future_targets = {
                'future_targets': future_targets_tt,
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
        return int(self.Y.shape[0]) if self.is_test_set else int(self.Y.shape[0]) - self.n_prediction_steps

    def get_target_values(self, index: int) -> np.ndarray:
        """
        Get the visible targets in the datasets without generating a tensor. This can be used to create a dummy pipeline
        Args:
            index (int):
                target index

        Returns:
            y (np.ndarray):
                the last visible target value
        """
        if index < 0:
            index = self.__len__() + index
        return self.Y[index]

    def cache_time_features(self) -> None:
        """
        compute time features if it is not cached. For test sets, we also need to compute the time features for future
        """
        if self._cached_time_features is None:
            periods = self.Y.shape[0]
            if self.is_test_set:
                periods += self.n_prediction_steps
            self._cached_time_features = compute_time_features(self.start_time, periods,
                                                               periods, self.freq, self.time_feature_transform)

        else:
            if self.is_test_set:
                if self._cached_time_features.shape[0] == self.Y.shape[0]:
                    time_feature_future = compute_time_features(self.start_time,
                                                                self.n_prediction_steps + self.Y.shape[0],
                                                                self.n_prediction_steps,
                                                                self.freq, self.time_feature_transform)
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
            transform (torchvision.transforms.Compose):
                The transformations proposed by the current pipeline
            train (bool):
                    Whether to update the train or validation transform

        Returns:
            self: A copy of the update pipeline
        """
        if train:
            self.train_transform = transform
        else:
            self.val_transform = transform
        return self

    def get_val_seq_set(self, index: int) -> "TimeSeriesSequence":
        if self.is_test_set:
            raise ValueError("get_val_seq_set is not supported for the test sequences!")
        if index < 0:
            index = self.__len__() + index
        if index >= self.__len__() - 1:
            # TODO consider X_test?
            val_set = copy.deepcopy(self)
            if val_set.X is not None:
                val_set.X_test = val_set.X[-self.n_prediction_steps:]
                val_set.X = val_set.X[:-self.n_prediction_steps]
            val_set.Y_test = val_set.Y[-self.n_prediction_steps:]
            val_set.Y = val_set.Y[:-self.n_prediction_steps]
            val_set.future_observed_target = val_set.observed_target[-self.n_prediction_steps:]
            val_set.observed_target = val_set.observed_target[:-self.n_prediction_steps]
            val_set.is_test_set = True

            return val_set
        else:
            if self.X is not None:
                X = self.X[:index + 1]
            else:
                X = None
            if self.known_future_features_index:
                X_test = self.X[index + 1: index + 1 + self.n_prediction_steps]  # type: ignore[index]
            else:
                X_test = None
            if self._cached_time_features is None:
                cached_time_features = None
            else:
                cached_time_features = self._cached_time_features[:index + 1 + self.n_prediction_steps]

            val_set = TimeSeriesSequence(X=X,
                                         Y=self.Y[:index + 1],
                                         X_test=X_test,
                                         Y_test=self.Y[index + 1: index + 1 + self.n_prediction_steps],
                                         start_time=self.start_time,
                                         freq=self.freq,
                                         time_feature_transform=self.time_feature_transform,
                                         train_transforms=self.train_transform,
                                         val_transforms=self.val_transform,
                                         n_prediction_steps=self.n_prediction_steps,
                                         known_future_features_index=self.known_future_features_index,
                                         sp=self.sp,
                                         compute_mase_coefficient_value=False,
                                         time_features=cached_time_features,
                                         is_test_set=True)

            return val_set

    def get_test_target(self, test_idx: int) -> np.ndarray:
        if self.is_test_set:
            raise ValueError("get_test_target is not supported for test sequences!")
        if test_idx < 0:
            test_idx = self.__len__() + test_idx
        Y_future = self.Y[test_idx + 1: test_idx + self.n_prediction_steps + 1]
        return Y_future

    def update_attribute(self, **kwargs: Any) -> None:
        for key, value in kwargs.items():
            if not hasattr(self, key):
                raise ValueError('Trying to update invalid attribute for TimeSeriesSequence!')
            setattr(self, key, value)


class TimeSeriesForecastingDataset(BaseDataset, ConcatDataset):
    """
    Dataset class for time series forecasting used in AutoPyTorch. It consists of multiple TimeSeriesSequence.
    Train and test tensors are stored as pd.DataFrame whereas their index indicates which series the data belongs to

    Args:
        X (Optional[Union[np.ndarray, List[Union[pd.DataFrame, np.ndarray]]]]):
            time series features. can be None if we work with a uni-variant forecasting task
        Y (Union[np.ndarray, List[Union[pd.DataFrame, np.ndarray]]]):
            forecasting targets. Must be given
        X_test (Optional[Union[np.ndarray, List[Union[pd.DataFrame, np.ndarray]]]]):
            known future features. It is a collection of series that has the same amount of data as X. It
            is designed to be at the tail of X. If no feature is known in the future, this value can be omitted.
        Y_test (Optional[Union[np.ndarray, List[Union[pd.DataFrame, np.ndarray]]]] = None):
            future targets. It is a collection of series that has the same data of series as Y. It is designed to be at
            the tail of Y after the timestamps that need to be predicted.
        start_times (Optional[List[pd.DatetimeIndex]]):
            starting time of each series when they are sampled. If it is not given, we simply start with a fixed
            timestamp.
        series_idx (Optional[Union[List[Union[str, int]], str, int]]):
            (only works if X is stored as pd.DataFrame). This value is applied to identify  towhich series the data
            belongs if the data is presented as a "chunk" dataframe
        known_future_features (Optional[Union[Tuple[Union[str, int]], Tuple[()]]]):
            future features that are known in advance. For instance, holidays.
        time_feature_transform (Optional[List[TimeFeature]]):
            A list of time feature transformation methods implemented in gluonts. For more information, please check
            gluonts.time_feature
        freq (Optional[Union[str, int, List[int]]]):
            the frequency that the data is sampled. It needs to keep consistent within one dataset
        resampling_strategy (Optional[ResamplingStrategies])
            resampling strategy. We designed several special resampling resampling_strategy for forecasting tasks.
            Please refer to autoPyTorch.datasets.resampling_strategy
        resampling_strategy_args (Optional[Dict[str, Any]]):
            arguments passed to resampling_strategy
        seed (int):
            random seeds
        train_transforms (Optional[torchvision.transforms.Compose]):
            Transformation applied to training data before it is fed to the dataloader
        val_transforms (Optional[torchvision.transforms.Compose]):
            Transformation applied to validation data before it is fed to the dataloader
        validator (Optional[TimeSeriesForecastingInputValidator]):
            Input Validator
        lagged_value (Optional[List[int]])
            We could consider past targets as additional features for the current timestep. This item indicates the
            number of timesteps in advanced that we want to apply the targets as our current features
        n_prediction_steps (int):
            The number of steps you want to forecast into the future (forecast horizon)
        dataset_name (Optional[str]):
            dataset name
        normalize_y(bool):
            if targets are normalized within each series
    """

    datasets: List[TimeSeriesSequence]
    cumulative_sizes: List[int]

    def __init__(self,
                 X: Optional[Union[np.ndarray, List[Union[pd.DataFrame, np.ndarray]]]],
                 Y: Union[np.ndarray, List[Union[pd.DataFrame, np.ndarray]]],
                 X_test: Optional[Union[np.ndarray, List[Union[pd.DataFrame, np.ndarray]]]] = None,
                 Y_test: Optional[Union[np.ndarray, List[Union[pd.DataFrame, np.ndarray]]]] = None,
                 start_times: Optional[List[pd.DatetimeIndex]] = None,
                 series_idx: Optional[Union[List[Union[str, int]], str, int]] = None,
                 known_future_features: Optional[Union[Tuple[Union[str, int]], Tuple[()]]] = None,
                 time_feature_transform: Optional[List[TimeFeature]] = None,
                 freq: Optional[Union[str, int, List[int]]] = None,
                 resampling_strategy: Optional[ResamplingStrategies] = HoldoutValTypes.time_series_hold_out_validation,
                 resampling_strategy_args: Optional[Dict[str, Any]] = None,
                 seed: Optional[int] = 42,
                 train_transforms: Optional[torchvision.transforms.Compose] = None,
                 val_transforms: Optional[torchvision.transforms.Compose] = None,
                 validator: Optional[TimeSeriesForecastingInputValidator] = None,
                 lagged_value: Optional[List[int]] = None,
                 n_prediction_steps: int = 1,
                 dataset_name: Optional[str] = None,
                 normalize_y: bool = False,
                 ):
        # Preprocess time series data information
        assert X is not Y, "Training and Test data needs to belong two different object!!!"

        seasonality, freq, freq_value = self.compute_freq_values(freq, n_prediction_steps)
        self.seasonality = int(seasonality)

        self.freq: str = freq
        self.freq_value: Union[float, int] = freq_value

        self.n_prediction_steps = n_prediction_steps

        if dataset_name is None:
            dataset_name = str(uuid.uuid1(clock_seq=os.getpid()))
        self.dataset_name = dataset_name

        # Data Validation
        if validator is None:
            validator = TimeSeriesForecastingInputValidator(is_classification=False)
        self.validator: TimeSeriesForecastingInputValidator = validator

        if not isinstance(validator, TimeSeriesForecastingInputValidator):
            raise ValueError(f"This dataset only support TimeSeriesForecastingInputValidator "
                             f"but receive {type(validator)}")

        if not self.validator._is_fitted:
            self.validator.fit(X_train=X, y_train=Y, X_test=X_test, y_test=Y_test, series_idx=series_idx,
                               start_times=start_times)

        self.is_uni_variant = self.validator._is_uni_variant

        self.numerical_columns = self.validator.feature_validator.numerical_columns
        self.categorical_columns = self.validator.feature_validator.categorical_columns

        self.num_features: int = self.validator.feature_validator.num_features  # type: ignore[assignment]
        self.num_targets: int = self.validator.target_validator.out_dimensionality  # type: ignore[assignment]

        self.categories = self.validator.feature_validator.categories

        self.feature_shapes = self.validator.feature_shapes
        self.feature_names = tuple(self.validator.feature_names)

        assert self.validator.start_times is not None
        self.start_times = self.validator.start_times

        self.static_features = self.validator.feature_validator.static_features

        self._transform_time_features = False
        if not time_feature_transform:
            time_feature_transform = time_features_from_frequency_str(self.freq)
            if not time_feature_transform:
                # If time features are empty (as for yearly data), we add a
                # constant feature of 0
                time_feature_transform = [ConstantTransform()]

        self.time_feature_transform = time_feature_transform
        self.time_feature_names = tuple([f'time_feature_{t.__class__.__name__}' for t in self.time_feature_transform])

        # We also need to be able to transform the data, be it for pre-processing
        # or for augmentation
        self.train_transform = train_transforms
        self.val_transform = val_transforms

        # Construct time series sequences
        if known_future_features is None:
            known_future_features = tuple()  # type: ignore[assignment]
        known_future_features_index = extract_feature_index(self.feature_shapes,
                                                            self.feature_names,  # type: ignore[arg-type]
                                                            queried_features=known_future_features)  # type: ignore

        self.known_future_features = tuple(known_future_features)  # type: ignore[arg-type]

        # initialize datasets
        self.sequences_builder_kwargs = {"freq": self.freq,
                                         "time_feature_transform": self.time_feature_transform,
                                         "train_transforms": self.train_transform,
                                         "val_transforms": self.val_transform,
                                         "n_prediction_steps": n_prediction_steps,
                                         "sp": self.seasonality,
                                         "known_future_features_index": known_future_features_index}

        self.normalize_y = normalize_y

        training_sets = self.transform_data_into_time_series_sequence(X, Y,
                                                                      start_times=self.start_times,
                                                                      X_test=X_test,
                                                                      Y_test=Y_test, )
        sequence_datasets, train_tensors, test_tensors, sequence_lengths = training_sets
        Y: pd.DataFrame = train_tensors[1]  # type: ignore[no-redef]

        ConcatDataset.__init__(self, datasets=sequence_datasets)

        self.num_sequences = len(Y)
        self.sequence_lengths_train: np.ndarray = np.asarray(sequence_lengths) - n_prediction_steps

        self.seq_length_min = int(np.min(self.sequence_lengths_train))
        self.seq_length_median = int(np.median(self.sequence_lengths_train))
        self.seq_length_max = int(np.max(self.sequence_lengths_train))

        if int(freq_value) > self.seq_length_median:
            self.base_window_size = self.seq_length_median
        else:
            self.base_window_size = int(freq_value)

        self.train_tensors: Tuple[Optional[pd.DataFrame], pd.DataFrame] = train_tensors

        self.test_tensors: Optional[Tuple[Optional[pd.DataFrame], pd.DataFrame]] = test_tensors
        self.val_tensors = None

        self.issparse: bool = issparse(self.train_tensors[0])

        self.input_shape: Tuple[int, int] = (self.seq_length_min, self.num_features)  # type: ignore[assignment]

        # process known future features
        if known_future_features is None:
            future_feature_shapes: Tuple[int, int] = (self.seq_length_min, 0)
        else:
            future_feature_shapes = (self.seq_length_min, len(known_future_features))
        self.encoder_can_be_auto_regressive = (self.input_shape[-1] == future_feature_shapes[-1])

        if len(self.train_tensors) == 2 and self.train_tensors[1] is not None:
            self.output_type: str = type_of_target(self.train_tensors[1][0].fillna(method="pad"))

            if self.output_type in ["binary", "multiclass"]:
                # TODO in the future we also want forecasting classification task, we need to find a way to distinguish
                # TODO these tasks with the integral forecasting tasks!
                self.output_type = "continuous"

            if STRING_TO_OUTPUT_TYPES[self.output_type] in CLASSIFICATION_OUTPUTS:
                num_targets: int = len(np.unique(Y))
            else:
                num_targets = Y.shape[-1] if Y.ndim > 1 else 1  # type: ignore[union-attr]
            self.output_shape = [self.n_prediction_steps, num_targets]  # type: ignore
        else:
            raise ValueError('Forecasting dataset must contain target values!')

        # TODO: Look for a criteria to define small enough to preprocess
        self.is_small_preprocess = True

        # dataset split
        self.task_type: str = TASK_TYPES_TO_STRING[TIMESERIES_FORECASTING]

        self.numerical_features: List[int] = self.numerical_columns
        self.categorical_features: List[int] = self.categorical_columns

        self.random_state = np.random.RandomState(seed=seed)

        resampling_strategy_opt, resampling_strategy_args_opt = self.get_split_strategy(
            sequence_lengths=sequence_lengths,
            n_prediction_steps=n_prediction_steps,
            freq_value=self.freq_value,
            resampling_strategy=resampling_strategy,  # type: ignore[arg-type]
            resampling_strategy_args=resampling_strategy_args
        )

        self.resampling_strategy = resampling_strategy_opt   # type: ignore[assignment]
        self.resampling_strategy_args = resampling_strategy_args_opt

        if isinstance(self.resampling_strategy, CrossValTypes):
            self.cross_validators = CrossValFuncs.get_cross_validators(self.resampling_strategy)
        else:
            self.cross_validators = CrossValFuncs.get_cross_validators(CrossValTypes.time_series_cross_validation)
        if isinstance(self.resampling_strategy, HoldoutValTypes):
            self.holdout_validators = HoldOutFuncs.get_holdout_validators(self.resampling_strategy)

        else:
            self.holdout_validators = HoldOutFuncs.get_holdout_validators(
                HoldoutValTypes.time_series_hold_out_validation)

        self.splits = self.get_splits_from_resampling_strategy()  # type: ignore[assignment]

        valid_splits = []
        for i, split in enumerate(self.splits):
            if len(split[0]) > 0:
                valid_splits.append(split)

        if len(valid_splits) == 0:
            raise ValueError(f'The passed value for {n_prediction_steps} is unsuited for the current dataset, please '
                             'consider reducing n_prediction_steps')

        self.splits = valid_splits

        # TODO doing experiments to give the most proper way of defining these two values
        if lagged_value is None:
            try:
                lagged_value = [0] + get_lags_for_frequency(freq)
            except Exception:
                lagged_value = list(range(8))

        self.lagged_value = lagged_value

    @staticmethod
    def compute_freq_values(freq: Optional[Union[str, int, List[int]]],
                            n_prediction_steps: int) -> Tuple[Union[int, float], str, Union[int, float]]:
        """
        Compute frequency related values
        """
        if freq is None:
            freq = '1Y'

        if isinstance(freq, str):
            if freq not in SEASONALITY_MAP:
                Warning("The given freq name is not supported by our dataset, we will use the default "
                        "configuration space on the hyperparameter window_size, if you want to adapt this value"
                        "you could pass freq with a numerical value")
            freq_value = SEASONALITY_MAP.get(freq, 1)
        else:
            freq_value = freq
            freq = '1Y'

        seasonality = freq_value
        if isinstance(freq_value, list):
            min_base_size = min(n_prediction_steps, MAX_WINDOW_SIZE_BASE)
            if np.max(freq_value) < min_base_size:
                tmp_freq = max(freq_value)
            else:
                tmp_freq = min([freq_value_item for
                                freq_value_item in freq_value if freq_value_item >= min_base_size])
            freq_value = tmp_freq

        if isinstance(seasonality, list):
            seasonality = min(seasonality)  # Use to calculate MASE
        return seasonality, freq, freq_value  # type: ignore[return-value]

    @staticmethod
    def compute_time_features(start_times: List[pd.DatetimeIndex],
                              seq_lengths: List[int],
                              freq: Union[str, pd.DateOffset],
                              time_feature_transform: List[TimeFeature]) -> Dict[pd.DatetimeIndex, np.ndarray]:
        """
        compute the max series length for each start_time and compute their corresponding time_features. As lots of
        series in a dataset share the same start time, we could only compute the features for longest possible series
        and reuse them
        """
        series_lengths_max: Dict[pd.DatetimeIndex, int] = {}
        for start_t, seq_l in zip(start_times, seq_lengths):
            if start_t not in series_lengths_max or seq_l > series_lengths_max[start_t]:
                series_lengths_max[start_t] = seq_l
        series_time_features = {}
        for start_t, max_l in series_lengths_max.items():
            series_time_features[start_t] = compute_time_features(start_t, max_l, max_l, freq, time_feature_transform)
        return series_time_features

    def _get_dataset_indices(self, idx: int, only_dataset_idx: bool = False) -> Union[int, Tuple[int, int]]:
        """get which series the data point belongs to"""
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

    def __len__(self) -> int:
        return ConcatDataset.__len__(self)  # type: ignore[no-any-return]

    def __getitem__(self, idx: int,  # type: ignore[override]
                    train: bool = True) -> Tuple[Dict[str, torch.Tensor], Optional[Dict[str, torch.Tensor]]]:
        dataset_idx, sample_idx = self._get_dataset_indices(idx)  # type: ignore[misc]
        return self.datasets[dataset_idx].__getitem__(sample_idx, train)

    def get_validation_set(self, idx: int) -> TimeSeriesSequence:
        """generate validation series given the index. It ends at the position of the index"""
        dataset_idx, sample_idx = self._get_dataset_indices(idx)  # type: ignore[misc]
        return self.datasets[dataset_idx].get_val_seq_set(sample_idx)

    def get_time_series_seq(self, idx: int) -> TimeSeriesSequence:
        """get the series that the data point belongs to"""
        dataset_idx = self._get_dataset_indices(idx, True)
        return self.datasets[dataset_idx]  # type: ignore[index]

    def get_test_target(self, test_indices: np.ndarray) -> np.ndarray:
        """get the target data only. This function simply returns a np.array instead of a dictionary"""

        test_indices = np.where(test_indices < 0, test_indices + len(self), test_indices)
        y_test = np.ones([len(test_indices), self.n_prediction_steps, self.num_targets])
        y_test_argsort = np.argsort(test_indices)
        dataset_idx: int = self._get_dataset_indices(test_indices[y_test_argsort[0]],  # type: ignore[assignment]
                                                     only_dataset_idx=True)

        for y_i in y_test_argsort:
            test_idx = test_indices[y_i]
            while test_idx > self.cumulative_sizes[dataset_idx]:
                dataset_idx += 1
            if dataset_idx != 0:
                test_idx = test_idx - self.cumulative_sizes[dataset_idx - 1]
            y_test[y_i] = self.datasets[dataset_idx].get_test_target(test_idx)

        return y_test.reshape([-1, self.num_targets])

    def transform_data_into_time_series_sequence(self,
                                                 X: Optional[Union[np.ndarray, List[Union[pd.DataFrame, np.ndarray]]]],
                                                 Y: Union[np.ndarray, List[Union[pd.DataFrame, np.ndarray]]],
                                                 start_times: List[pd.DatetimeIndex],
                                                 X_test: Optional[
                                                     Union[np.ndarray, List[Union[pd.DataFrame, np.ndarray]]]] = None,
                                                 Y_test: Optional[
                                                     Union[np.ndarray, List[Union[pd.DataFrame, np.ndarray]]]] = None,
                                                 is_test_set: bool = False, ) -> Tuple[
        List[TimeSeriesSequence],
        Tuple[Optional[pd.DataFrame], pd.DataFrame],
        Optional[Tuple[Optional[pd.DataFrame], pd.DataFrame]],
        List[int]
    ]:
        """
        Transform the raw data into a list of TimeSeriesSequence that can be processed by AutoPyTorch Time Series
                build a series time sequence datasets

        Args:
            X: Optional[Union[np.ndarray, List[Union[pd.DataFrame, np.ndarray]]]]
                features, if is_test_set is True, then its length of
            Y: pd.DataFrame (N_all, N_target)
                flattened train target array with size N_all (the sum of all the series sequences) and number of targets
            start_times: List[pd.DatetimeIndex]
                start time of each training series
            X_test: Optional[np.ndarray (N_all_test, N_feature)]
                flattened test feature array with size N_all_test (the sum of all the series sequences) and N_feature,
                number of features
            Y_test: np.ndarray (N_all_test, N_target)
                flattened test target array with size N_all (the sum of all the series sequences) and number of targets
            is_test_set: Optional[List[pd.DatetimeIndex]]
                if the generated sequence used for test

        Returns:
            sequence_datasets : List[TimeSeriesSequence]
                a list of datasets
            train_tensors: Tuple[Optional[pd.DataFrame], pd.DataFrame]
                training tensors
            test_tensors: Optional[Tuple[Optional[pd.DataFrame], pd.DataFrame]]
                test tensors

        """
        dataset_with_future_features = X is not None and len(self.known_future_features) > 0
        X, Y, sequence_lengths = self.validator.transform(X, Y)
        time_features = self.compute_time_features(start_times,
                                                   sequence_lengths,
                                                   self.freq,
                                                   self.time_feature_transform)

        if Y_test is not None or X_test is not None:
            X_test, Y_test, _ = self.validator.transform(X_test, Y_test,
                                                         validate_for_future_features=dataset_with_future_features)

        y_groups: pd.DataFrameGroupBy = Y.groupby(Y.index)  # type: ignore[union-attr]
        if self.normalize_y:
            mean = y_groups.agg("mean")
            std = y_groups.agg("std")
            std[std == 0] = 1.
            std.fillna(1.)
            Y = (Y - mean) / std
            self.y_mean = mean
            self.y_std = std
            if Y_test is not None:
                Y_test = (Y_test[mean.columns] - mean) / std

        sequence_datasets, train_tensors, test_tensors = self.make_sequences_datasets(
            X=X, Y=Y,
            X_test=X_test, Y_test=Y_test,
            start_times=start_times,
            time_features=time_features,
            is_test_set=is_test_set,
            **self.sequences_builder_kwargs)
        return sequence_datasets, train_tensors, test_tensors, sequence_lengths

    @staticmethod
    def make_sequences_datasets(X: Optional[pd.DataFrame],
                                Y: pd.DataFrame,
                                start_times: List[pd.DatetimeIndex],
                                time_features: Optional[Dict[pd.DatetimeIndex, np.ndarray]] = None,
                                X_test: Optional[pd.DataFrame] = None,
                                Y_test: Optional[pd.DataFrame] = None,
                                is_test_set: bool = False,
                                **sequences_kwargs: Any) -> Tuple[
        List[TimeSeriesSequence],
        Tuple[Optional[pd.DataFrame], pd.DataFrame],
        Optional[Tuple[Optional[pd.DataFrame], pd.DataFrame]]
    ]:
        """
        build a series time sequence datasets

        Args:
            X: pd.DataFrame (N_all, N_feature)
                flattened train feature DataFrame with size N_all (the sum of all the series sequences) and N_feature,
                number of features, X's index should contain the information identifying its series number
            Y: pd.DataFrame (N_all, N_target)
                flattened train target array with size N_all (the sum of all the series sequences) and number of targets
            start_times: List[pd.DatetimeIndex]
                start time of each training series
            time_features: Dict[pd.Timestamp, np.ndarray]:
                time features for each possible start training times
            X_test: Optional[np.ndarray (N_all_test, N_feature)]
                flattened test feature array with size N_all_test (the sum of all the series sequences) and N_feature,
                number of features
            Y_test: np.ndarray (N_all_test, N_target)
                flattened test target array with size N_all (the sum of all the series sequences) and number of targets
            is_test_set (bool):
                if the generated sequence used for test
            sequences_kwargs: Dict
                additional arguments for test sets

        Returns:
            sequence_datasets : List[TimeSeriesSequence]
                a list of datasets
            train_tensors: Tuple[pd.DataFrame, pd.DataFrame]
                training tensors
            train_tensors: Optional[Tuple[pd.DataFrame, pd.DataFrame]]
                training tensors

        """
        sequence_datasets = []

        y_group = Y.groupby(Y.index)
        if X is not None:
            x_group = X.groupby(X.index)
        if Y_test is not None:
            y_test_group = Y_test.groupby(Y_test.index)

        if X_test is not None:
            x_test_group = X_test.groupby(X_test.index)

        for i_ser, (start_time, y) in enumerate(zip(start_times, y_group)):
            ser_id = y[0]
            y_ser = y[1].transform(np.array).values
            x_ser = x_group.get_group(ser_id).transform(np.array).values if X is not None else None

            y_test_ser = y_test_group.get_group(ser_id).transform(np.array).values if Y_test is not None else None
            x_test_ser = x_test_group.get_group(ser_id).transform(np.array).values if X_test is not None else None

            sequence = TimeSeriesSequence(
                X=x_ser,
                Y=y_ser,
                start_time=start_time,
                X_test=x_test_ser,
                Y_test=y_test_ser,
                time_features=time_features[start_time][:len(y_ser)] if time_features is not None else None,
                is_test_set=is_test_set,
                **sequences_kwargs)
            sequence_datasets.append(sequence)

        train_tensors = (X, Y)
        # we could guarantee that Y_test has shape [len(seq) * n_prediction_steps, num_targets]
        test_tensors = (X_test, Y_test.values) if Y_test is not None else None

        return sequence_datasets, train_tensors, test_tensors

    def replace_data(self,
                     X_train: pd.DataFrame,
                     X_test: Optional[pd.DataFrame],
                     known_future_features_index: Optional[Tuple[int]] = None) -> 'BaseDataset':
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
            seq.known_future_features_index = known_future_features_index
            seq.is_pre_processed = True

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
            transform (torchvision.transforms.Compose):
                The transformations proposed by the current pipeline
            train (bool):
                Whether to update the train or validation transform

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
    def transform_time_features(self) -> bool:
        return self._transform_time_features

    @transform_time_features.setter
    def transform_time_features(self, value: bool) -> None:
        self._transform_time_features = value
        for seq in self.datasets:
            seq.transform_time_features = value

    def get_splits_from_resampling_strategy(self) -> List[Tuple[List[int], Optional[List[int]]]]:
        """
        Creates a set of splits based on a resampling strategy provided, here each item in test_split represent
        n_prediction_steps element in the dataset. (The start of timestep that we want to predict)

        Returns
            ( List[Tuple[List[int], Optional[List[int]]]]):
            splits in the [train_indices, val_indices] format
        """
        splits = []
        if isinstance(self.resampling_strategy, HoldoutValTypes):
            val_share = DEFAULT_RESAMPLING_PARAMETERS[self.resampling_strategy].get(
                'val_share', None)
            if self.resampling_strategy_args is not None:
                val_share = self.resampling_strategy_args.get('val_share', val_share)
                n_repeats = self.resampling_strategy_args.get("n_repeats", 1)
            else:
                n_repeats = 1
            splits.append(self.create_holdout_val_split(holdout_val_type=self.resampling_strategy,
                                                        val_share=val_share,
                                                        n_repeats=n_repeats))

        elif isinstance(self.resampling_strategy, CrossValTypes):
            num_splits = DEFAULT_RESAMPLING_PARAMETERS[self.resampling_strategy].get(
                'num_splits', None)
            if self.resampling_strategy_args is not None:
                num_splits = self.resampling_strategy_args.get('num_splits', num_splits)
                n_repeats = self.resampling_strategy_args.get("n_repeats", 1)
            else:
                n_repeats = 1
            # Create the split if it was not created before
            splits.extend(self.create_cross_val_splits(  # type: ignore[arg-type]
                cross_val_type=self.resampling_strategy,
                num_splits=cast(int, num_splits),
                n_repeats=n_repeats
            ))
        elif isinstance(self.resampling_strategy, NoResamplingStrategyTypes):
            splits.append(self.create_refit_split())
        else:
            raise ValueError(f"Unsupported resampling strategy={self.resampling_strategy}")
        return splits  # type: ignore[return-value]

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
                                   'sp': self.seasonality,  # For metric computation
                                   'input_shape': self.input_shape,
                                   'time_feature_transform': self.time_feature_transform,
                                   'uni_variant': self.is_uni_variant,
                                   'static_features_shape': len(self.static_features),
                                   'future_feature_shapes': (self.n_prediction_steps, len(self.known_future_features)),
                                   'targets_have_missing_values': self.train_tensors[1].isnull().values.any(),
                                   'encoder_can_be_auto_regressive': self.encoder_can_be_auto_regressive,
                                   'features_have_missing_values': False if self.train_tensors[0] is None
                                   else self.train_tensors[0].isnull().values.any()})
        return dataset_properties

    @staticmethod
    def get_split_strategy(sequence_lengths: List[int],
                           n_prediction_steps: int,
                           freq_value: Union[float, int],
                           resampling_strategy: ResamplingStrategies = HoldoutValTypes.time_series_hold_out_validation,
                           resampling_strategy_args: Optional[Dict[str, Any]] = None, ) -> \
            Tuple[ResamplingStrategies, Optional[Dict[str, Any]]]:
        """
        Determines the most possible sampling strategy for the datasets: the lengths of each sequence might not be long
        enough to support cross-validation split, thus we need to carefully compute the number of folds. Additionally,
        each fold might contain multiple forecasting instances (each with length n_prediction_steps and there is no
        overlapping between the test instances). This value is considered as 'n_repeats'

        Args:
            sequence_lengths (List[int]):
                lengths of each sequence
            n_prediction_steps (int):
                forecasting horizon
            freq_value (Union[float, int]):
                period of the dataset, determined by its sampling frequency
            resampling_strategy(ResamplingStrategies):
                resampling strategy to be checked
            resampling_strategy_args (Optional[Dict[str, Any]]):
                resampling strategy arguments to be checked

        Returns:
            resampling_strategy(ResamplingStrategies):
                resampling strategy
            resampling_strategy_args (Optional[Dict[str, Any]]):
                resampling strategy arguments
        """
        # check if dataset could be split with cross validation
        minimal_seq_length = np.min(sequence_lengths) - n_prediction_steps
        if isinstance(resampling_strategy, CrossValTypes):
            num_splits = DEFAULT_RESAMPLING_PARAMETERS[resampling_strategy].get(
                'num_splits', 5)
            if resampling_strategy_args is not None:
                num_splits = resampling_strategy_args.get('num_splits', num_splits)

            # Check if all the series can be properly split, if not, we reduce the number of split
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
                    np.round((n_prediction_steps // int(freq_value) + 1) * freq_value)
                )

                while minimal_seq_length < (num_splits - 1) * seasonality_h_value:
                    if num_splits <= 2:
                        break
                    num_splits -= 1
                if resampling_strategy_args is None:
                    resampling_strategy_args = {'num_splits': num_splits}
                else:
                    resampling_strategy_args.update({'num_splits': num_splits})

        num_seqs = len(sequence_lengths)

        if resampling_strategy_args is not None and "n_repeats" in resampling_strategy_args:
            n_repeats = resampling_strategy_args["n_repeats"]
        else:
            # we want to keep the amount of forecasting instances large enough to generalize well or make full use of
            # the information from the training set
            # if there are not enough series in the dataset or the minimal length of the sequence is large enough
            # to support multiple predictions
            if (num_seqs < 100 and minimal_seq_length > 10 * n_prediction_steps) or \
                    minimal_seq_length > 50 * n_prediction_steps:
                if num_seqs < 100:
                    n_repeats = int(np.ceil(100.0 / num_seqs))
                else:
                    n_repeats = int(np.round(minimal_seq_length / (50 * n_prediction_steps)))
            else:
                n_repeats = 1

        if resampling_strategy == CrossValTypes.time_series_cross_validation:
            n_repeats = min(n_repeats, minimal_seq_length // (5 * n_prediction_steps * num_splits))
        elif resampling_strategy == CrossValTypes.time_series_ts_cross_validation:
            seasonality_h_value = int(np.round(
                (n_prediction_steps // int(freq_value) + 1) * freq_value)
            )
            while minimal_seq_length // 5 < (num_splits - 1) * n_repeats * seasonality_h_value - n_prediction_steps:
                n_repeats -= 1

        elif resampling_strategy == HoldoutValTypes.time_series_hold_out_validation:
            n_repeats = min(n_repeats, minimal_seq_length // (5 * n_prediction_steps) - 1)

        else:
            n_repeats = 1

        n_repeats = max(n_repeats, 1)

        if resampling_strategy_args is None:
            resampling_strategy_args = {'n_repeats': n_repeats}
        else:
            resampling_strategy_args.update({'n_repeats': n_repeats})
        return resampling_strategy, resampling_strategy_args

    def create_cross_val_splits(
            self,
            cross_val_type: CrossValTypes,
            num_splits: int,
            n_repeats: int = 1,
    ) -> List[Tuple[Union[List[int], np.ndarray], Union[List[int], np.ndarray]]]:
        """
        This function creates the cross validation split for the given task.

        It is done once per dataset to have comparable results among pipelines

        Args:
            cross_val_type (CrossValTypes):
                cross validation type
            num_splits (int):
                number of splits to be created
            n_repeats (int):
                how many n_prediction_steps to repeat in the validation set

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
            kwargs.update({'seasonality_h_value': seasonality_h_value})
        kwargs["n_repeats"] = n_repeats

        splits: List[List[Tuple]] = [[() for _ in range(len(self.datasets))] for _ in range(num_splits)]

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
            split = splits[i]  # type: ignore[assignment]
            train_indices = np.hstack([sp[0] for sp in split])
            test_indices = np.hstack([sp[1] for sp in split])
            splits_merged.append((train_indices, test_indices))
        return splits_merged  # type: ignore[return-value]

    def create_holdout_val_split(
            self,
            holdout_val_type: HoldoutValTypes,
            val_share: float,
            n_repeats: int = 1,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        This function creates the holdout split for the given task.

        It is done once per dataset to have comparable results among pipelines

        Args:
            holdout_val_type (HoldoutValTypes):
                holdout type
            val_share (float):
                share of the validation data
            n_repeats (int):
                how many n_prediction_steps to repeat in the validation set

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
                  "n_repeats": n_repeats}

        splits = [[() for _ in range(len(self.datasets))] for _ in range(2)]
        idx_start = 0
        for idx_seq, dataset in enumerate(self.datasets):
            split = self.holdout_validators[holdout_val_type.name](self.random_state,
                                                                   val_share,
                                                                   indices=np.arange(len(dataset)) + idx_start,
                                                                   **kwargs)
            for idx_split in range(2):
                splits[idx_split][idx_seq] = split[idx_split]  # type: ignore[call-overload]
            idx_start += self.sequence_lengths_train[idx_seq]

        train_indices = np.hstack([sp for sp in splits[0]])
        test_indices = np.hstack([sp for sp in splits[1]])

        return train_indices, test_indices

    def create_refit_split(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        This function creates the refit split for the given task. All the data in the dataset will be considered as
        training sets

        Returns:
            (Tuple[np.ndarray, np.ndarray]): Tuple containing (train_indices, val_indices)
        """
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
        """create a refit set that allows the network to be trained with the entire training-validation sets"""
        refit_set: TimeSeriesForecastingDataset = copy.deepcopy(self)
        refit_set.resampling_strategy = NoResamplingStrategyTypes.no_resampling
        refit_set.splits = refit_set.get_splits_from_resampling_strategy()
        return refit_set

    def generate_test_seqs(self) -> List[TimeSeriesSequence]:
        """
        A function that generate a set of test series from the information available at this dataset. By calling this
        function, we could make use of the cached information such as time features to accelerate the computation time

        Returns:
            test_sets(List[TimeSeriesSequence])
                generated test sets
        """
        test_sets = copy.deepcopy(self.datasets)
        for test_seq in test_sets:
            test_seq.is_test_set = True
        return test_sets
