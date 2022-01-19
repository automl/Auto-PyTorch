import os
from typing import Any, Dict, List, Optional, Tuple, Union, cast
import uuid
import bisect
import copy
import warnings

import numpy as np

import pandas as pd
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
from autoPyTorch.utils.forecasting_time_features import FREQUENCY_MAP


from autoPyTorch.data.time_series_forecasting_validator import TimeSeriesForecastingInputValidator
from autoPyTorch.pipeline.components.preprocessing.time_series_preprocessing.TimeSeriesTransformer import \
    TimeSeriesTransformer
from autoPyTorch.utils.common import FitRequirement
from autoPyTorch.constants_forecasting import SEASONALITY_MAP
from autoPyTorch.pipeline.components.training.metrics.metrics import compute_mase_coefficient


TIME_SERIES_FORECASTING_INPUT = Tuple[np.ndarray, np.ndarray]  # currently only numpy arrays are supported
TIME_SERIES_REGRESSION_INPUT = Tuple[np.ndarray, np.ndarray]
TIME_SERIES_CLASSIFICATION_INPUT = Tuple[np.ndarray, np.ndarray]


class TimeSeriesSequence(Dataset):
    def __init__(self,
                 X: Union[np.ndarray, pd.DataFrame],
                 Y: Union[np.ndarray, pd.Series],
                 X_test: Optional[Union[np.ndarray, pd.DataFrame]] = None,
                 Y_test: Optional[Union[np.ndarray, pd.DataFrame]] = None,
                 train_transforms: Optional[torchvision.transforms.Compose] = None,
                 val_transforms: Optional[torchvision.transforms.Compose] = None,
                 n_prediction_steps: int = 1,
                 sp: int = 1,
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

        self.X_val = None
        self.Y_val = None

        self.X_test = X_test
        self.Y_tet = Y_test

        # We also need to be able to transform the data, be it for pre-processing
        # or for augmentation
        self.train_transform = train_transforms
        self.val_transform = val_transforms
        self.sp = sp

        self.mase_coefficient = compute_mase_coefficient(self.X, sp=self.sp, n_prediction_steps=n_prediction_steps)

    def __getitem__(self, index: int, train: bool = True) \
            -> Tuple[Dict[str, torch.Tensor], Optional[Dict[str, torch.Tensor]]]:
        """
        get a subsequent of time series data, unlike vanilla tabular dataset, we obtain all the previous sequences
        until the given index, this allows us to do further transformation when the

        Args:
            index (int): what element to yield from all the train/test tensors
            train (bool): Whether to apply a train or test transformation, if any

        Returns:
            features from past, targets from past and future
        """
        if index < 0:
            index = self.__len__() + index

        if hasattr(self.X, 'loc'):
            X = self.X.iloc[:index + 1]
        else:
            X = self.X[:index + 1]

        if self.train_transform is not None and train:
            X = self.train_transform(X)
        elif self.val_transform is not None and not train:
            X = self.val_transform(X)

        # In case of prediction, the targets are not provided
        Y = self.Y
        if Y is not None:
            # Y = Y[:index + self.n_prediction_steps]
            # Y = Y[index + 1: index + self.n_prediction_steps + 1]
            Y_future = Y[index + 1: index + self.n_prediction_steps + 1]

            Y_future = torch.from_numpy(Y_future)
            # Y_Past does not need to be fed to the network, we keep it as np array
        else:
            Y_future = None

        return {"past_target": torch.from_numpy(X),
                "mase_coefficient": self.mase_coefficient}, Y_future

    def __len__(self) -> int:
        return self.X.shape[0]

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
        if index < 0:
            index = self.__len__() + index
        if index == self.__len__() - 1:
            return copy.copy(self)
        else:
            return TimeSeriesSequence(self.X[:index + 1],
                                      self.Y[:index + 1 + self.n_prediction_steps],
                                      train_transforms=self.train_transform,
                                      val_transforms=self.val_transform,
                                      n_prediction_steps=self.n_prediction_steps,
                                      sp=self.sp)




class TimeSeriesForecastingDataset(BaseDataset, ConcatDataset):
    datasets: List[TimeSeriesSequence]
    cumulative_sizes: List[int]

    def __init__(self,
                 X: Union[np.ndarray, List[List]],
                 Y: Union[np.ndarray, pd.Series],
                 X_test: Optional[Union[np.ndarray, pd.DataFrame]] = None,
                 Y_test: Optional[Union[np.ndarray, pd.DataFrame]] = None,
                 target_variables: Optional[Union[Tuple[int], int]] = None,
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
                 shift_input_data: bool = True,
                 normalize_y: bool = True,
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
        if isinstance(freq, list):
            tmp_freq = min([freq_value for freq_value in freq if freq_value >= n_prediction_steps])
            freq_value = tmp_freq
        if isinstance(freq_value, list):
            tmp_freq = min([freq_value_item for freq_value_item in freq_value if freq_value_item >= n_prediction_steps])
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
            self.validator.fit(X_train=X, y_train=Y, X_test=X_test, y_test=Y_test, )

        self.numerical_columns = self.validator.feature_validator.numerical_columns
        self.categorical_columns = self.validator.feature_validator.categorical_columns

        self.num_features = self.validator.feature_validator.num_features  # type: int
        self.num_target = self.validator.target_validator.out_dimensionality  # type: int

        self.categories = self.validator.feature_validator.categories

        self.shift_input_data = shift_input_data
        self.target_variables = target_variables
        if target_variables is None:
            if self.num_target != 1:
                raise ValueError("target_variables must be specified if more the input has more than one feature value")
            self.target_columns = (0, ) # to keep the output dimension unchanged
        elif isinstance(target_variables, int):
            self.target_columns = (target_variables, )
        else:
            self.target_columns = target_variables

        X, sequence_lengths, Y = self.validator.transform(X, Y,
                                                          shift_input_data=shift_input_data,
                                                          n_prediction_steps=n_prediction_steps)
        if X_test is not None:
            X_test, self.sequence_lengths_tests, Y_test = self.validator.transform(X_test, Y_test,
                                                                                   shift_input_data=shift_input_data,
                                                                                   n_prediction_steps=n_prediction_steps
                                                                                   )
        else:
            self.sequence_lengths_tests = None

        self.shuffle = shuffle
        self.random_state = np.random.RandomState(seed=seed)

        minimal_seq_length = np.min(sequence_lengths)
        if isinstance(resampling_strategy, CrossValTypes):
            num_splits = DEFAULT_RESAMPLING_PARAMETERS[resampling_strategy].get(
                'num_splits', None)
            if resampling_strategy_args is not None:
                num_splits = resampling_strategy_args.get('num_split', num_splits)
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

        self.resampling_strategy = resampling_strategy
        self.resampling_strategy_args = resampling_strategy_args

        # We also need to be able to transform the data, be it for pre-processing
        # or for augmentation
        self.train_transform = train_transforms
        self.val_transform = val_transforms

        self.num_sequences = len(X)
        self.sequence_lengths_train = sequence_lengths

        # initialize datasets
        sequences_kwargs = {"train_transforms": self.train_transform,
                            "val_transforms": self.val_transform,
                            "n_prediction_steps": n_prediction_steps,
                            "sp": self.seasonality}

        self.y_train_mean = [0] * len(self.sequence_lengths_train)
        self.y_train_std = [1] * len(self.sequence_lengths_train)

        sequence_datasets, train_tensors, test_tensors = self.make_sequences_datasets(X=X, Y=Y,
                                                                                      X_test=X_test, Y_test=Y_test,
                                                                                      normalize_y=normalize_y,
                                                                                      **sequences_kwargs)

        self.normalize_y = normalize_y

        ConcatDataset.__init__(self, datasets=sequence_datasets)

        self.seq_length_min = np.min(self.sequence_lengths_train)

        self.train_tensors = train_tensors

        self.test_tensors = test_tensors
        self.val_tensors = None

        self.task_type: Optional[str] = None
        self.issparse: bool = issparse(self.train_tensors[0])
        # TODO find a way to edit input shape!
        self.input_shape: Tuple[int] = (self.seq_length_min, self.num_features)

        if len(self.train_tensors) == 2 and self.train_tensors[1] is not None:
            self.output_type: str = type_of_target(self.train_tensors[1][0])

            if self.output_type in ["binary", "multiclass"]:
                self.output_type = "continuous"

            if STRING_TO_OUTPUT_TYPES[self.output_type] in CLASSIFICATION_OUTPUTS:
                num_target = len(np.unique(Y))
                # self.output_shape = len(np.unique(Y))
            else:
                # self.output_shape = self.train_tensors[1].shape[-1] if self.train_tensors[1].ndim > 1 else 1
                num_target = X.shape[-1] if X.ndim > 1 else 1
            self.output_shape = [self.n_prediction_steps, num_target]

        # TODO: Look for a criteria to define small enough to preprocess
        self.is_small_preprocess = True

        self.task_type = TASK_TYPES_TO_STRING[TIMESERIES_FORECASTING]

        self.numerical_features: List[int] = list(range(self.num_features))
        self.categorical_features: List[int] = []
        self.cross_validators = CrossValFuncs.get_cross_validators(CrossValTypes.time_series_cross_validation)
        self.holdout_validators = HoldOutFuncs.get_holdout_validators(HoldoutValTypes.time_series_hold_out_validation)

        self.splits = self.get_splits_from_resampling_strategy()

        # TODO doing experiments to give the most proper way of defining these two values
        if lagged_value is None:
            freq = FREQUENCY_MAP[self.freq]
            lagged_value = [0] + get_lags_for_frequency(freq)

        self.lagged_value = lagged_value

    def __getitem__(self, idx, train=True):
        if idx < 0:
            if -idx > len(self):
                raise ValueError("absolute value of index should not exceed dataset length")
            idx = len(self) + idx
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        return self.datasets[dataset_idx].__getitem__(sample_idx, train)

    def get_validation_set(self, idx):
        if idx < 0:
            if -idx > len(self):
                raise ValueError("absolute value of index should not exceed dataset length")
            idx = len(self) + idx
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        return self.datasets[dataset_idx].get_val_seq_set(sample_idx)

    def get_time_series_seq(self, idx) -> TimeSeriesSequence:
        if idx < 0:
            if -idx > len(self):
                raise ValueError("absolute value of index should not exceed dataset length")
            idx = len(self) + idx
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        return self.datasets[dataset_idx]

    def make_sequences_datasets(self,
                                X: np.ndarray,
                                Y: np.ndarray,
                                X_test: Optional[np.ndarray] = None,
                                Y_test: Optional[np.ndarray] = None,
                                normalize_y: bool = True,
                                **sequences_kwargs: Optional[Dict]) -> \
            Tuple[List[TimeSeriesSequence], Tuple[List, List], Tuple[List, List]]:
        """
        build a series time seequences datasets
        Args:
            X: np.ndarray (N_all, N_feature)
                flattened train feature array with size N_all (the sum of all the series sequences) and N_feature,
                number of features
            Y: np.ndarray (N_all, N_target)
                flattened train target array with size N_all (the sum of all the series sequences) and number of targets
            sequence_lengths_train: List[int]
                a list containing all the sequences length in the training set
            X_test: Optional[np.ndarray (N_all_test, N_feature)]
                flattened test feature array with size N_all_test (the sum of all the series sequences) and N_feature,
                number of features
            Y_test: np.ndarray (N_all_test, N_target)
                flattened test target array with size N_all (the sum of all the series sequences) and number of targets
            sequence_lengths_test: Optional[List[int]]
                a list containing all the sequences length in the test set
            normalize_y: bool
                if we want to normalize target vaues (normalization is conducted w.r.t. each sequence)
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
        idx_start_train = 0
        idx_start_test = 0

        for seq_idx, seq_length_train in enumerate(self.sequence_lengths_train):
            idx_end_train = idx_start_train + seq_length_train

            X_seq = X[idx_start_train: idx_end_train]
            if self.shift_input_data:
                Y_seq = Y[idx_start_train + seq_idx * self.n_prediction_steps:
                          idx_end_train + (1 + seq_idx) * self.n_prediction_steps]
            else:
                Y_seq = Y[idx_start_train: idx_end_train]

            if normalize_y:
                Y_seq_mean = np.mean(Y_seq)
                Y_seq_std = np.std(Y_seq)
                Y_seq = (Y_seq - Y_seq_mean) / Y_seq_std

            if self.shift_input_data:
                Y[idx_start_train + seq_idx * self.n_prediction_steps:
                  idx_end_train + (1 + seq_idx) * self.n_prediction_steps] = Y_seq
            else:
                Y[idx_start_train: idx_end_train] = Y_seq

            if X_test is not None and Y_test is not None:
                seq_length_test = self.sequence_lengths_tests[seq_idx]
                idx_end_test = idx_start_test + seq_length_test

                X_test_seq = X_test[idx_start_test: idx_end_test]
                if self.shift_input_data:
                    Y_test_seq = Y[idx_start_test + seq_idx * self.n_prediction_steps:
                                   idx_end_test + (1 + seq_idx) * self.n_prediction_steps]
                else:
                    Y_test_seq = Y_test[idx_start_test: idx_end_test]

                if normalize_y:
                    Y_test_seq_mean = np.mean(Y_test_seq)
                    Y_test_seq_std = np.std(Y_test_seq)
                    Y_seq = (Y_seq - Y_test_seq_mean) / Y_test_seq_std

                if self.shift_input_data:
                    Y_test[idx_start_test + seq_idx * self.n_prediction_steps:
                           idx_end_test + (1 + seq_idx) * self.n_prediction_steps] = Y_seq
                else:
                    Y_test[idx_start_test: idx_end_test] = Y_seq

            else:
                X_test_seq = None
                Y_test_seq = None

            sequence = TimeSeriesSequence(X=X_seq,
                                          Y=Y_seq,
                                          X_test=X_test_seq,
                                          Y_test=Y_test_seq,
                                          **sequences_kwargs)
            sequence_datasets.append(sequence)
            idx_start_train = idx_end_train

            # self.sequence_lengths_train[seq_idx] = len(sequence)

            # X_seq_all.append(X_seq)
            # Y_seq_all.append(Y_seq)

            # X_test_seq_all.append(X_test_seq)
            # Y_test_seq_all.append(Y_test_seq)
        # train_tensors = (X_seq_all, Y_seq_all)
        train_tensors = (X, Y)
        if X_test is None or Y_test is None:
            test_tensors = None
        else:
            # test_tensors = (X_test_seq_all, Y_test_seq_all)
            test_tensors = (X_test, Y_test)

        return sequence_datasets, train_tensors, test_tensors

    def replace_data(self, X_train: BaseDatasetInputType, X_test: Optional[BaseDatasetInputType]) -> 'BaseDataset':
        super(TimeSeriesForecastingDataset, self).replace_data(X_train=X_train, X_test=X_test)
        self.update_tensros_seqs(X_train, self.sequence_lengths_train, is_train=True)
        if X_test is not None:
            self.update_tensros_seqs(X_test, self.sequence_lengths_tests, is_train=False)
        return self

    def update_tensros_seqs(self, X, sequence_lengths, is_train=True):
        idx_start = 0
        if is_train:
            for seq, seq_length in zip(self.datasets, sequence_lengths):
                idx_end = idx_start + seq_length
                seq.X = X[idx_start: idx_end]
                idx_start = idx_end
        else:
            for seq, seq_length in zip(self.datasets, sequence_lengths):
                idx_end = idx_start + seq_length
                seq.X_test = X[idx_start: idx_end]
                idx_start = idx_end

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
            splits.append(self.create_holdout_val_split(holdout_val_type=self.resampling_strategy,
                                                        val_share=val_share))

            if self.val_tensors is not None:
                upper_window_size = np.min(self.sequence_lengths_train) - self.n_prediction_steps
            else:
                upper_window_size = int(np.min(self.sequence_lengths_train) * 1 - val_share)

        elif isinstance(self.resampling_strategy, CrossValTypes):
            num_splits = DEFAULT_RESAMPLING_PARAMETERS[self.resampling_strategy].get(
                'num_splits', None)
            if self.resampling_strategy_args is not None:
                num_splits = self.resampling_strategy_args.get('num_splits', num_splits)
            # Create the split if it was not created before
            splits.extend(self.create_cross_val_splits(
                cross_val_type=self.resampling_strategy,
                num_splits=cast(int, num_splits),
            ))
            upper_window_size = (np.min(self.sequence_lengths_train)) - self.n_prediction_steps * (num_splits - 1)
        elif self.resampling_strategy is None:
            splits.append(self.create_refit_split())
            upper_window_size = np.inf
        else:
            raise ValueError(f"Unsupported resampling strategy={self.resampling_strategy}")
        self.upper_window_size = upper_window_size
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
            'target_columns': self.target_columns
        })
        return info

    def get_dataset_properties(self, dataset_requirements: List[FitRequirement]) -> Dict[str, Any]:
        dataset_properties = super().get_dataset_properties(dataset_requirements=dataset_requirements)
        dataset_properties.update({'n_prediction_steps': self.n_prediction_steps,
                                   'upper_window_size': self.upper_window_size,
                                   'sp': self.seasonality,  # For metric computation,
                                   'freq': self.freq,
                                   'sequence_lengths_train': self.sequence_lengths_train,
                                   'lagged_value': self.lagged_value})
        return dataset_properties

    def create_cross_val_splits(
            self,
            cross_val_type: CrossValTypes,
            num_splits: int
    ) -> List[Tuple[Union[List[int], np.ndarray], Union[List[int], np.ndarray]]]:
        """
        This function creates the cross validation split for the given task.

        It is done once per dataset to have comparable results among pipelines
        Args:
            cross_val_type (CrossValTypes):
            num_splits (int): number of splits to be created

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
        splits = [[() for _ in range(len(self.datasets))] for _ in range(num_splits)]
        idx_all = self._get_indices()

        for idx_seq, dataset in enumerate(self.datasets):
            if self.shift_input_data:
                split = self.cross_validators[cross_val_type.name](self.random_state,
                                                                   num_splits,
                                                                   indices=idx_start + np.arange(len(dataset)), **kwargs)
            else:
                # If the data is not shifted, we need to discard the last n_prediction_steps such that we have enough
                # y values
                split = self.cross_validators[cross_val_type.name](self.random_state,
                                                                   num_splits,
                                                                   indices=idx_start + np.arange(
                                                                       len(dataset) - self.n_prediction_steps),
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
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        This function creates the holdout split for the given task.

        It is done once per dataset to have comparable results among pipelines
        Args:
            holdout_val_type (HoldoutValTypes):
            val_share (float): share of the validation data

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
        kwargs = {"n_prediction_steps": self.n_prediction_steps}

        splits = [[() for _ in range(len(self.datasets))] for _ in range(2)]
        idx_start = 0
        for idx_seq, dataset in enumerate(self.datasets):
            if self.shift_input_data:
                split = self.holdout_validators[holdout_val_type.name](self.random_state,
                                                                       val_share,
                                                                       indices=np.arange(len(dataset)),
                                                                       **kwargs)
            else:
                split = self.holdout_validators[holdout_val_type.name](self.random_state,
                                                                       val_share,
                                                                       indices=np.arange(
                                                                           len(dataset) - self.n_prediction_steps),
                                                                       **kwargs)

            for idx_split in range(2):
                splits[idx_split][idx_seq] = idx_start + split[idx_split]
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
            if self.shift_input_data:
                split = [np.arange(len(dataset)), np.array([len(dataset) - 1])]
            else:
                last_idx = len(dataset) - self.n_prediction_steps -1
                split = [np.arange(len(dataset) - self.n_prediction_steps), np.array([last_idx])]

            for idx_split in range(2):
                splits[idx_split][idx_seq] = idx_start + split[idx_split]
            idx_start += self.sequence_lengths_train[idx_seq]

        train_indices = np.hstack([sp for sp in splits[0]])
        test_indices = np.hstack([sp for sp in splits[1]])

        return train_indices, test_indices


def _check_time_series_forecasting_inputs(train: np.ndarray,
                                          val: Optional[np.ndarray] = None) -> None:
    if train.ndim != 3 or any(isinstance(i, (list, np.ndarray)) for i in train):
        raise ValueError(
            "The training data for time series forecasting has to be a three-dimensional tensor of shape PxLxM. or a"
            "nested list")
    if val is not None:
        if val.ndim != 3 or any(isinstance(i, (list, np.ndarray)) for i in val):
            raise ValueError(
                "The validation data for time series forecasting "
                "has to be a three-dimensional tensor of shape PxLxM or a nested list.")


class TimeSeriesDataset(BaseDataset):
    """
    Common dataset for time series classification and regression data
    Args:
        X (np.ndarray): input training data.
        Y (Union[np.ndarray, pd.Series]): training data targets.
        X_test (Optional[np.ndarray]):  input testing data.
        Y_test (Optional[Union[np.ndarray, pd.DataFrame]]): testing data targets
        resampling_strategy (Union[CrossValTypes, HoldoutValTypes]),
            (default=HoldoutValTypes.holdout_validation):
            strategy to split the training data.
        resampling_strategy_args (Optional[Dict[str, Any]]): arguments
            required for the chosen resampling strategy. If None, uses
            the default values provided in DEFAULT_RESAMPLING_PARAMETERS
            in ```datasets/resampling_strategy.py```.
        shuffle:  Whether to shuffle the data before performing splits
        seed (int), (default=1): seed to be used for reproducibility.
        train_transforms (Optional[torchvision.transforms.Compose]):
            Additional Transforms to be applied to the training data.
        val_transforms (Optional[torchvision.transforms.Compose]):
            Additional Transforms to be applied to the validation/test data.

        Notes: Support for Numpy Arrays is missing Strings.

        """

    def __init__(self,
                 train: TIME_SERIES_CLASSIFICATION_INPUT,
                 val: Optional[TIME_SERIES_CLASSIFICATION_INPUT] = None):
        _check_time_series_inputs(train=train,
                                  val=val,
                                  task_type="time_series_classification")
        super().__init__(train_tensors=train, val_tensors=val, shuffle=True)
        self.cross_validators = CrossValFuncs.get_cross_validators(
            CrossValTypes.stratified_k_fold_cross_validation,
            CrossValTypes.k_fold_cross_validation,
            CrossValTypes.shuffle_split_cross_validation,
            CrossValTypes.stratified_shuffle_split_cross_validation
        )
        self.holdout_validators = HoldOutFuncs.get_holdout_validators(
            HoldoutValTypes.holdout_validation,
            HoldoutValTypes.stratified_holdout_validation
        )


class TimeSeriesRegressionDataset(BaseDataset):
    def __init__(self, train: Tuple[np.ndarray, np.ndarray], val: Optional[Tuple[np.ndarray, np.ndarray]] = None):
        _check_time_series_inputs(train=train,
                                  val=val,
                                  task_type="time_series_regression")
        super().__init__(train_tensors=train, val_tensors=val, shuffle=True)
        self.cross_validators = CrossValFuncs.get_cross_validators(
            CrossValTypes.k_fold_cross_validation,
            CrossValTypes.shuffle_split_cross_validation
        )
        self.holdout_validators = HoldOutFuncs.get_holdout_validators(
            HoldoutValTypes.holdout_validation
        )


def _check_time_series_inputs(task_type: str,
                              train: Union[TIME_SERIES_CLASSIFICATION_INPUT, TIME_SERIES_REGRESSION_INPUT],
                              val: Optional[
                                  Union[TIME_SERIES_CLASSIFICATION_INPUT, TIME_SERIES_REGRESSION_INPUT]] = None
                              ) -> None:
    if len(train) != 2:
        raise ValueError(f"There must be exactly two training tensors for {task_type}. "
                         f"The first one containing the data and the second one containing the targets.")
    if train[0].ndim != 3:
        raise ValueError(
            f"The training data for {task_type} has to be a three-dimensional tensor of shape NxSxM.")
    if train[1].ndim != 1:
        raise ValueError(
            f"The training targets for {task_type} have to be of shape N."
        )
    if val is not None:
        if len(val) != 2:
            raise ValueError(
                f"There must be exactly two validation tensors for{task_type}. "
                f"The first one containing the data and the second one containing the targets.")
        if val[0].ndim != 3:
            raise ValueError(
                f"The validation data for {task_type} has to be a "
                f"three-dimensional tensor of shape NxSxM.")
        if val[0].ndim != 1:
            raise ValueError(
                f"The validation targets for {task_type} have to be of shape N."
            )
