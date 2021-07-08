from typing import Any, Dict, List, Optional, Tuple, Union, cast
import warnings

import numpy as np

import pandas as pd
from scipy.sparse import issparse

from torch.utils.data.dataset import Dataset, Subset, ConcatDataset

import torchvision.transforms

from autoPyTorch.constants import (
    CLASSIFICATION_OUTPUTS,
    CLASSIFICATION_TASKS,
    REGRESSION_OUTPUTS,
    STRING_TO_OUTPUT_TYPES,
    STRING_TO_TASK_TYPES,
    TASK_TYPES_TO_STRING,
    TIMESERIES_CLASSIFICATION,
    TIMESERIES_REGRESSION,
    TIMESERIES_FORECASTING,
)
from autoPyTorch.data.base_validator import BaseInputValidator
from autoPyTorch.datasets.base_dataset import BaseDataset, type_check, type_of_target, TransformSubset
from autoPyTorch.datasets.resampling_strategy import (
    DEFAULT_RESAMPLING_PARAMETERS,
    CrossValTypes,
    HoldoutValTypes,
    get_cross_validators,
    get_holdout_validators,
    is_stratified,
)

from autoPyTorch.utils.common import FitRequirement
from autoPyTorch.data.time_series_forecasting_validator import TimeSeriesForecastingInputValidator
from autoPyTorch.utils.common import FitRequirement, hash_array_or_matrix
from autoPyTorch.datasets.tabular_dataset import TabularDataset

#TIME_SERIES_FORECASTING_INPUT = Tuple[np.ndarray, np.ndarray]  # currently only numpy arrays are supported
#TIME_SERIES_REGRESSION_INPUT = Tuple[np.ndarray, np.ndarray]
#TIME_SERIES_CLASSIFICATION_INPUT = Tuple[np.ndarray, np.ndarray]

"""
class TimeSeriesSequence(BaseDataset):
    def __init__(self,
                 train_tensors: Union[np.ndarray, List[List]],
                 dataset_name: Optional[str] = None,
                 val_tensors: Optional[ Union[np.ndarray, List[List]]] = None,
                 test_tensors: Optional[ Union[np.ndarray, List[List]]] = None,
                 resampling_strategy: Union[CrossValTypes, HoldoutValTypes] = HoldoutValTypes.time_series_hold_out_validation,
                 resampling_strategy_args: Optional[Dict[str, Any]] = None,
                 seed: Optional[int] = 42,
                 train_transforms: Optional[torchvision.transforms.Compose] = None,
                 val_transforms: Optional[torchvision.transforms.Compose] = None,
                 ):
        if dataset_name is not None:
            self.dataset_name = dataset_name
        else:
            self.dataset_name = hash_array_or_matrix(train_tensors[0])
        if not hasattr(train_tensors[0], 'shape'):
            type_check(train_tensors, val_tensors)
        self.train_tensors = train_tensors
        self.val_tensors = val_tensors
        self.test_tensors = test_tensors
        self.cross_validators = {}
        self.holdout_validators = {}
        self.rand = np.random.RandomState(seed=seed)
        self.shuffle = False
        self.task_type: Optional[str] = None

        self.resampling_strategy = resampling_strategy
        self.resampling_strategy_args = resampling_strategy_args

        self.cross_validators = get_cross_validators(CrossValTypes.time_series_cross_validation)
        self.holdout_validators = get_holdout_validators(HoldoutValTypes.time_series_hold_out_validation)

        self.splits = self.get_splits_from_resampling_strategy()

        # We also need to be able to transform the data, be it for pre-processing
        # or for augmentation
        self.train_transform = train_transforms
        self.val_transform = val_transforms
"""


class TimeSeriesForecastingDataset(BaseDataset):
    def __init__(self,
                 X: Union[np.ndarray, List[List]],
                 Y: Union[np.ndarray, pd.Series],
                 X_test: Optional[Union[np.ndarray, pd.DataFrame]] = None,
                 Y_test: Optional[Union[np.ndarray, pd.DataFrame]] = None,
                 dataset_name: Optional[str] = None,
                 resampling_strategy: Union[CrossValTypes, HoldoutValTypes] = HoldoutValTypes.time_series_hold_out_validation,
                 resampling_strategy_args: Optional[Dict[str, Any]] = None,
                 shuffle: Optional[bool] = False,
                 seed: Optional[int] = 42,
                 train_transforms: Optional[torchvision.transforms.Compose] = None,
                 val_transforms: Optional[torchvision.transforms.Compose] = None,
                 validator: Optional[TimeSeriesForecastingInputValidator] = None,
                 n_prediction_steps: int = 1,
                 ):
        """
        :param target_variables: The indices of the variables you want to forecast
        :param sequence_length: The amount of past data you want to use to forecast future value
        :param n_steps: The number of steps you want to forecast into the future
        :param train: Tuple with one tensor holding the training data
        :param val: Tuple with one tensor holding the validation data
        """
        assert X is not Y, "Training and Test data needs to belong two different object!!!"
        self.n_prediction_steps = n_prediction_steps
        self.validator = validator
        if self.validator is not None:
            if not isinstance(validator, TimeSeriesForecastingInputValidator):
                raise ValueError(f"This dataset only support TimeSeriesForecastingInputValidator "
                                 f"but receive {type(validator)}")

            X, Y = self.validator.transform(X, Y)
            self.num_features = self.validator.feature_validator.n_feature_dims
            self.num_target = self.validator.target_validator.out_dimensionality

            if X_test is not None:
                X_test, Y_test = self.validator.transform(X_test, Y_test)
        else:
            self.num_features = np.shape(X[0])[-1]
            self.num_target = np.shape(Y[0])[-1]

        self.num_sequences = len(X)
        self.sequence_lengths_train = [0] * self.num_sequences
        for seq_idx in range(self.num_sequences):
            self.sequence_lengths_train[seq_idx] = len(X[seq_idx])

        self.sequence_lengths_val = [0] * self.num_sequences
        self.sequence_lengths_test = [0] * self.num_sequences

        self.categorical_columns = validator.feature_validator.categorical_columns
        self.numerical_columns = validator.feature_validator.numerical_columns

        num_train_data = np.sum(self.sequence_lengths_train)
        X_train_flatten = np.empty([num_train_data, self.num_features])
        y_train_flatten = np.empty([num_train_data, self.num_features])
        start_idx = 0

        self.sequences = []

        if shuffle:
            warnings.WarningMessage("Time Series Forecasting will not shuffle the data")
        for seq_idx, seq_length in enumerate(self.sequence_lengths_train):
            end_idx = start_idx + seq_length
            X_train_flatten[start_idx: end_idx] = np.array(X[seq_idx])
            y_train_flatten[start_idx: end_idx] = np.array(Y[seq_idx])
            start_idx = end_idx

        train_tensors = (X_train_flatten, y_train_flatten)

        if X_test is not None and Y_test is not None:
            for seq_idx in range(self.num_sequences):
                self.sequence_lengths_test[seq_idx] = len(X_test[seq_idx])
            num_test_data = np.sum(self.sequence_lengths_test)
            X_test_flatten = np.empty([num_test_data, self.num_features])
            y_test_flatten = np.empty([num_test_data, self.num_target])
            start_idx = 0

            for seq_idx, seq_length in enumerate(self.sequence_lengths_test):
                end_idx = start_idx + seq_length
                X_test_flatten[start_idx: end_idx] = np.array(X_test[seq_idx])
                y_test_flatten[start_idx: end_idx] = np.array(Y_test[seq_idx])
                start_idx = end_idx
            test_tensors = (X_test_flatten, y_test_flatten)
        else:
            test_tensors = None
        """
        super(TimeSeriesForecastingDataset, self).__init__(train_tensors=train_tensors,
                                                     dataset_name=dataset_name,
                                                     test_tensors=test_tensors,
                                                     resampling_strategy=resampling_strategy,
                                                     resampling_strategy_args=resampling_strategy_args,
                                                     shuffle=False,
                                                     seed=seed,
                                                     train_transforms=train_transforms,
                                                     val_transforms=val_transforms)
        """
        if dataset_name is not None:
            self.dataset_name = dataset_name
        else:
            self.dataset_name = hash_array_or_matrix(train_tensors[0])

        self.train_tensors = train_tensors
        self.val_tensors = None
        self.test_tensors = test_tensors
        self.rand = np.random.RandomState(seed=seed)
        self.shuffle = False
        self.resampling_strategy = resampling_strategy
        self.resampling_strategy_args = resampling_strategy_args
        self.task_type: Optional[str] = None
        self.issparse: bool = issparse(self.train_tensors[0])
        # TODO find a way to edit input shape!
        self.input_shape: Tuple[int] = [np.min(self.sequence_lengths_train), self.num_features]

        if len(self.train_tensors) == 2 and self.train_tensors[1] is not None:
            self.output_type: str = type_of_target(self.train_tensors[1])

            if STRING_TO_OUTPUT_TYPES[self.output_type] in CLASSIFICATION_OUTPUTS:
                self.output_shape = len(np.unique(self.train_tensors[1]))
            else:
                # self.output_shape = self.train_tensors[1].shape[-1] if self.train_tensors[1].ndim > 1 else 1
                self.output_shape = self.train_tensors[1].shape[-1] if self.train_tensors[1].ndim > 1 else 1

        # TODO: Look for a criteria to define small enough to preprocess
        self.is_small_preprocess = False

        # We also need to be able to transform the data, be it for pre-processing
        # or for augmentation
        self.train_transform = train_transforms
        self.val_transform = val_transforms


        self.task_type = TASK_TYPES_TO_STRING[TIMESERIES_FORECASTING]

        self.numerical_features: List[int] = list(range(self.num_features))
        self.categorical_features: List[int] = []

        self.cross_validators = get_cross_validators(CrossValTypes.time_series_cross_validation)
        self.holdout_validators = get_holdout_validators(HoldoutValTypes.time_series_hold_out_validation)

        self.splits = self.get_splits_from_resampling_strategy()

        # We also need to be able to transform the data, be it for pre-processing
        # or for augmentation
        self.train_transform = train_transforms
        self.val_transform = val_transforms

    def get_splits_from_resampling_strategy(self) -> List[Tuple[List[int], List[int]]]:
        """
        Creates a set of splits based on a resampling strategy provided, apart from the
        'get_splits_from_resampling_strategy' implemented in base_dataset, here we will get self.upper_sequence_length
        with the given value

        Returns
            (List[Tuple[List[int], List[int]]]): splits in the [train_indices, val_indices] format
        """
        splits= []
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
                upper_window_size = int(np.min(self.sequence_lengths_train) * val_share) - self.n_prediction_steps

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
            upper_window_size = (np.min(self.sequence_lengths_train) // num_splits) - self.n_prediction_steps
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
            'upper_window_size': self.upper_window_size,
        })
        return info

    def get_dataset_properties(self, dataset_requirements: List[FitRequirement]) -> Dict[str, Any]:
        dataset_properties = super().get_dataset_properties(dataset_requirements=dataset_requirements)
        dataset_properties.update({'upper_window_size': self.upper_window_size})
        return dataset_properties

    def update_sequence_lengths_train(self, sequence_length):
        if len(sequence_length) != self.num_sequences:
            raise ValueError("number of sequence must match!")
        if np.sum(sequence_length) != self.train_tensors[0].shape[0]:
            raise ValueError("sequence length needs to be consistent with train tensors")
        self.sequence_lengths_train = sequence_length

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
        kwargs = {"n_prediction_steps": self.n_prediction_steps}
        splits = [[() for _ in range(self.num_sequences)] for _ in range(num_splits)]
        idx_all = self._get_indices()
        idx_start = 0
        for idx_seq, seq_length in enumerate(self.sequence_lengths_train):
            idx_end = idx_start + seq_length
            split = self.cross_validators[cross_val_type.name](num_splits, idx_all[idx_start: idx_end], **kwargs)
            for idx_split in range(num_splits):
                splits[idx_split][idx_seq] = split[idx_split]
            idx_start = idx_end
        # in this case, splits is stored as :
        #  [ first split, second_split ...]
        #  first_split = [([0], [1]), ([2], [3])] ....
        splits_merged = []
        for i in range(num_splits):
            split = splits[i]
            train_indices = np.concatenate([sp[0] for sp in split])
            test_indices = np.concatenate([sp[1] for sp in split])
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
        if self.val_tensors is not None:
            raise ValueError(
                '`val_share` specified, but the Dataset was a given a pre-defined split at initialization already.')
        if val_share < 0 or val_share > 1:
            raise ValueError(f"`val_share` must be between 0 and 1, got {val_share}.")
        if not isinstance(holdout_val_type, HoldoutValTypes):
            raise NotImplementedError(f'The specified `holdout_val_type` "{holdout_val_type}" is not supported.')
        kwargs = {"n_prediction_steps": self.n_prediction_steps}

        splits = [[() for _ in range(self.num_sequences)] for _ in range(2)]
        idx_all = self._get_indices()
        idx_start = 0
        for idx_seq, seq_length in enumerate(self.sequence_lengths_train):
            idx_end = idx_start + seq_length
            split = self.holdout_validators[holdout_val_type.name](val_share, idx_all[idx_start: idx_end], **kwargs)
            for idx_split in range(2):
                splits[idx_split][idx_seq] = split[idx_split]
            idx_start = idx_end

        train_indices = np.concatenate([sp for sp in splits[0]])
        test_indices = np.concatenate([sp for sp in splits[1]])

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
                 X: np.ndarray,
                 Y: Union[np.ndarray, pd.Series],
                 X_test: Optional[Union[np.ndarray, pd.DataFrame]] = None,
                 Y_test: Optional[Union[np.ndarray, pd.DataFrame]] = None,
                 resampling_strategy: Union[CrossValTypes, HoldoutValTypes] = HoldoutValTypes.holdout_validation,
                 resampling_strategy_args: Optional[Dict[str, Any]] = None,
                 shuffle: Optional[bool] = True,
                 seed: Optional[int] = 42,
                 train_transforms: Optional[torchvision.transforms.Compose] = None,
                 val_transforms: Optional[torchvision.transforms.Compose] = None,
                 dataset_name: Optional[str] = None,
                 validator: Optional[BaseInputValidator] = None,
                 ):
        # Take information from the validator, which guarantees clean data for the
        # dataset.
        # TODO: Consider moving the validator to the pipeline itself when we
        # move to using the fit_params on scikit learn 0.24
        if validator is None:
            raise ValueError("A feature validator is required to build a time series pipeline")

        self.validator = validator

        X, Y = self.validator.transform(X, Y)
        if X_test is not None:
            X_test, Y_test = self.validator.transform(X_test, Y_test)

        super().__init__(train_tensors=(X, Y),
                         test_tensors=(X_test, Y_test),
                         shuffle=shuffle,
                         resampling_strategy=resampling_strategy,
                         resampling_strategy_args=resampling_strategy_args,
                         seed=seed, train_transforms=train_transforms,
                         dataset_name=dataset_name,
                         val_transforms=val_transforms)

        if self.output_type is not None:
            if STRING_TO_OUTPUT_TYPES[self.output_type] in CLASSIFICATION_OUTPUTS:
                self.task_type = TASK_TYPES_TO_STRING[TIMESERIES_CLASSIFICATION]
            elif STRING_TO_OUTPUT_TYPES[self.output_type] in REGRESSION_OUTPUTS:
                self.task_type = TASK_TYPES_TO_STRING[TIMESERIES_REGRESSION]
            else:
                raise ValueError(f"Output type {self.output_type} currently not supported ")
        else:
            raise ValueError("Task type not currently supported ")
        if STRING_TO_TASK_TYPES[self.task_type] in CLASSIFICATION_TASKS:
            self.num_classes: int = len(np.unique(self.train_tensors[1]))

        # filter the default cross and holdout validators if we have a regression task
        # since we cannot use stratification there
        if self.task_type == TASK_TYPES_TO_STRING[TIMESERIES_REGRESSION]:
            self.cross_validators = {cv_type: cv for cv_type, cv in self.cross_validators.items()
                                     if not is_stratified(cv_type)}
            self.holdout_validators = {hv_type: hv for hv_type, hv in self.holdout_validators.items()
                                       if not is_stratified(hv_type)}

        self.num_features = self.train_tensors[0].shape[2]
        self.numerical_features: List[int] = list(range(self.num_features))
        self.categorical_features: List[int] = []

    def get_required_dataset_info(self) -> Dict[str, Any]:
        """
        Returns a dictionary containing required dataset properties to instantiate a pipeline,
        """
        info = super().get_required_dataset_info()
        info.update({
            'task_type': self.task_type,
            'numerical_features': self.numerical_features,
            'categorical_features': self.categorical_features,

        })
        return info
