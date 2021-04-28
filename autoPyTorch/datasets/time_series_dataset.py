from typing import Any, Dict, Optional, Tuple, Union

import numpy as np

import torchvision.transforms

from autoPyTorch.datasets.base_dataset import BaseDataset
from autoPyTorch.datasets.resampling_strategy import (
    CrossValFuncs,
    CrossValTypes,
    HoldOutFuncs,
    HoldoutValTypes
)

TIME_SERIES_FORECASTING_INPUT = Tuple[np.ndarray, np.ndarray]  # currently only numpy arrays are supported
TIME_SERIES_REGRESSION_INPUT = Tuple[np.ndarray, np.ndarray]
TIME_SERIES_CLASSIFICATION_INPUT = Tuple[np.ndarray, np.ndarray]


class TimeSeriesForecastingDataset(BaseDataset):
    def __init__(self,
                 target_variables: Tuple[int],
                 sequence_length: int,
                 n_steps: int,
                 train: TIME_SERIES_FORECASTING_INPUT,
                 val: Optional[TIME_SERIES_FORECASTING_INPUT] = None,
                 resampling_strategy: Union[CrossValTypes, HoldoutValTypes] = HoldoutValTypes.holdout_validation,
                 resampling_strategy_args: Optional[Dict[str, Any]] = None,
                 shuffle: Optional[bool] = False,
                 seed: Optional[int] = 42,
                 train_transforms: Optional[torchvision.transforms.Compose] = None,
                 val_transforms: Optional[torchvision.transforms.Compose] = None,
                 ):
        """

        :param target_variables: The indices of the variables you want to forecast
        :param sequence_length: The amount of past data you want to use to forecast future value
        :param n_steps: The number of steps you want to forecast into the future
        :param train: Tuple with one tensor holding the training data
        :param val: Tuple with one tensor holding the validation data
        """
        _check_time_series_forecasting_inputs(
            target_variables=target_variables,
            sequence_length=sequence_length,
            n_steps=n_steps,
            train=train,
            val=val)
        train = _prepare_time_series_forecasting_tensor(tensor=train,
                                                        target_variables=target_variables,
                                                        sequence_length=sequence_length,
                                                        n_steps=n_steps)
        if val is not None:
            val = _prepare_time_series_forecasting_tensor(tensor=val,
                                                          target_variables=target_variables,
                                                          sequence_length=sequence_length,
                                                          n_steps=n_steps)
        super().__init__(train_tensors=train, val_tensors=val, shuffle=shuffle,
                         resampling_strategy=resampling_strategy, resampling_strategy_args=resampling_strategy_args,
                         seed=seed,
                         train_transforms=train_transforms,
                         val_transforms=val_transforms,
                         )
        self.cross_validators = CrossValFuncs.get_cross_validators(CrossValTypes.time_series_cross_validation)
        self.holdout_validators = HoldOutFuncs.get_holdout_validators(HoldoutValTypes.holdout_validation)


def _check_time_series_forecasting_inputs(target_variables: Tuple[int],
                                          sequence_length: int,
                                          n_steps: int,
                                          train: TIME_SERIES_FORECASTING_INPUT,
                                          val: Optional[TIME_SERIES_FORECASTING_INPUT] = None) -> None:
    if train[0].ndim != 3:
        raise ValueError(
            "The training data for time series forecasting has to be a three-dimensional tensor of shape PxLxM.")
    if val is not None:
        if val[0].ndim != 3:
            raise ValueError(
                "The validation data for time series forecasting "
                "has to be a three-dimensional tensor of shape PxLxM.")
    _, time_series_length, num_features = train[0].shape
    if sequence_length + n_steps > time_series_length:
        raise ValueError(f"Invalid sequence length: Cannot create dataset "
                         f"using sequence_length={sequence_length} and n_steps={n_steps} "
                         f"when the time series are of length {time_series_length}")
    for t in target_variables:
        if t < 0 or t >= num_features:
            raise ValueError(f"Target variable {t} is out of bounds. Number of features is {num_features}, "
                             f"so each target variable has to be between 0 and {num_features - 1}.")


def _prepare_time_series_forecasting_tensor(tensor: TIME_SERIES_FORECASTING_INPUT,
                                            target_variables: Tuple[int],
                                            sequence_length: int,
                                            n_steps: int) -> Tuple[np.ndarray, np.ndarray]:
    population_size, time_series_length, num_features = tensor[0].shape
    num_targets = len(target_variables)
    num_datapoints = time_series_length - sequence_length - n_steps + 1
    x_tensor = np.zeros((num_datapoints, population_size, sequence_length, num_features), dtype=np.float)
    y_tensor = np.zeros((num_datapoints, population_size, num_targets), dtype=np.float)

    for p in range(population_size):
        for i in range(num_datapoints):
            x_tensor[i, p, :, :] = tensor[0][p, i:i + sequence_length, :]
            y_tensor[i, p, :] = tensor[0][p, i + sequence_length + n_steps - 1, target_variables]

    # get rid of population dimension by reshaping
    x_tensor = x_tensor.reshape((-1, sequence_length, num_features))
    y_tensor = y_tensor.reshape((-1, num_targets))
    return x_tensor, y_tensor


class TimeSeriesClassificationDataset(BaseDataset):
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
