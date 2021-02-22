from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

import pandas as pd

import torchvision.transforms

from autoPyTorch.constants import CLASSIFICATION_OUTPUTS, CLASSIFICATION_TASKS, REGRESSION_OUTPUTS, \
    STRING_TO_OUTPUT_TYPES, STRING_TO_TASK_TYPES, TASK_TYPES_TO_STRING, TIMESERIES_CLASSIFICATION, TIMESERIES_REGRESSION
from autoPyTorch.data.base_validator import BaseInputValidator
from autoPyTorch.datasets.base_dataset import BaseDataset
from autoPyTorch.datasets.resampling_strategy import (
    CrossValTypes,
    HoldoutValTypes,
    get_cross_validators,
    get_holdout_validators, is_stratified
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
        self.cross_validators = get_cross_validators(CrossValTypes.time_series_cross_validation)
        self.holdout_validators = get_holdout_validators(HoldoutValTypes.holdout_validation)


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
    x_tensor = np.zeros((num_datapoints, population_size, sequence_length, num_features), dtype=np.float32)
    y_tensor = np.zeros((num_datapoints, population_size, num_targets), dtype=np.float32)

    for p in range(population_size):
        for i in range(num_datapoints):
            x_tensor[i, p, :, :] = tensor[0][p, i:i + sequence_length, :]
            y_tensor[i, p, :] = tensor[0][p, i + sequence_length + n_steps - 1, target_variables]

    # get rid of population dimension by reshaping
    x_tensor = x_tensor.reshape((-1, sequence_length, num_features))
    y_tensor = y_tensor.reshape((-1, num_targets))
    return x_tensor, y_tensor


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
