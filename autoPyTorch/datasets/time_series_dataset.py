from typing import Any, Dict, List, Optional, Tuple, Union
import warnings

import numpy as np

import pandas as pd
import sklearn

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
from autoPyTorch.datasets.base_dataset import BaseDataset
from autoPyTorch.datasets.resampling_strategy import (
    DEFAULT_RESAMPLING_PARAMETERS,
    CrossValTypes,
    HoldoutValTypes,
    get_cross_validators,
    get_holdout_validators,
    is_stratified,
)

from autoPyTorch.utils.common import FitRequirement

#TIME_SERIES_FORECASTING_INPUT = Tuple[np.ndarray, np.ndarray]  # currently only numpy arrays are supported
#TIME_SERIES_REGRESSION_INPUT = Tuple[np.ndarray, np.ndarray]
#TIME_SERIES_CLASSIFICATION_INPUT = Tuple[np.ndarray, np.ndarray]

class TimeSeriesForecastingDataset(BaseDataset):
    def __init__(self,
                 X: np.ndarray,
                 Y: Union[np.ndarray, pd.Series],
                 X_test: Optional[Union[np.ndarray, pd.DataFrame]] = None,
                 Y_test: Optional[Union[np.ndarray, pd.DataFrame]] = None,
                 resampling_strategy: Union[CrossValTypes, HoldoutValTypes] = HoldoutValTypes.holdout_validation,
                 resampling_strategy_args: Optional[Dict[str, Any]] = None,
                 shuffle: Optional[bool] = False,
                 seed: Optional[int] = 42,
                 train_transforms: Optional[torchvision.transforms.Compose] = None,
                 val_transforms: Optional[torchvision.transforms.Compose] = None,
                 validator: Optional[BaseInputValidator] = None,
                 n_prediction_steps: int = 1,
                 ):
        """

        :param target_variables: The indices of the variables you want to forecast
        :param sequence_length: The amount of past data you want to use to forecast future value
        :param n_steps: The number of steps you want to forecast into the future
        :param train: Tuple with one tensor holding the training data
        :param val: Tuple with one tensor holding the validation data
        """

        self.validator = validator
        if self.validator is not None:
            X, Y = self.validator.transform(X, Y)
            if X_test is not None:
                X_test, Y_test = self.validator.transform(X_test, Y_test)

        _check_time_series_forecasting_inputs(train=X, val=X_test)
        # swap the axis of population_size and sequence_length hence the splitter will split the dataset w.r.t. sequence
        X = np.swapaxes(X, 0, 1)
        Y = np.swapaxes(Y, 0, 1)
        train_tensors = (X.astype(np.float32), Y.astype(np.float32)[0])
        if X_test is not None and Y_test is not None:
            X_test = np.swapaxes(X_test, 0, 1)
            Y_test = np.swapaxes(Y_test, 0, 1)
            test_tensors = (X_test.astype(np.float32)[0], Y_test.astype(np.float32))
        else:
            test_tensors = None
        if shuffle:
            warnings.WarningMessage("Time Series Forecasting will not shuffle the data")
        super().__init__(train_tensors=train_tensors, test_tensors=test_tensors, shuffle=False,
                         resampling_strategy=resampling_strategy, resampling_strategy_args=resampling_strategy_args,
                         seed=seed,
                         train_transforms=train_transforms,
                         val_transforms=val_transforms,
                         )
        self.is_small_preprocess = False

        self.task_type = TASK_TYPES_TO_STRING[TIMESERIES_FORECASTING]

        self.train_tensors = (X.astype(np.float32), Y.astype(np.float32))
        self.test_tensors = (X_test.astype(np.float32), Y.astype(np.float32))
        self.num_features = self.train_tensors[0].shape[2]
        self.numerical_features: List[int] = list(range(self.num_features))
        self.categorical_features: List[int] = []
        self.n_prediction_steps = n_prediction_steps

        self.cross_validators = get_cross_validators(CrossValTypes.time_series_cross_validation)
        self.holdout_validators = get_holdout_validators(HoldoutValTypes.holdout_validation)

        self.splits = self.get_splits_from_resampling_strategy()

        # We also need to be able to transform the data, be it for pre-processing
        # or for augmentation
        self.train_transform = train_transforms
        self.val_transform = val_transforms

        time_series_length = self.train_tensors[0].shape[0]
        if isinstance(self.resampling_strategy, HoldoutValTypes):
            if self.val_tensors is not None:
                max_sequence_length = time_series_length - self.n_prediction_steps
            else:
                val_share = DEFAULT_RESAMPLING_PARAMETERS[self.resampling_strategy].get(
                    'val_share', None)
                if self.resampling_strategy_args is not None:
                    val_share = self.resampling_strategy_args.get('val_share', val_share)
                upper_sequence_length = int(time_series_length * val_share) - self.n_prediction_steps

        elif isinstance(self.resampling_strategy, CrossValTypes):
            num_splits = DEFAULT_RESAMPLING_PARAMETERS[self.resampling_strategy].get(
                'num_splits', None)
            if self.resampling_strategy_args is not None:
                num_splits = self.resampling_strategy_args.get('num_splits', num_splits)
            upper_sequence_length = (time_series_length // num_splits) - self.n_prediction_steps
        else:
            raise ValueError()
        self.upper_sequence_length = upper_sequence_length

    def get_required_dataset_info(self) -> Dict[str, Any]:
        """
        Returns a dictionary containing required dataset properties to instantiate a pipeline,
        """
        info = super().get_required_dataset_info()
        info.update({
            'task_type': self.task_type,
            'numerical_features': self.numerical_features,
            'categorical_features': self.categorical_features,
            'upper_sequence_length': self.upper_sequence_length,
        })
        return info

    def get_dataset_properties(self, dataset_requirements: List[FitRequirement]) -> Dict[str, Any]:
        dataset_properties = super().get_dataset_properties(dataset_requirements=dataset_requirements)
        dataset_properties.update({'upper_sequence_length': self.upper_sequence_length})
        return dataset_properties


def _check_time_series_forecasting_inputs(train: np.ndarray,
                                          val: Optional[np.ndarray] = None) -> None:
    if train.ndim != 3:
        raise ValueError(
            "The training data for time series forecasting has to be a three-dimensional tensor of shape PxLxM.")
    if val is not None:
        if val.ndim != 3:
            raise ValueError(
                "The validation data for time series forecasting "
                "has to be a three-dimensional tensor of shape PxLxM.")


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
