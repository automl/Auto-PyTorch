from typing import Any, Dict, Optional, Union

import numpy as np

import pandas as pd

import torchvision.transforms

from autoPyTorch.constants import (
    CLASSIFICATION_OUTPUTS,
    CLASSIFICATION_TASKS,
    REGRESSION_OUTPUTS,
    STRING_TO_OUTPUT_TYPES,
    STRING_TO_TASK_TYPES,
    TABULAR_CLASSIFICATION,
    TABULAR_REGRESSION,
    TASK_TYPES_TO_STRING,
)
from autoPyTorch.data.base_validator import BaseInputValidator
from autoPyTorch.datasets.base_dataset import BaseDataset, BaseDatasetPropertiesType
from autoPyTorch.datasets.resampling_strategy import (
    CrossValTypes,
    HoldoutValTypes,
    NoResamplingStrategyTypes
)


class TabularDataset(BaseDataset):
    """
        Base class for datasets used in AutoPyTorch
        Args:
            X (Union[np.ndarray, pd.DataFrame]): input training data.
            Y (Union[np.ndarray, pd.Series]): training data targets.
            X_test (Optional[Union[np.ndarray, pd.DataFrame]]):  input testing data.
            Y_test (Optional[Union[np.ndarray, pd.DataFrame]]): testing data targets
            resampling_strategy (Union[CrossValTypes, HoldoutValTypes, NoResamplingStrategyTypes]),
                (default=HoldoutValTypes.holdout_validation):
                strategy to split the training data.
            resampling_strategy_args (Optional[Dict[str, Any]]):
                arguments required for the chosen resampling strategy. If None, uses
                the default values provided in DEFAULT_RESAMPLING_PARAMETERS
                in ```datasets/resampling_strategy.py```.
            shuffle:  Whether to shuffle the data before performing splits
            seed (int: default=1): seed to be used for reproducibility.
            train_transforms (Optional[torchvision.transforms.Compose]):
                Additional Transforms to be applied to the training data.
            val_transforms (Optional[torchvision.transforms.Compose]):
                Additional Transforms to be applied to the validation/test data.

        Notes: Support for Numpy Arrays is missing Strings.

        """

    def __init__(self,
                 X: Union[np.ndarray, pd.DataFrame],
                 Y: Union[np.ndarray, pd.Series],
                 X_test: Optional[Union[np.ndarray, pd.DataFrame]] = None,
                 Y_test: Optional[Union[np.ndarray, pd.DataFrame]] = None,
                 resampling_strategy: Union[CrossValTypes,
                                            HoldoutValTypes,
                                            NoResamplingStrategyTypes] = HoldoutValTypes.holdout_validation,
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
            raise ValueError("A feature validator is required to build a tabular pipeline")

        X, Y = validator.transform(X, Y)
        if X_test is not None:
            X_test, Y_test = validator.transform(X_test, Y_test)
        self.categorical_columns = validator.feature_validator.categorical_columns
        self.numerical_columns = validator.feature_validator.numerical_columns
        self.num_features = validator.feature_validator.num_features
        self.categories = validator.feature_validator.categories

        super().__init__(train_tensors=(X, Y), test_tensors=(X_test, Y_test), shuffle=shuffle,
                         resampling_strategy=resampling_strategy,
                         resampling_strategy_args=resampling_strategy_args,
                         seed=seed, train_transforms=train_transforms,
                         dataset_name=dataset_name,
                         val_transforms=val_transforms)
        self.issigned = bool(np.any((X.data if self.issparse else X) < 0))
        if self.output_type is not None:
            if STRING_TO_OUTPUT_TYPES[self.output_type] in CLASSIFICATION_OUTPUTS:
                self.task_type = TASK_TYPES_TO_STRING[TABULAR_CLASSIFICATION]
            elif STRING_TO_OUTPUT_TYPES[self.output_type] in REGRESSION_OUTPUTS:
                self.task_type = TASK_TYPES_TO_STRING[TABULAR_REGRESSION]
            else:
                raise ValueError("Output type not currently supported ")
        else:
            raise ValueError("Task type not currently supported ")
        if STRING_TO_TASK_TYPES[self.task_type] in CLASSIFICATION_TASKS:
            self.num_classes: int = len(np.unique(self.train_tensors[1]))

    def get_required_dataset_info(self) -> Dict[str, BaseDatasetPropertiesType]:
        """
        Returns a dictionary containing required dataset
        properties to instantiate a pipeline.
        For a Tabular Dataset this includes-
            1. 'output_type'- Enum indicating the type of the output for this problem.
                We currently use the `sklearn type_of_target
                <https://scikit-learn.org/stable/modules/generated/sklearn.utils.multiclass.type_of_target.html>`
                to infer the output type from the data and we encode it to an
                Enum for which you can find more info in  `autopytorch/constants.py
                <https://github.com/automl/Auto-PyTorch/blob/refactor_development/autoPyTorch/constants.py>`
            2. 'issparse'- A flag indicating if the input is in a sparse matrix.
            3. 'numerical_columns'- a list which contains the column numbers
                for the numerical columns in the input dataset
            4. 'categorical_columns'- a list which contains the column numbers
                for the categorical columns in the input dataset
            5. 'task_type'- Enum indicating the type of task. For tabular datasets,
                currently we support 'tabular_classification' and 'tabular_regression'. and we encode it to an
                Enum for which you can find more info in  `autopytorch/constants.py
                <https://github.com/automl/Auto-PyTorch/blob/refactor_development/autoPyTorch/constants.py>`
        """
        info = super().get_required_dataset_info()
        assert self.task_type is not None, "Expected value for task type but got None"
        info.update({
            'numerical_columns': self.numerical_columns,
            'categorical_columns': self.categorical_columns,
            'task_type': self.task_type,
            'issigned': self.issigned
        })
        return info
