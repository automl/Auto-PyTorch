from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

import pandas as pd

from sklearn.utils import check_array

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
from autoPyTorch.datasets.base_dataset import BaseDataset
from autoPyTorch.datasets.resampling_strategy import (
    CrossValTypes,
    HoldOutTypes,
)


class DataTypes(Enum):
    Canonical = 1
    Float = 2
    String = 3
    Categorical = 4


class Value2Index(object):
    def __init__(self, values: list):
        assert all(not (pd.isna(v)) for v in values)
        self.values = {v: i for i, v in enumerate(values)}

    def __getitem__(self, item: Any) -> int:
        if pd.isna(item):
            # SHUHEI MEMO: Why do we add the location for nan?
            return 0
        else:
            return self.values[item] + 1


class TabularDataset(BaseDataset):
    """
        Base class for datasets used in AutoPyTorch
        Args:
            X (Union[np.ndarray, pd.DataFrame]): input training data.
            Y (Union[np.ndarray, pd.Series]): training data targets.
            X_test (Optional[Union[np.ndarray, pd.DataFrame]]):  input testing data.
            Y_test (Optional[Union[np.ndarray, pd.DataFrame]]): testing data targets
            splitting_type (Union[CrossValTypes, HoldOutTypes]),
                (default=HoldOutTypes.holdout_validation):
                strategy to split the training data.
            splitting_params (Optional[Dict[str, Any]]): arguments
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

    def __init__(self, X: Union[np.ndarray, pd.DataFrame],
                 Y: Union[np.ndarray, pd.Series],
                 X_test: Optional[Union[np.ndarray, pd.DataFrame]] = None,
                 Y_test: Optional[Union[np.ndarray, pd.DataFrame]] = None,
                 splitting_type: Union[CrossValTypes, HoldOutTypes] = HoldOutTypes.holdout_validation,
                 splitting_params: Optional[Dict[str, Any]] = None,
                 shuffle: Optional[bool] = True,
                 seed: Optional[int] = 42,
                 train_transforms: Optional[torchvision.transforms.Compose] = None,
                 val_transforms: Optional[torchvision.transforms.Compose] = None,
                 dataset_name: Optional[str] = None,
                 ):

        X, self.data_types, self.nan_mask, self.itovs, self.vtois = self.interpret_columns(X)

        if Y is not None:
            Y, _, self.target_nan_mask, self.target_itov, self.target_vtoi = self.interpret_columns(
                Y, assert_single_column=True)
            # For tabular classification, we expect also that it complies with Sklearn
            # The below check_array performs input data checks and make sure that a numpy array
            # is returned, as both Pytorch/Sklearn deal directly with numpy/list objects.
            # In this particular case, the interpret() returns a pandas object (needed to extract)
            # the data types, yet check_array translate the np.array. When Sklearn support pandas
            # the below function will simply return Pandas DataFrame.
            Y = check_array(Y, ensure_2d=False)

        # SHUHEI MEMO: num_features overlaps with input_shape in BaseDataset
        self.categorical_columns, self.numerical_columns, self.categories, self.num_features = \
            self.infer_dataset_properties(X)

        # Allow support for X_test, Y_test. They will NOT be used for optimization, but
        # rather to have a performance through time on the test data
        if X_test is not None:
            X_test, self._test_data_types, _, _, _ = self.interpret_columns(X_test)

            # Some quality checks on the data
            if self.data_types != self._test_data_types:
                raise ValueError(f"The train data inferred types {self.data_types} are "
                                 "different than the test inferred types {self._test_data_types}")
            if Y_test is not None:
                Y_test, _, _, _, _ = self.interpret_columns(
                    Y_test, assert_single_column=True)
                Y_test = check_array(Y_test, ensure_2d=False)

        """TODO: rename the variable names"""
        super().__init__(train_tensors=(X, Y), test_tensors=(X_test, Y_test), shuffle=shuffle,
                         splitting_type=splitting_type,
                         splitting_params=splitting_params,
                         random_state=seed, train_transforms=train_transforms,
                         dataset_name=dataset_name,
                         val_transforms=val_transforms)
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

    def interpret_columns(self,
                          data: Union[np.ndarray, pd.DataFrame, pd.Series],
                          assert_single_column: bool = False
                          ) -> Tuple[Union[pd.DataFrame, Any], List[DataTypes],
                                     Union[np.ndarray],
                                     List[Optional[list]],
                                     List[Optional[Value2Index]]]:
        """
        Interpret information such as data, data_types, nan_mask, itovs, vtois
        about the columns from the given data.

        Args:
            data (Union[np.ndarray, pd.DataFrame, pd.Series]): data to be
                interpreted.
            assert_single_column (bool), (default=False): flag for
                asserting that the data contains a single column

        Returns:
            data (pd.DataFrame): Converted data
            data_types (List[DataTypes]): Datatypes of each column
            nan_mask (Union[np.ndarray]): locations of nan in data
            itovs (List[Optional[list]]): The table value in the location (col, row)
            vtois (List[Optional[Value2Index]]): The index of the value in the specified column

            Tuple[pd.DataFrame, List[DataTypes],
                 Union[np.ndarray],
                 List[Optional[list]],
                 List[Optional[Value2Index]]]: Tuple of information
        """
        single_column = False
        if isinstance(data, np.ndarray):
            # SHUHEI MEMO: When does ',' not in str(data.dtype) happen?
            if len(data.shape) == 1 and ',' not in str(data.dtype):
                single_column = True
                data = data[:, None]
            data = pd.DataFrame(data).infer_objects().convert_dtypes()
        elif isinstance(data, pd.DataFrame):
            data = data.infer_objects().convert_dtypes()
        elif isinstance(data, pd.Series):
            single_column = True
            data = data.to_frame()
        else:
            raise ValueError('Provided data needs to be either an np.ndarray or a pd.DataFrame for TabularDataset.')
        if assert_single_column:
            assert single_column, \
                "The data is asserted to be only of a single column, but it isn't. \
                Most likely your targets are not a vector or series."

        data_types = []
        nan_mask = data.isna().to_numpy()
        for col_index, dtype in enumerate(data.dtypes):
            # SHUHEI MEMO: dtype.kind (https://numpy.org/doc/stable/reference/generated/numpy.dtype.kind.html)
            if dtype.kind == 'f':
                data_types.append(DataTypes.Float)
            elif dtype.kind in ('i', 'u', 'b'):
                data_types.append(DataTypes.Canonical)
            elif isinstance(dtype, pd.StringDtype):
                data_types.append(DataTypes.String)
            elif dtype.name == 'category':
                # OpenML format categorical columns as category
                # So add support for that
                data_types.append(DataTypes.Categorical)
            else:
                raise ValueError(f"The dtype in column {col_index} is {dtype} which is not supported.")
        # SHUHEI MEMO: index to value, value to index
        itovs: List[Optional[List[Any]]] = []
        vtois: List[Optional[Value2Index]] = []
        for col_index, (_, col) in enumerate(data.iteritems()):
            if data_types[col_index] != DataTypes.Float:
                # SHUHEI MEMO: Since we are taking a set, no replacement, but why is it fine?
                non_na_values = [v for v in set(col) if not pd.isna(v)]
                non_na_values.sort()
                itovs.append([np.nan] + non_na_values)
                vtois.append(Value2Index(non_na_values))
            else:
                itovs.append(None)
                vtois.append(None)

        if single_column:
            return data.iloc[:, 0], data_types, nan_mask, itovs, vtois

        return data, data_types, nan_mask, itovs, vtois

    def infer_dataset_properties(self, X: Any) \
            -> Tuple[List[int], List[int], List[object], int]:
        """
        Infers the properties of the dataset like
        categorical_columns, numerical_columns, categories, num_features
        Args:
            X: input training data

        Returns:
            categorical_columns (List[int]): The list of indices specifying categorical columns
            numerical_columns (List[int]): The list of indices specifying numerical columns
            categories (List[object]): The list of choices of each category
            num_features (int): The number of columns or features in a given tabular data

            (Tuple[List[int], List[int], List[object], int]):
        """
        categorical_columns = []
        numerical_columns = []
        for i, data_type in enumerate(self.data_types):
            if data_type == DataTypes.String or data_type == DataTypes.Categorical:
                categorical_columns.append(i)
            else:
                numerical_columns.append(i)
        # SHUHEI MEMO: Why don't we make it dict?
        categories = [np.unique(X.iloc[:, col_idx]).tolist() for col_idx in categorical_columns]
        num_features = X.shape[1]

        return categorical_columns, numerical_columns, categories, num_features
