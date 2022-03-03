"""
TODO:
    1. Add dtypes argument to TabularFeatureValidator
    2. Modify dtypes from List[str] to Dict[str, str]
    3. Add the feature to enforce the dtype to the provided dtypes
"""
import functools
from logging import Logger
from typing import Any, Dict, List, Mapping, Optional, Set, Tuple, Union, cast

import numpy as np

import pandas as pd
from pandas.api.types import is_numeric_dtype

from scipy.sparse import issparse, spmatrix

from sklearn import preprocessing
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import NotFittedError
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline

from autoPyTorch.data.base_feature_validator import BaseFeatureValidator, SupportedFeatTypes
from autoPyTorch.data.utils import (
    ColumnDTypes,
    DatasetCompressionInputType,
    DatasetDTypeContainerType,
    _categorical_left_mover,
    _check_and_to_array,
    _get_columns_to_encode,
    has_object_columns,
    reduce_dataset_size_if_too_large
)
from autoPyTorch.utils.common import ispandas
from autoPyTorch.utils.logging_ import PicklableClientLogger


def _create_column_transformer(
    preprocessors: Dict[str, List[BaseEstimator]],
    categorical_columns: List[str],
) -> ColumnTransformer:
    """
    Given a dictionary of preprocessors, this function
    creates a sklearn column transformer with appropriate
    columns associated with their preprocessors.

    Args:
        preprocessors (Dict[str, List[BaseEstimator]]):
            Dictionary containing list of numerical and categorical preprocessors.
        categorical_columns (List[str]):
            List of names of categorical columns

    Returns:
        ColumnTransformer
    """

    categorical_pipeline = make_pipeline(*preprocessors['categorical'])

    return ColumnTransformer([
        ('categorical_pipeline', categorical_pipeline, categorical_columns)],
        remainder='passthrough'
    )


def get_tabular_preprocessors() -> Dict[str, List[BaseEstimator]]:
    """
    This function creates a Dictionary containing a list
    of numerical and categorical preprocessors

    Returns:
        Dict[str, List[BaseEstimator]]
    """
    preprocessors: Dict[str, List[BaseEstimator]] = dict()

    # Categorical Preprocessors
    onehot_encoder = preprocessing.OrdinalEncoder(handle_unknown='use_encoded_value',
                                                  unknown_value=-1)
    categorical_imputer = SimpleImputer(strategy='constant', copy=False)

    preprocessors['categorical'] = [categorical_imputer, onehot_encoder]

    return preprocessors


class TabularFeatureValidator(BaseFeatureValidator):
    """
    A subclass of `BaseFeatureValidator` made for tabular data.
    It ensures that the dataset provided is of the expected format.
    Subsequently, it preprocesses the data by fitting a column
    transformer.

    Attributes:
        categories (List[List[str]]):
            List for which an element at each index is a
            list containing the categories for the respective
            categorical column.
        enc_columns (List[str])
            List of columns that were transformed.
        column_transformer (Optional[BaseEstimator])
            Hosts an imputer and an encoder object if the data
            requires transformation (for example, if provided a
            categorical column in a pandas DataFrame)
        column_order (List[str]):
            List of the features stored in the order that
            was fitted.
        numerical_columns (List[int]):
            List of indices of numerical columns
        categorical_columns (List[int]):
            List of indices of categorical columns
    """
    def __init__(
        self,
        logger: Optional[Union[PicklableClientLogger, Logger]] = None,
        dataset_compression: Optional[Mapping[str, Any]] = None,
        dtypes: Optional[Dict[str, str]] = None,
    ) -> None:
        super().__init__(logger)
        self._dataset_compression = dataset_compression
        self._reduced_dtype: Optional[DatasetDTypeContainerType] = None
        self.all_nan_columns: Optional[Set[str]] = None
        self.dtypes = dtypes if dtypes is not None else {}
        self._called_infer_object = False

    def _convert_all_nan_columns_to_numeric(self, X: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
        """
        Convert columns whose values were all nan in the training dataset to numeric.

        Args:
            X (pd.DataFrame):
                The data to transform.
            fit (bool):
                Whether this call is the fit to X or the transform using pre-fitted transformer.
        """
        if not fit and not issparse(X) and self.all_nan_columns is None:
            raise ValueError('_fit must be called before calling transform')

        if fit:
            all_nan_columns = X.columns[X.isna().all()]
        else:
            assert self.all_nan_columns is not None
            all_nan_columns = list(self.all_nan_columns)

        for col in all_nan_columns:
            X[col] = np.nan
            X[col] = pd.to_numeric(X[col])
            if len(self.dtypes):
                self.dtypes[col] = X[col].dtype.name

        if has_object_columns(X.dtypes.values):
            X = self.infer_objects(X)

        if fit:
            # TODO: Check how to integrate below
            # self.dtypes = [dt.name for dt in X.dtypes]
            self.all_nan_columns = set(all_nan_columns)

        return X

    @staticmethod
    def _comparator(cmp1: str, cmp2: str) -> int:
        return _categorical_left_mover(cmp1, cmp2)

    def _encode_categories(self, X: pd.DataFrame) -> None:
        preprocessors = get_tabular_preprocessors()
        self.column_transformer = _create_column_transformer(
            preprocessors=preprocessors,
            categorical_columns=self.enc_columns,
        )

        assert self.column_transformer is not None  # Mypy redefinition
        self.column_transformer.fit(X)

        # The column transformer moves categoricals to the left side
        assert self.feat_type is not None
        self.feat_type = sorted(self.feat_type, key=functools.cmp_to_key(self._comparator))

        encoded_categories = self.column_transformer.\
            named_transformers_['categorical_pipeline'].\
            named_steps['ordinalencoder'].categories_

        # An ordinal encoder for each categorical columns
        self.categories = [
            list(range(len(cat)))
            for cat in encoded_categories
        ]

    def _fit(self, X: SupportedFeatTypes) -> BaseEstimator:
        """
        In case input data is a pandas DataFrame, this utility encodes the user provided
        features (from categorical for example) to a numerical value that further stages
        will be able to use

        Args:
            X (SupportedFeatTypes):
                A set of features that are going to be validated (type and dimensionality
                checks) and an encoder fitted in the case the data needs encoding

        Returns:
            self:
                The fitted base estimator
        """

        # The final output of a validator is a numpy array. But pandas
        # gives us information about the column dtype
        if isinstance(X, np.ndarray):
            X = self.numpy_to_pandas(X)

        if ispandas(X) and not issparse(X):
            X = cast(pd.DataFrame, X)
            X = self._convert_all_nan_columns_to_numeric(X, fit=True)
            self.enc_columns, self.feat_type = self._get_columns_to_encode(X)

            assert self.feat_type is not None
            if len(self.enc_columns) > 0:
                self._encode_categories(X)

            for i, type_name in enumerate(self.feat_type):
                if ColumnDTypes.numerical in type_name:
                    self.numerical_columns.append(i)
                else:
                    self.categorical_columns.append(i)

        self.num_features = np.shape(X)[1]
        return self

    def transform(self, X: SupportedFeatTypes) -> Union[np.ndarray, spmatrix, pd.DataFrame]:
        """
        Validates and fit a categorical encoder (if needed) to the features.
        The supported data types are List, numpy arrays and pandas DataFrames.

        Args:
            X_train (SupportedFeatTypes):
                A set of features, whose categorical features are going to be
                transformed

        Return:
            np.ndarray:
                The transformed array

        Note:
            The default transform performs the folloing:
                * simple imputation for both
                * scaling for numerical
                * one-hot encoding for categorical
            For example, here is a simple case
            of which all the columns are categorical.
                data = [
                    {'A': 1, 'B': np.nan, 'C': np.nan},
                    {'A': np.nan, 'B': 3, 'C': np.nan},
                    {'A': 2, 'B': np.nan, 'C': np.nan}
                ]
            and suppose all the columns are categorical,
            then
                * `A` in {np.nan, 1, 2}
                * `B` in {np.nan, 3}
                * `C` in {np.nan} <=== it will be dropped.

            So in the column A,
                * np.nan ==> [1, 0, 0] (always the index 0)
                * 1      ==> [0, 1, 0]
                * 2      ==> [0, 0, 1]
            in the column B,
                * np.nan ==> [1, 0]
                * 3      ==> [0, 1]
            Therefore, by concatenating,
                * {'A': 1, 'B': np.nan, 'C': np.nan} ==> [0, 1, 0, 1, 0]
                * {'A': np.nan, 'B': 3, 'C': np.nan} ==> [1, 0, 0, 0, 1]
                * {'A': 2, 'B': np.nan, 'C': np.nan} ==> [0, 0, 1, 1, 0]
                ==> [
                    [0, 1, 0, 1, 0],
                    [1, 0, 0, 0, 1],
                    [0, 0, 1, 1, 0]
                ]
        """
        if not self._is_fitted:
            raise NotFittedError("Cannot call transform on a validator that is not fitted")

        # If a list was provided, it will be converted to pandas
        if isinstance(X, list):
            X = self.list_to_pandas(X)
        elif isinstance(X, np.ndarray):
            X = self.numpy_to_pandas(X)

        if ispandas(X) and not issparse(X):
            X = self._convert_all_nan_columns_to_numeric(X)

        # Check the data here so we catch problems on new test data
        self._check_data(X)

        # Pandas related transformations
        if ispandas(X) and self.column_transformer is not None:
            X = self.column_transformer.transform(X)

        # Sparse related transformations
        # Not all sparse format support index sorting
        if issparse(X) and hasattr(X, 'sort_indices'):
            X.sort_indices()

        try:
            X = _check_and_to_array(X)
        except Exception as e:
            self.logger.exception(
                f"Conversion failed for input {X.dtypes} {X}"
                "This means AutoPyTorch was not able to properly "
                "Extract the dtypes of the provided input features. "
                "Please try to manually cast it to a supported "
                "numerical or categorical values."
            )
            raise e

        X = self._compress_dataset(X)

        return X

    # TODO: modify once we have added subsampling as well.
    def _compress_dataset(self, X: DatasetCompressionInputType) -> DatasetCompressionInputType:
        """
        Compress the dataset. This function ensures that
        the testing data is converted to the same dtype as
        the training data.

        Args:
            X (DatasetCompressionInputType):
                Dataset

        Returns:
            DatasetCompressionInputType:
                Compressed dataset.
        """
        is_dataframe = ispandas(X)
        is_reducible_type = isinstance(X, np.ndarray) or issparse(X) or is_dataframe
        if not is_reducible_type or self._dataset_compression is None:
            return X
        elif self._reduced_dtype is not None:
            X = X.astype(self._reduced_dtype)
            return X
        else:
            X = reduce_dataset_size_if_too_large(X, **self._dataset_compression)
            self._reduced_dtype = dict(X.dtypes) if is_dataframe else X.dtype
            return X

    def _check_dataframe(self, X: pd.DataFrame) -> None:
        err_msg = " of the features must be identical before/after fit(), "
        err_msg += "but different between training and test datasets:\n"

        # Define the column to be encoded as the feature validator is fitted once per estimator
        self.enc_columns, self.feat_type = self._get_columns_to_encode(X)

        column_order = [column for column in X.columns]
        if len(self.column_order) == 0:
            self.column_order = column_order
        elif self.column_order != column_order:
            raise ValueError(f"The column order{err_msg}train: {self.column_order}\ntest: {column_order}")

        dtypes = {col: dtype.name for col, dtype in zip(X.columns, X.dtypes)}
        if len(self.dtypes) == 0:
            self.dtypes = dtypes
        elif self.dtypes != dtypes:
            raise ValueError(f"The dtypes{err_msg}train: {self.dtypes}\ntest: {dtypes}")

    def _check_data(self, X: SupportedFeatTypes) -> None:
        """
        Feature dimensionality and data type checks

        Args:
            X (SupportedFeatTypes):
                A set of features that are going to be validated (type and dimensionality
                checks) and an encoder fitted in the case the data needs encoding
        """

        if not isinstance(X, (np.ndarray, pd.DataFrame)) and not issparse(X):
            raise TypeError(
                "AutoPyTorch only supports numpy.ndarray, pandas.DataFrame,"
                f" scipy.sparse and List, but got {type(X)}"
            )

        if self.data_type is None:
            self.data_type = type(X)
        if self.data_type != type(X):
            self.logger.warning(
                f"AutoPyTorch previously received features of type {str(self.data_type)}, "
                f"but got type {str(type(X))} in the current features. This change might cause problems"
            )

        if ispandas(X):  # For pandas, no support of nan in categorical cols
            X = cast(pd.DataFrame, X)
            self._check_dataframe(X)

        # For ndarray, no support of category/string
        if isinstance(X, np.ndarray) and not np.issubdtype(X.dtype.type, np.number):
            dt = X.dtype.type
            raise TypeError(
                f"AutoPyTorch does not support numpy.ndarray with non-numerical dtype, but got {dt}"
            )

    def _get_columns_to_encode(self, X: pd.DataFrame) -> Tuple[List[str], Dict[str, str]]:
        """
        Return the columns to be encoded from a pandas dataframe

        Args:
            X (pd.DataFrame)
                A set of features that are going to be validated (type and dimensionality
                checks) and an encoder fitted in the case the data needs encoding

        Returns:
            enc_columns (List[str]):
                Columns to encode
            feat_type (Dict[str, str]):
                Whether each column is numerical or categorical
        """

        if len(self.enc_columns) > 0 and self.feat_type is not None:
            return self.enc_columns, self.feat_type
        else:
            return _get_columns_to_encode(X)

    def list_to_pandas(self, X: SupportedFeatTypes) -> pd.DataFrame:
        """
        Convert a list to a pandas DataFrame. In this process, column types are inferred.

        Args:
            X (SupportedFeatTypes):
                A set of features that are going to be validated (type and dimensionality
                checks) and an encoder fitted in the case the data needs encoding

        Returns:
            pd.DataFrame:
                transformed data from list to pandas DataFrame
        """

        # If a list was provided, it will be converted to pandas
        X = pd.DataFrame(data=X).infer_objects()
        data_info = [(col, t) for col, t in zip(X.columns, X.dtypes)]
        self.logger.warning(
            "The provided feature types to AutoPyTorch are list."
            f"Features have been interpreted as: {data_info}"
        )
        return X

    def numpy_to_pandas(self, X: np.ndarray) -> pd.DataFrame:
        """
        Converts a numpy array to pandas for type inference

        Args:
            X (np.ndarray):
                data to be interpreted.

        Returns:
            pd.DataFrame
        """
        return pd.DataFrame(X).infer_objects().convert_dtypes()

    def infer_objects(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        In case the input contains object columns, their type is inferred if possible

        This has to be done once, so the test and train data are treated equally

        Args:
            X (pd.DataFrame):
                data to be interpreted.

        Returns:
            pd.DataFrame
        """
        if self._called_infer_object:
            # honor the training data types
            try:
                # Mypy does not process the has attr.
                X = X.astype(self.dtypes)  # type: ignore[has-type]
            except Exception as e:
                self.logger.warning(
                    'Casting the columns to training dtypes '
                    f'{self.dtypes} caused the exception {e}'  # type: ignore[has-type]
                )
        elif len(self.dtypes):  # Overwrite the dtypes in test data by those in the training data
            X = X.astype(self.dtypes)
        else:  # Calling for the first time to infer the categories
            X = X.infer_objects()
            cat_dtypes = {col: 'category' for col, dtype in zip(X.columns, X.dtypes) if not is_numeric_dtype(dtype)}
            X = X.astype(cat_dtypes)

        self.dtypes.update({col: dtype.name for col, dtype in zip(X.columns, X.dtypes)})
        self.logger.debug(f"New dtypes of data: {self.dtypes}")
        self._called_infer_object = True

        return X
