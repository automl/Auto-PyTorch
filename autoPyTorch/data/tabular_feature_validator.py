import functools
from logging import Logger
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union, cast

import numpy as np

import pandas as pd
from pandas.api.types import is_numeric_dtype

from scipy.sparse import issparse, spmatrix

import sklearn.utils
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import NotFittedError
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OrdinalEncoder

from autoPyTorch.data.base_feature_validator import BaseFeatureValidator, SupportedFeatTypes
from autoPyTorch.data.utils import (
    DatasetCompressionInputType,
    DatasetDTypeContainerType,
    reduce_dataset_size_if_too_large
)
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
    ordinal_encoder = OrdinalEncoder(handle_unknown='use_encoded_value',
                                     unknown_value=-1)
    categorical_imputer = SimpleImputer(strategy='constant', copy=False)

    preprocessors['categorical'] = [categorical_imputer, ordinal_encoder]

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
        transformed_columns (List[str])
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
    ) -> None:
        self._dataset_compression = dataset_compression
        self._reduced_dtype: Optional[DatasetDTypeContainerType] = None
        super().__init__(logger)

    @staticmethod
    def _comparator(cmp1: str, cmp2: str) -> int:
        """Order so that categorical columns come left and numerical columns come right

        Args:
            cmp1 (str): First variable to compare
            cmp2 (str): Second variable to compare

        Raises:
            ValueError: if the values of the variables to compare
            are not in 'categorical' or 'numerical'

        Returns:
            int: either [0, -1, 1]
        """
        choices = ['categorical', 'numerical']
        if cmp1 not in choices or cmp2 not in choices:
            raise ValueError('The comparator for the column order only accepts {}, '
                             'but got {} and {}'.format(choices, cmp1, cmp2))

        idx1, idx2 = choices.index(cmp1), choices.index(cmp2)
        return idx1 - idx2

    def _fit(
        self,
        X: SupportedFeatTypes,
    ) -> BaseEstimator:
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
            # Replace the data type from the previously saved type.
            self.data_type = type(X)
            # save all the information about the column order and data types
            self._check_data(X)

        if hasattr(X, "iloc") and not issparse(X):
            X = cast(pd.DataFrame, X)

            all_nan_columns = X.columns[X.isna().all()]
            for col in all_nan_columns:
                X[col] = pd.to_numeric(X[col])

            # Handle objects if possible
            exist_object_columns = has_object_columns(X.dtypes.values)
            if exist_object_columns:
                X = self.infer_objects(X)

            self.dtypes = [dt.name for dt in X.dtypes]  # Also note this change in self.dtypes
            self.all_nan_columns = set(all_nan_columns)

            self.enc_columns, self.feat_type = self._get_columns_info(X)

            if len(self.enc_columns) > 0:

                preprocessors = get_tabular_preprocessors()
                self.column_transformer = _create_column_transformer(
                    preprocessors=preprocessors,
                    categorical_columns=self.enc_columns,
                )

                # Mypy redefinition
                assert self.column_transformer is not None
                self.column_transformer.fit(X)

                # The column transformer moves categorical columns before all numerical columns
                # therefore, we need to sort categorical columns so that it complies this change

                self.feat_type = sorted(
                    self.feat_type,
                    key=functools.cmp_to_key(self._comparator)
                )

                encoded_categories = self.column_transformer.\
                    named_transformers_['categorical_pipeline'].\
                    named_steps['ordinalencoder'].categories_
                self.categories = [
                    list(range(len(cat)))
                    for cat in encoded_categories
                ]

            # differently to categorical_columns and numerical_columns,
            # this saves the index of the column.
            for i, type_ in enumerate(self.feat_type):
                if 'numerical' in type_:
                    self.numerical_columns.append(i)
                else:
                    self.categorical_columns.append(i)

        # Lastly, store the number of features
        self.num_features = np.shape(X)[1]

        return self

    def transform(
        self,
        X: SupportedFeatTypes,
    ) -> Union[np.ndarray, spmatrix, pd.DataFrame]:
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
            X, _ = self.list_to_pandas(X)

        if isinstance(X, np.ndarray):
            X = self.numpy_to_pandas(X)

        if hasattr(X, "iloc") and not scipy.sparse.issparse(X):
            X = cast(Type[pd.DataFrame], X)

            if self.all_nan_columns is None:
                raise ValueError('_fit must be called before calling transform')

            for col in list(self.all_nan_columns):
                X[col] = np.nan
                X[col] = pd.to_numeric(X[col])

        if len(self.categorical_columns) > 0:
            # when some categorical columns are not all nan in the training set
            # but they are all nan in the testing or validation set
            # we change those columns to `object` dtype
            # to ensure that these columns are changed to appropriate dtype
            # in self.infer_objects
            all_nan_cat_cols = set(X[self.enc_columns].columns[X[self.enc_columns].isna().all()])
            dtype_dict = {col: 'object' for col in self.enc_columns if col in all_nan_cat_cols}
            X = X.astype(dtype_dict)

        # Check the data here so we catch problems on new test data
        self._check_data(X)

        # in case of test data being all none and train data
        # having a value for a categorical column.
        # We need to convert the column in test data to
        # object otherwise the test column is interpreted as float
        if self.column_transformer is not None:
            X = self.column_transformer.transform(X)

        # Sparse related transformations
        # Not all sparse format support index sorting
        if issparse(X) and hasattr(X, 'sort_indices'):
            X.sort_indices()

        try:
            X = sklearn.utils.check_array(
                X,
                force_all_finite=False,
                accept_sparse='csr'
            )
        except Exception as e:
            self.logger.exception(f"Conversion failed for input {X.dtypes} {X}"
                                  "This means AutoPyTorch was not able to properly "
                                  "Extract the dtypes of the provided input features. "
                                  "Please try to manually cast it to a supported "
                                  "numerical or categorical values.")
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
        is_dataframe = hasattr(X, 'iloc')
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

    def _check_data(
        self,
        X: SupportedFeatTypes,
    ) -> None:
        """
        Feature dimensionality and data type checks

        Args:
            X (SupportedFeatTypes):
                A set of features that are going to be validated (type and dimensionality
                checks) and an encoder fitted in the case the data needs encoding
        """

        if not isinstance(X, (np.ndarray, pd.DataFrame)) and not issparse(X):
            raise ValueError("AutoPyTorch only supports Numpy arrays, Pandas DataFrames,"
                             " scipy sparse and Python Lists, yet, the provided input is"
                             " of type {}".format(type(X))
                             )

        if self.data_type is None:
            self.data_type = type(X)
        if self.data_type != type(X):
            self.logger.warning("AutoPyTorch previously received features of type %s "
                                "yet the current features have type %s. Changing the dtype "
                                "of inputs to an estimator might cause problems" % (
                                    str(self.data_type),
                                    str(type(X)),
                                ),
                                )

        # Do not support category/string numpy data. Only numbers
        if hasattr(X, "dtype"):
            if not np.issubdtype(X.dtype.type, np.number):  # type: ignore[union-attr]
                raise ValueError(
                    "When providing a numpy array to AutoPyTorch, the only valid "
                    "dtypes are numerical ones. The provided data type {} is not supported."
                    "".format(
                        X.dtype.type,  # type: ignore[union-attr]
                    )
                )

        # Then for Pandas, we do not support Nan in categorical columns
        if hasattr(X, "iloc"):
            # If entered here, we have a pandas dataframe
            X = cast(pd.DataFrame, X)

            # Handle objects if possible
            exist_object_columns = has_object_columns(X.dtypes.values)
            if exist_object_columns:
                X = self.infer_objects(X)

            column_order = [column for column in X.columns]
            if len(self.column_order) > 0:
                if self.column_order != column_order:
                    raise ValueError("The column order of the features must not be changed after fit(), but"
                                     " the column order are different between training ({}) and"
                                     " test ({}) datasets.".format(self.column_order, column_order))
            else:
                self.column_order = column_order

            dtypes = [dtype.name for dtype in X.dtypes]
            diff_cols = X.columns[[s_dtype != dtype for s_dtype, dtype in zip(self.dtypes, dtypes)]]
            if len(self.dtypes) == 0:
                self.dtypes = dtypes
            elif not self._is_datasets_consistent(diff_cols, X):
                raise ValueError("The dtype of the features must not be changed after fit(), but"
                                 " the dtypes of some columns are different between training ({}) and"
                                 " test ({}) datasets.".format(self.dtypes, dtypes))

    def _get_columns_info(
        self,
        X: pd.DataFrame,
    ) -> Tuple[List[str], List[str]]:
        """
        Return the columns to be encoded from a pandas dataframe

        Args:
            X (pd.DataFrame)
                A set of features that are going to be validated (type and dimensionality
                checks) and an encoder fitted in the case the data needs encoding

        Returns:
            categorical_columns (List[str])
                List of the names of categorical columns.
            numerical_columns (List[str])
                List of the names of numerical columns.
            feat_type (List[str])
                Type of each column numerical/categorical
        """

        # Register if a column needs encoding
        categorical_columns = []
        # Also, register the feature types for the estimator
        feat_type = []

        # Make sure each column is a valid type
        for i, column in enumerate(X.columns):
            column_dtype = self.dtypes[i]
            err_msg = "Valid types are `numerical`, `categorical` or `boolean`, " \
                      "but input column {} has an invalid type `{}`.".format(column, column_dtype)
            if column_dtype in ['category', 'bool']:
                categorical_columns.append(column)
                feat_type.append('categorical')
            # Move away from np.issubdtype as it causes
            # TypeError: data type not understood in certain pandas types
            elif is_numeric_dtype(column_dtype):
                feat_type.append('numerical')
            elif column_dtype == 'object':
                # TODO verify how would this happen when we always convert the object dtypes to category
                raise TypeError(
                    "{} Cast it to a valid dtype before feeding it to AutoPyTorch. "
                    "You can cast it to a valid dtype using pandas.Series.astype."
                    "If you are working with string objects, the following "
                    "tutorial illustrates how to work with text data: "
                    "https://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html".format(
                        # noqa: E501
                        err_msg,
                    )
                )
            elif pd.core.dtypes.common.is_datetime_or_timedelta_dtype(column_dtype):
                raise TypeError(
                    "{} Convert the time information to a numerical value"
                    " before feeding it to AutoPyTorch. "
                    "One example of the conversion can be found on "
                    "https://stats.stackexchange.com/questions/311494/".format(err_msg)
                )
            else:
                raise TypeError(
                    "{} Make sure your data is formatted in a correct way"
                    "before feeding it to AutoPyTorch.".format(err_msg)
                )

        return categorical_columns, feat_type

    def list_to_pandas(
        self,
        X_train: SupportedFeatTypes,
        X_test: Optional[SupportedFeatTypes] = None,
    ) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        """
        Converts a list to a pandas DataFrame. In this process, column types are inferred.

        If test data is provided, we proactively match it to train data

        Args:
            X_train (SupportedFeatTypes):
                A set of features that are going to be validated (type and dimensionality
                checks) and a encoder fitted in the case the data needs encoding
            X_test (Optional[SupportedFeatTypes]):
                A hold out set of data used for checking

        Returns:
            pd.DataFrame:
                transformed train data from list to pandas DataFrame
            pd.DataFrame:
                transformed test data from list to pandas DataFrame
        """

        # If a list was provided, it will be converted to pandas
        X_train = pd.DataFrame(data=X_train).convert_dtypes()
        self.logger.warning("The provided feature types to AutoPyTorch are of type list."
                            "Features have been interpreted as: {}".format([(col, t) for col, t in
                                                                            zip(X_train.columns, X_train.dtypes)]))
        if X_test is not None:
            if not isinstance(X_test, list):
                self.logger.warning("Train features are a list while the provided test data"
                                    "is {}. X_test will be casted as DataFrame.".format(type(X_test))
                                    )
            X_test = pd.DataFrame(data=X_test).convert_dtypes()

        return X_train, X_test

    @staticmethod
    def numpy_to_pandas(
        X: np.ndarray,
    ) -> pd.DataFrame:
        """
        Converts a numpy array to pandas for type inference

        Args:
            X (np.ndarray):
                data to be interpreted.

        Returns:
            pd.DataFrame
        """
        return pd.DataFrame(X).convert_dtypes()

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
        if hasattr(self, 'object_dtype_mapping'):
            # honor the training data types
            try:
                # Mypy does not process the has attr.
                X = X.astype(self.object_dtype_mapping)  # type: ignore[has-type]
            except Exception as e:
                # Try inference if possible
                self.logger.warning(f'Casting the columns to training dtypes '  # type: ignore[has-type]
                                    f'{self.object_dtype_mapping} caused the exception {e}')
                pass
        else:
            if len(self.dtypes) != 0:
                # when train data has no object dtype, but test does
                # we prioritise the datatype given in training data
                dtype_dict = {col: dtype for col, dtype in zip(X.columns, self.dtypes)}
                X = X.astype(dtype_dict)
            else:
                # Calling for the first time to infer the categories
                X = X.infer_objects()
                dtype_dict = {col: 'category' for col, dtype in zip(X.columns, X.dtypes) if not is_numeric_dtype(dtype)}
                X = X.astype(dtype_dict)
            # only numerical attributes and categories
            self.object_dtype_mapping = {column: data_type for column, data_type in zip(X.columns, X.dtypes)}

        self.logger.debug(f"Infer Objects: {self.object_dtype_mapping}")

        return X

    def _is_datasets_consistent(self, diff_cols: List[Union[int, str]], X: pd.DataFrame) -> bool:
        """
        Check the consistency of dtypes between training and test datasets.
        The dtypes can be different if the column belongs to `self.all_nan_columns`
        (list of column names with all nans in training data) or if the column is
        all nan as these columns would be imputed.

        Args:
            diff_cols (List[bool]):
                The column labels that have different dtypes.
            X (pd.DataFrame):
                A validation or test dataset to be compared with the training dataset
        Returns:
            _ (bool): Whether the training and test datasets are consistent.
        """
        if self.all_nan_columns is None:
            if len(diff_cols) == 0:
                return True
            else:
                return all(X[diff_cols].isna().all())

        # dtype is different ==> the column in at least either of train or test datasets must be all NaN
        # inconsistent <==> dtype is different and the col in both train and test is not all NaN
        inconsistent_cols = list(set(diff_cols) - self.all_nan_columns)

        return len(inconsistent_cols) == 0 or all(X[inconsistent_cols].isna().all())


def has_object_columns(
    feature_types: pd.Series,
) -> bool:
    """
    Indicate whether on a Series of dtypes for a Pandas DataFrame
    there exists one or more object columns.

    Args:
        feature_types (pd.Series): The feature types for a DataFrame.

    Returns:
        bool:
            True if the DataFrame dtypes contain an object column, False
            otherwise.
    """
    return np.dtype('O') in feature_types
