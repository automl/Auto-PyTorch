import functools
from logging import Logger
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union, cast

import numpy as np

import pandas as pd
from pandas.api.types import is_numeric_dtype

from scipy.sparse import issparse, spmatrix

import sklearn.utils
from sklearn import preprocessing
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import NotFittedError
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline

from autoPyTorch.data.base_feature_validator import BaseFeatureValidator, SupportedFeatTypes
from autoPyTorch.data.utils import (
    DatasetCompressionInputType,
    DatasetDTypeContainerType,
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


def _error_due_to_unsupported_column(X: pd.DataFrame, column: str) -> None:
    # Move away from np.issubdtype as it causes
    # TypeError: data type not understood in certain pandas types
    def _generate_error_message_prefix(type_name: str, proc_type: Optional[str] = None) -> str:
        msg1 = f"column `{column}` has an invalid type `{type_name}`. "
        msg2 = "Cast it to a numerical type, category type or bool type by astype method. "
        msg3 = f"The following link might help you to know {proc_type} processing: "
        return msg1 + msg2 + ("" if proc_type is None else msg3)

    dtype = X[column].dtype
    if dtype.name == 'object':
        err_msg = _generate_error_message_prefix(type_name="object", proc_type="string")
        url = "https://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html"
        raise TypeError(f"{err_msg}{url}")
    elif pd.core.dtypes.common.is_datetime_or_timedelta_dtype(dtype):
        err_msg = _generate_error_message_prefix(type_name="time and/or date datatype", proc_type="datetime")
        raise TypeError(f"{err_msg}https://stats.stackexchange.com/questions/311494/")
    else:
        err_msg = _generate_error_message_prefix(type_name=dtype.name)
        raise TypeError(err_msg)


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
            X = self.numpy_array_to_pandas(X)

        if ispandas(X) and not issparse(X):
            X = cast(pd.DataFrame, X)
            # Treat a column with all instances a NaN as numerical
            # This will prevent doing encoding to a categorical column made completely
            # out of nan values -- which will trigger a fail, as encoding is not supported
            # with nan values.
            # Columns that are completely made of NaN values are provided to the pipeline
            # so that later stages decide how to handle them
            if np.any(pd.isnull(X)):
                for column in X.columns:
                    if X[column].isna().all():
                        X[column] = pd.to_numeric(X[column])
                        # Also note this change in self.dtypes
                        if len(self.dtypes) != 0:
                            self.dtypes[list(X.columns).index(column)] = X[column].dtype

            if not X.select_dtypes(include='object').empty:
                X = self.infer_objects(X)

            self.transformed_columns, self.feat_type = self._get_columns_to_encode(X)

            assert self.feat_type is not None

            if len(self.transformed_columns) > 0:

                preprocessors = get_tabular_preprocessors()
                self.column_transformer = _create_column_transformer(
                    preprocessors=preprocessors,
                    categorical_columns=self.transformed_columns,
                )

                # Mypy redefinition
                assert self.column_transformer is not None
                self.column_transformer.fit(X)

                # The column transformer reorders the feature types
                # therefore, we need to change the order of columns as well
                # This means categorical columns are shifted to the left
                self.feat_type = sorted(
                    self.feat_type,
                    key=functools.cmp_to_key(self._comparator)
                )

                encoded_categories = self.column_transformer.\
                    named_transformers_['categorical_pipeline'].\
                    named_steps['ordinalencoder'].categories_
                self.categories = [
                    # We fit an ordinal encoder, where all categorical
                    # columns are shifted to the left
                    list(range(len(cat)))
                    for cat in encoded_categories
                ]

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
        """
        if not self._is_fitted:
            raise NotFittedError("Cannot call transform on a validator that is not fitted")

        # If a list was provided, it will be converted to pandas
        if isinstance(X, list):
            X, _ = self.list_to_dataframe(X)

        if isinstance(X, np.ndarray):
            X = self.numpy_array_to_pandas(X)

        if ispandas(X) and not issparse(X):
            if np.any(pd.isnull(X)):
                for column in X.columns:
                    if X[column].isna().all():
                        X[column] = pd.to_numeric(X[column])

            # Also remove the object dtype for new data
            if not X.select_dtypes(include='object').empty:
                X = self.infer_objects(X)

        # Check the data here so we catch problems on new test data
        self._check_data(X)

        # Pandas related transformations
        if ispandas(X) and self.column_transformer is not None:
            if np.any(pd.isnull(X)):
                # After above check it means that if there is a NaN
                # the whole column must be NaN
                # Make sure it is numerical and let the pipeline handle it
                for column in X.columns:
                    if X[column].isna().all():
                        X[column] = pd.to_numeric(X[column])

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
        if ispandas(X):
            # If entered here, we have a pandas dataframe
            X = cast(pd.DataFrame, X)

            # Handle objects if possible
            if not X.select_dtypes(include='object').empty:
                X = self.infer_objects(X)

            # Define the column to be encoded here as the feature validator is fitted once
            # per estimator
            self.transformed_columns, self.feat_type = self._get_columns_to_encode(X)

            column_order = [column for column in X.columns]
            if len(self.column_order) > 0:
                if self.column_order != column_order:
                    raise ValueError("Changing the column order of the features after fit() is "
                                     "not supported. Fit() method was called with "
                                     "{} whereas the new features have {} as type".format(self.column_order,
                                                                                          column_order,)
                                     )
            else:
                self.column_order = column_order

            dtypes = [dtype.name for dtype in X.dtypes]
            if len(self.dtypes) > 0:
                if self.dtypes != dtypes:
                    raise ValueError("Changing the dtype of the features after fit() is "
                                     "not supported. Fit() method was called with "
                                     "{} whereas the new features have {} as type".format(self.dtypes,
                                                                                          dtypes,
                                                                                          )
                                     )
            else:
                self.dtypes = dtypes

    def _get_columns_to_encode(
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
            transformed_columns (List[str]):
                Columns to encode, if any
            feat_type:
                Type of each column numerical/categorical
        """

        if len(self.transformed_columns) > 0 and self.feat_type is not None:
            return self.transformed_columns, self.feat_type

        # Register if a column needs encoding
        transformed_columns = []

        # Also, register the feature types for the estimator
        feat_type = []

        # Make sure each column is a valid type
        for dtype, column in zip(X.dtypes, X.columns):
            if dtype.name in ['category', 'bool']:
                transformed_columns.append(column)
                feat_type.append('categorical')
            elif is_numeric_dtype(dtype):
                feat_type.append('numerical')
            else:
                _error_due_to_unsupported_column(X, column)

        return transformed_columns, feat_type

    def list_to_dataframe(
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
        X_train = pd.DataFrame(data=X_train).infer_objects()
        self.logger.warning("The provided feature types to AutoPyTorch are of type list."
                            "Features have been interpreted as: {}".format([(col, t) for col, t in
                                                                            zip(X_train.columns, X_train.dtypes)]))
        if X_test is not None:
            if not isinstance(X_test, list):
                self.logger.warning("Train features are a list while the provided test data"
                                    "is {}. X_test will be casted as DataFrame.".format(type(X_test))
                                    )
            X_test = pd.DataFrame(data=X_test).infer_objects()
        return X_train, X_test

    def numpy_array_to_pandas(
        self,
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
        if hasattr(self, 'object_dtype_mapping'):
            # Mypy does not process the has attr. This dict is defined below
            for key, dtype in self.object_dtype_mapping.items():  # type: ignore[has-type]
                if 'int' in dtype.name:
                    # In the case train data was interpreted as int
                    # and test data was interpreted as float, because of 0.0
                    # for example, honor training data
                    X[key] = X[key].applymap(np.int64)
                else:
                    try:
                        X[key] = X[key].astype(dtype.name)
                    except Exception as e:
                        # Try inference if possible
                        self.logger.warning(f"Tried to cast column {key} to {dtype} caused {e}")
                        pass
        else:
            X = X.infer_objects()
            for column in X.columns:
                if not is_numeric_dtype(X[column]):
                    X[column] = X[column].astype('category')
            self.object_dtype_mapping = {column: X[column].dtype for column in X.columns}
        self.logger.debug(f"Infer Objects: {self.object_dtype_mapping}")
        return X
