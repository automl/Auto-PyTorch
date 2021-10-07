import functools
from typing import Any, Dict, List, Optional, Tuple, Union, cast

import numpy as np

import pandas as pd
from pandas.api.types import is_numeric_dtype

import scipy.sparse

import sklearn.utils
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import NotFittedError
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from autoPyTorch.data.base_feature_validator import BaseFeatureValidator, SUPPORTED_FEAT_TYPES


def _create_column_transformer(
    preprocessors: Dict[str, List[BaseEstimator]],
    numerical_columns: List[str],
    categorical_columns: List[str],
) -> ColumnTransformer:
    """
    Given a dictionary of preprocessors, this function
    creates a sklearn column transformer with appropriate
    columns associated with their preprocessors.
    Args:
        preprocessors (Dict[str, List]):
            Dictionary containing list of numerical and categorical preprocessors.
        numerical_columns (List[int]):
            List of names of numerical columns
        categorical_columns (List[int]):
            List of names of categorical columns
    Returns:
        ColumnTransformer
    """

    numerical_pipeline = 'drop'
    categorical_pipeline = 'drop'
    if len(numerical_columns) > 0:
        numerical_pipeline = make_pipeline(*preprocessors['numerical'])
    if len(categorical_columns) > 0:
        categorical_pipeline = make_pipeline(*preprocessors['categorical'])

    return ColumnTransformer([
        ('categorical_pipeline', categorical_pipeline, categorical_columns),
        ('numerical_pipeline', numerical_pipeline, numerical_columns)],
        remainder='passthrough'
    )


def get_tabular_preprocessors() -> Dict[str, List[BaseEstimator]]:
    """
    This function creates a Dictionary containing list
    of numerical and categorical preprocessors
    Returns:
        Dict[str, List[BaseEstimator]]
    """
    preprocessors: Dict[str, List[BaseEstimator]] = dict()

    # Categorical Preprocessors
    onehot_encoder = OneHotEncoder(categories='auto', sparse=False, handle_unknown='ignore')
    categorical_imputer = SimpleImputer(strategy='constant', copy=False)

    # Numerical Preprocessors
    numerical_imputer = SimpleImputer(strategy='median', copy=False)
    standard_scaler = StandardScaler(with_mean=True, with_std=True, copy=False)

    preprocessors['categorical'] = [categorical_imputer, onehot_encoder]
    preprocessors['numerical'] = [numerical_imputer, standard_scaler]

    return preprocessors


class TabularFeatureValidator(BaseFeatureValidator):

    def _fit(
        self,
        X: SUPPORTED_FEAT_TYPES,
    ) -> BaseEstimator:
        """
        In case input data is a pandas DataFrame, this utility encodes the user provided
        features (from categorical for example) to a numerical value that further stages
        will be able to use

        Arguments:
            X (SUPPORTED_FEAT_TYPES):
                A set of features that are going to be validated (type and dimensionality
                checks) and a encoder fitted in the case the data needs encoding
        Returns:
            self:
                The fitted base estimator
        """

        # The final output of a validator is a numpy array. But pandas
        # gives us information about the column dtype
        if isinstance(X, np.ndarray):

            X = self.numpy_array_to_pandas(X)
            # Replace the data type from the previously saved type.
            self.data_type = type(X)
            # save all the information about the column order and data types
            self._check_data(X)

        if hasattr(X, "iloc") and not scipy.sparse.issparse(X):

            X = cast(pd.DataFrame, X)
            categorical_columns, numerical_columns, feat_type = self._get_columns_info(X)
            print("enc_columns", categorical_columns)
            print("all_nan_columns", self.all_nan_columns)

            self.enc_columns = categorical_columns

            preprocessors = get_tabular_preprocessors()
            self.column_transformer = _create_column_transformer(
                preprocessors=preprocessors,
                numerical_columns=numerical_columns,
                categorical_columns=categorical_columns,
            )

            # Mypy redefinition
            assert self.column_transformer is not None
            self.column_transformer.fit(X)

            # The column transformer reoders the feature types - we therefore need to change
            # it as well
            # This means columns are shifted to the right
            def comparator(cmp1: str, cmp2: str) -> int:
                if (
                    cmp1 == 'categorical' and cmp2 == 'categorical'
                    or cmp1 == 'numerical' and cmp2 == 'numerical'
                ):
                    return 0
                elif cmp1 == 'categorical' and cmp2 == 'numerical':
                    return -1
                elif cmp1 == 'numerical' and cmp2 == 'categorical':
                    return 1
                else:
                    raise ValueError((cmp1, cmp2))

            self.feat_type = sorted(
                feat_type,
                key=functools.cmp_to_key(comparator)
            )

            if len(categorical_columns) > 0:
                self.categories = [
                    # We fit a one-hot encoder, where all categorical
                    # columns are shifted to the left
                    list(range(len(cat)))
                    for cat in self.column_transformer.named_transformers_[
                        'categorical_pipeline'].named_steps['onehotencoder'].categories_
                ]

            # differently to categorical_columns and numerical_columns,
            # this saves the index of the column.
            for i, type_ in enumerate(self.feat_type):
                if 'numerical' in type_:
                    self.numerical_columns.append(i)
                else:
                    self.categorical_columns.append(i)

        # Lastly, store the number of features
        self.num_features = len(X.columns)

        return self

    def transform(
        self,
        X: SUPPORTED_FEAT_TYPES,
    ) -> np.ndarray:
        """
        Validates and fit a categorical encoder (if needed) to the features.
        The supported data types are List, numpy arrays and pandas DataFrames.

        Arguments:
            X_train (SUPPORTED_FEAT_TYPES):
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

        if hasattr(X, "iloc") and not scipy.sparse.issparse(X):
            X = cast(pd.DataFrame, X)

        # Check the data here so we catch problems on new test data
        self._check_data(X)
    
        X = self.column_transformer.transform(X)

        # Sparse related transformations
        # Not all sparse format support index sorting
        if scipy.sparse.issparse(X) and hasattr(X, 'sort_indices'):
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
        return X

    def _check_data(
        self,
        X: SUPPORTED_FEAT_TYPES,
    ) -> None:
        """
        Feature dimensionality and data type checks

        Arguments:
            X (SUPPORTED_FEAT_TYPES):
                A set of features that are going to be validated (type and dimensionality
                checks) and a encoder fitted in the case the data needs encoding
        """

        if not isinstance(X, (np.ndarray, pd.DataFrame)) and not scipy.sparse.issparse(X):
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

            # we should remove columns with all nans in the training set.
            if hasattr(self, 'all_nan_columns') and set(self.all_nan_columns).issubset(X.columns):
                X.drop(labels=self.all_nan_columns, axis=1, inplace=True)
            else:
                self.all_nan_columns: List[Union[int, str]] = list()
                for column in X.columns:
                    if X[column].isna().all():
                        self.all_nan_columns.append(column)
                X.drop(labels=self.all_nan_columns, axis=1, inplace=True)

            # Handle objects if possible
            object_columns_indicator = has_object_columns(X.dtypes.values)
            if object_columns_indicator:
                X = self.infer_objects(X)

            # Define the column to be encoded here as the feature validator is fitted once
            # per estimator
            # enc_columns, _ = self._get_columns_to_encode(X)
            column_order = [column for column in X.columns]
            if len(self.column_order) > 0:
                if self.column_order != column_order:
                    raise ValueError("Changing the column order of the features after fit() is "
                                     "not supported. Fit() method was called with "
                                     "{} whereas the new features have {} as type".format(self.column_order,
                                                                                          column_order, )
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

    def _get_columns_info(
        self,
        X: pd.DataFrame,
    ) -> Tuple[List[str], List[str], List[str]]:
        """
        Return the columns to be encoded from a pandas dataframe

        Arguments:
            X (pd.DataFrame)
                A set of features that are going to be validated (type and dimensionality
                checks) and a encoder fitted in the case the data needs encoding
        Returns:
            categorical_columns: (List[str])
                Categorical columns.
            numerical_columns: (List[str])
                Numerical columns.
            feat_type:
                Type of each column numerical/categorical
        """
        # Register if a column needs encoding
        numerical_columns = []
        categorical_columns = []
        # Also, register the feature types for the estimator
        feat_type = []

        # Make sure each column is a valid type
        for i, column in enumerate(X.columns):
            column_dtype = self.dtypes[i]
            if column_dtype in ['category', 'bool']:
                categorical_columns.append(column)
                feat_type.append('categorical')
            # Move away from np.issubdtype as it causes
            # TypeError: data type not understood in certain pandas types
            elif not is_numeric_dtype(column_dtype):
                # TODO verify how would this happen when we always convert the object dtypes to category
                if column_dtype == 'object':
                    raise ValueError(
                        "Input Column {} has invalid type object. "
                        "Cast it to a valid dtype before using it in AutoPyTorch. "
                        "Valid types are numerical, categorical or boolean. "
                        "You can cast it to a valid dtype using "
                        "pandas.Series.astype ."
                        "If working with string objects, the following "
                        "tutorial illustrates how to work with text data: "
                        "https://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html".format(
                            # noqa: E501
                            column,
                        )
                    )
                elif pd.core.dtypes.common.is_datetime_or_timedelta_dtype(
                    column_dtype
                ):
                    raise ValueError(
                        "AutoPyTorch does not support time and/or date datatype as given "
                        "in column {}. Please convert the time information to a numerical value "
                        "first. One example on how to do this can be found on "
                        "https://stats.stackexchange.com/questions/311494/".format(
                            column,
                        )
                    )
                else:
                    raise ValueError(
                        "Input Column {} has unsupported dtype {}. "
                        "Supported column types are categorical/bool/numerical dtypes. "
                        "Make sure your data is formatted in a correct way, "
                        "before feeding it to AutoPyTorch.".format(
                            column,
                            column_dtype,
                        )
                    )
            else:
                feat_type.append('numerical')
                numerical_columns.append(column)
        return categorical_columns, numerical_columns, feat_type

    def list_to_dataframe(
        self,
        X_train: SUPPORTED_FEAT_TYPES,
        X_test: Optional[SUPPORTED_FEAT_TYPES] = None,
    ) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        """
        Converts a list to a pandas DataFrame. In this process, column types are inferred.

        If test data is provided, we proactively match it to train data

        Arguments:
            X_train (SUPPORTED_FEAT_TYPES):
                A set of features that are going to be validated (type and dimensionality
                checks) and a encoder fitted in the case the data needs encoding
            X_test (Optional[SUPPORTED_FEAT_TYPES]):
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
    def numpy_array_to_pandas(
        X: np.ndarray,
    ) -> pd.DataFrame:
        """
        Converts a numpy array to pandas for type inference

        Arguments:
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

        Arguments:
            X (pd.DataFrame):
                data to be interpreted.

        Returns:
            pd.DataFrame
        """
        if hasattr(self, 'object_dtype_mapping'):
            # Mypy does not process the has attr. This dict is defined below
            for key, dtype in self.object_dtype_mapping.items():  # type: ignore[has-type]
                # honor the training data types
                try:
                    X[key] = X[key].astype(dtype.name)
                except Exception as e:
                    # Try inference if possible
                    self.logger.warning(f"Tried to cast column {key} to {dtype} caused {e}")
                    pass
        else:
            # Calling for the first time to infer the categories
            X = X.infer_objects()
            # initial data types
            data_types = X.dtypes
            for index, column in enumerate(X.columns):
                if not is_numeric_dtype(data_types[index]):
                    X[column] = X[column].astype('category')
            # only numerical attributes and categories
            data_types = X.dtypes
            self.object_dtype_mapping = {column: data_type for column, data_type in zip(X.columns, X.dtypes)}

        self.logger.debug(f"Infer Objects: {self.object_dtype_mapping}")

        return X


def has_object_columns(
    feature_types: pd.Series,
) -> bool:
    """
    Indicate whether on a Series of dtypes for a Pandas DataFrame
    there exists one or more object columns.

    Arguments:
        feature_types (pd.Series):
            The feature types for a DataFrame.
    Returns:
        bool:
            True if the DataFrame dtypes contain an object column, False
            otherwise.
    """
    return np.dtype('O') in feature_types

