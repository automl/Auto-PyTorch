import numpy as np

import pandas as pd

import pytest

from scipy import sparse

import sklearn.datasets
import sklearn.model_selection

from autoPyTorch.data.tabular_validator import TabularInputValidator
from autoPyTorch.data.utils import megabytes


@pytest.mark.parametrize('openmlid', [2, 40975, 40984])
@pytest.mark.parametrize('as_frame', [True, False])
def test_data_validation_for_classification(openmlid, as_frame):
    x, y = sklearn.datasets.fetch_openml(data_id=openmlid, return_X_y=True, as_frame=as_frame)
    validator = TabularInputValidator(is_classification=True)

    if as_frame:
        # NaN is not supported in categories, so
        # drop columns with them.
        nan_cols = [i for i in x.columns if x[i].isnull().any()]
        cat_cols = [i for i in x.columns if x[i].dtype.name in ['category', 'bool']]
        unsupported_columns = list(set(nan_cols) & set(cat_cols))
        if len(unsupported_columns) > 0:
            x.drop(unsupported_columns, axis=1, inplace=True)

    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
        x, y, test_size=0.33, random_state=0)

    validator.fit(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)

    X_train_t, y_train_t = validator.transform(X_train, y_train)
    assert np.shape(X_train) == np.shape(X_train_t)

    # Leave columns that are complete NaN
    # The sklearn pipeline will handle that
    if as_frame and np.any(pd.isnull(X_train).values.all(axis=0)):
        assert np.any(pd.isnull(X_train_t).values.all(axis=0))
    elif not as_frame and np.any(pd.isnull(X_train).all(axis=0)):
        assert np.any(pd.isnull(X_train_t).all(axis=0))

    # make sure everything was encoded to number
    assert np.issubdtype(X_train_t.dtype, np.number)
    assert np.issubdtype(y_train_t.dtype, np.number)

    # Categorical columns are sorted to the beginning
    if as_frame:
        validator.feature_validator.feat_types is not None
        ordered_unique_elements = list(dict.fromkeys(validator.feature_validator.feat_types))
        if len(ordered_unique_elements) > 1:
            assert ordered_unique_elements[0] == 'categorical'


@pytest.mark.parametrize('openmlid', [505, 546, 531])
@pytest.mark.parametrize('as_frame', [True, False])
def test_data_validation_for_regression(openmlid, as_frame):
    x, y = sklearn.datasets.fetch_openml(data_id=openmlid, return_X_y=True, as_frame=as_frame)
    validator = TabularInputValidator(is_classification=False)

    if as_frame:
        # NaN is not supported in categories, so
        # drop columns with them.
        nan_cols = [i for i in x.columns if x[i].isnull().any()]
        cat_cols = [i for i in x.columns if x[i].dtype.name in ['category', 'bool']]
        unsupported_columns = list(set(nan_cols) & set(cat_cols))
        if len(unsupported_columns) > 0:
            x.drop(unsupported_columns, axis=1, inplace=True)

    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
        x, y, test_size=0.33, random_state=0)

    validator.fit(X_train=X_train, y_train=y_train)

    X_train_t, y_train_t = validator.transform(X_train, y_train)
    assert np.shape(X_train) == np.shape(X_train_t)

    # Leave columns that are complete NaN
    # The sklearn pipeline will handle that
    if as_frame and np.any(pd.isnull(X_train).values.all(axis=0)):
        assert np.any(pd.isnull(X_train_t).values.all(axis=0))
    elif not as_frame and np.any(pd.isnull(X_train).all(axis=0)):
        assert np.any(pd.isnull(X_train_t).all(axis=0))

    # make sure everything was encoded to number
    assert np.issubdtype(X_train_t.dtype, np.number)
    assert np.issubdtype(y_train_t.dtype, np.number)

    # Categorical columns are sorted to the beginning
    if as_frame:
        validator.feature_validator.feat_types is not None
        ordered_unique_elements = list(dict.fromkeys(validator.feature_validator.feat_types))
        if len(ordered_unique_elements) > 1:
            assert ordered_unique_elements[0] == 'categorical'


def test_sparse_data_validation_for_regression():
    X, y = sklearn.datasets.make_regression(n_samples=100, n_features=50, random_state=0)
    X_sp = sparse.coo_matrix(X)
    validator = TabularInputValidator(is_classification=False)

    validator.fit(X_train=X_sp, y_train=y)

    X_t, y_t = validator.transform(X, y)
    assert np.shape(X) == np.shape(X_t)

    # make sure everything was encoded to number
    assert np.issubdtype(X_t.dtype, np.number)
    assert np.issubdtype(y_t.dtype, np.number)

    # Make sure we can change the sparse format
    X_t, y_t = validator.transform(sparse.csr_matrix(X), y)


def test_validation_unsupported():
    """
    Makes sure we raise a proper message to the user,
    when providing not supported data input
    """
    validator = TabularInputValidator()
    with pytest.raises(ValueError, match=r"Inconsistent number of train datapoints.*"):
        validator.fit(
            X_train=np.array([[0, 1, 0], [0, 1, 1]]),
            y_train=np.array([0, 1, 0, 0, 0, 0]),
        )
    with pytest.raises(ValueError, match=r"Inconsistent number of test datapoints.*"):
        validator.fit(
            X_train=np.array([[0, 1, 0], [0, 1, 1]]),
            y_train=np.array([0, 1]),
            X_test=np.array([[0, 1, 0], [0, 1, 1]]),
            y_test=np.array([0, 1, 0, 0, 0, 0]),
        )
    with pytest.raises(ValueError, match=r"Cannot call transform on a validator .*fitted"):
        validator.transform(
            X=np.array([[0, 1, 0], [0, 1, 1]]),
            y=np.array([0, 1]),
        )


@pytest.mark.parametrize(
    'input_data_featuretest',
    (
        'numpy_numericalonly_nonan',
        'numpy_numericalonly_nan',
        'numpy_mixed_nan',
        'pandas_numericalonly_nan',
        'sparse_bsr_nonan',
        'sparse_bsr_nan',
        'sparse_coo_nonan',
        'sparse_coo_nan',
        'sparse_csc_nonan',
        'sparse_csc_nan',
        'sparse_csr_nonan',
        'sparse_csr_nan',
        'sparse_dia_nonan',
        'sparse_dia_nan',
        'sparse_dok_nonan',
        'sparse_dok_nan',
        'openml_40981',  # Australian
    ),
    indirect=True
)
def test_featurevalidator_dataset_compression(input_data_featuretest):
    n_samples = input_data_featuretest.shape[0]
    input_data_targets = np.random.random_sample((n_samples))
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
        input_data_featuretest, input_data_targets, test_size=0.1, random_state=1)
    validator = TabularInputValidator(
        dataset_compression={'memory_allocation': 0.8 * megabytes(X_train), 'methods': ['precision', 'subsample']}
    )
    validator.fit(X_train=X_train, y_train=y_train)
    transformed_X_train, _ = validator.transform(X_train.copy(), y_train.copy())

    assert validator._reduced_dtype is not None
    assert megabytes(transformed_X_train) < megabytes(X_train)

    transformed_X_test, _ = validator.transform(X_test.copy(), y_test.copy())
    assert megabytes(transformed_X_test) < megabytes(X_test)
    if hasattr(transformed_X_train, 'iloc'):
        assert all(transformed_X_train.dtypes == transformed_X_test.dtypes)
        assert all(transformed_X_train.dtypes == validator._precision)
    else:
        assert transformed_X_train.dtype == transformed_X_test.dtype
    assert transformed_X_test.dtype == validator._reduced_dtype
