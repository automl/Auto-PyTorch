import copy
import functools

import numpy as np

import pandas as pd

import pytest

from scipy import sparse

import sklearn.datasets
import sklearn.model_selection

from autoPyTorch.data.tabular_feature_validator import TabularFeatureValidator


# Actual checks for the features
@pytest.mark.parametrize(
    'input_data_featuretest',
    (
        'numpy_categoricalonly_nonan',
        'numpy_numericalonly_nonan',
        'numpy_mixed_nonan',
        'numpy_categoricalonly_nan',
        'numpy_numericalonly_nan',
        'numpy_mixed_nan',
        'pandas_categoricalonly_nonan',
        'pandas_numericalonly_nonan',
        'pandas_mixed_nonan',
        'pandas_numericalonly_nan',
        'list_numericalonly_nonan',
        'list_numericalonly_nan',
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
        'sparse_lil_nonan',
        'sparse_lil_nan',
        'openml_40981',  # Australian
    ),
    indirect=True
)
def test_featurevalidator_supported_types(input_data_featuretest):
    validator = TabularFeatureValidator()
    validator.fit(input_data_featuretest, input_data_featuretest)
    transformed_X = validator.transform(input_data_featuretest)
    if sparse.issparse(input_data_featuretest):
        assert sparse.issparse(transformed_X)
    else:
        assert isinstance(transformed_X, np.ndarray)
    assert np.shape(input_data_featuretest) == np.shape(transformed_X)
    assert np.issubdtype(transformed_X.dtype, np.number)
    assert validator._is_fitted


@pytest.mark.parametrize(
    'input_data_featuretest',
    (
        'numpy_string_nonan',
        'numpy_string_nan',
    ),
    indirect=True
)
def test_featurevalidator_unsupported_numpy(input_data_featuretest):
    validator = TabularFeatureValidator()
    with pytest.raises(ValueError, match=r".*When providing a numpy array.*not supported."):
        validator.fit(input_data_featuretest)


@pytest.mark.parametrize(
    'input_data_featuretest',
    (
        'pandas_categoricalonly_nan',
        'pandas_mixed_nan',
        'openml_179',  # adult workclass has NaN in columns
    ),
    indirect=True
)
def test_featurevalidator_categorical_nan(input_data_featuretest):
    validator = TabularFeatureValidator()
    validator.fit(input_data_featuretest)
    transformed_X = validator.transform(input_data_featuretest)
    assert any(pd.isna(input_data_featuretest))
    categories_ = validator.column_transformer.named_transformers_['categorical_pipeline'].\
        named_steps['ordinalencoder'].categories_
    assert any(('0' in categories) or (0 in categories) or ('missing_value' in categories) for categories in
               categories_)
    assert np.shape(input_data_featuretest) == np.shape(transformed_X)
    assert np.issubdtype(transformed_X.dtype, np.number)
    assert validator._is_fitted
    assert isinstance(transformed_X, np.ndarray)


@pytest.mark.parametrize(
    'input_data_featuretest',
    (
        'numpy_categoricalonly_nonan',
        'numpy_mixed_nonan',
        'numpy_categoricalonly_nan',
        'numpy_mixed_nan',
        'pandas_categoricalonly_nonan',
        'pandas_mixed_nonan',
        'list_numericalonly_nonan',
        'list_numericalonly_nan',
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
        'sparse_lil_nonan',
    ),
    indirect=True
)
def test_featurevalidator_fitontypeA_transformtypeB(input_data_featuretest):
    """
    Check if we can fit in a given type (numpy) yet transform
    if the user changes the type (pandas then)

    This is problematic only in the case we create an encoder
    """
    validator = TabularFeatureValidator()
    validator.fit(input_data_featuretest, input_data_featuretest)
    if isinstance(input_data_featuretest, pd.DataFrame):
        pytest.skip("Column order change in pandas is not supported")
    elif isinstance(input_data_featuretest, np.ndarray):
        complementary_type = pd.DataFrame(input_data_featuretest)
    elif isinstance(input_data_featuretest, list):
        complementary_type = pd.DataFrame(input_data_featuretest)
    elif sparse.issparse(input_data_featuretest):
        complementary_type = sparse.csr_matrix(input_data_featuretest.todense())
    else:
        raise ValueError(type(input_data_featuretest))
    transformed_X = validator.transform(complementary_type)
    assert np.shape(input_data_featuretest) == np.shape(transformed_X)
    assert np.issubdtype(transformed_X.dtype, np.number)
    assert validator._is_fitted


def test_featurevalidator_get_columns_to_encode():
    """
    Makes sure that encoded columns are returned by _get_columns_to_encode
    whereas numerical columns are not returned
    """
    validator = TabularFeatureValidator()

    df = pd.DataFrame([
        {'int': 1, 'float': 1.0, 'category': 'one', 'bool': True},
        {'int': 2, 'float': 2.0, 'category': 'two', 'bool': False},
    ])

    for col in df.columns:
        df[col] = df[col].astype(col)

    transformed_columns, feature_types = validator._get_columns_to_encode(df)

    assert transformed_columns == ['category', 'bool']
    assert feature_types == ['numerical', 'numerical', 'categorical', 'categorical']


def test_features_unsupported_calls_are_raised():
    """
    Makes sure we raise a proper message to the user,
    when providing not supported data input or using the validator in a way that is not
    expected
    """
    validator = TabularFeatureValidator()
    with pytest.raises(ValueError, match=r"AutoPyTorch does not support time"):
        validator.fit(
            pd.DataFrame({'datetime': [pd.Timestamp('20180310')]})
        )
    with pytest.raises(ValueError, match=r"AutoPyTorch only supports.*yet, the provided input"):
        validator.fit({'input1': 1, 'input2': 2})
    with pytest.raises(ValueError, match=r"has unsupported dtype string"):
        validator.fit(pd.DataFrame([{'A': 1, 'B': 2}], dtype='string'))
    with pytest.raises(ValueError, match=r"The feature dimensionality of the train and test"):
        validator.fit(X_train=np.array([[1, 2, 3], [4, 5, 6]]),
                      X_test=np.array([[1, 2, 3, 4], [4, 5, 6, 7]]),
                      )
    with pytest.raises(ValueError, match=r"Cannot call transform on a validator that is not fit"):
        validator.transform(np.array([[1, 2, 3], [4, 5, 6]]))


@pytest.mark.parametrize(
    'input_data_featuretest',
    (
        'numpy_numericalonly_nonan',
        'numpy_numericalonly_nan',
        'pandas_numericalonly_nonan',
        'pandas_numericalonly_nan',
        'list_numericalonly_nonan',
        'list_numericalonly_nan',
        # Category in numpy is handled via feat_type
        'numpy_categoricalonly_nonan',
        'numpy_mixed_nonan',
        'numpy_categoricalonly_nan',
        'numpy_mixed_nan',
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
        'sparse_lil_nonan',
        'sparse_lil_nan',
    ),
    indirect=True
)
def test_no_column_transformer_created(input_data_featuretest):
    """
    Makes sure that for numerical only features, no encoder is created
    """
    validator = TabularFeatureValidator()
    validator.fit(input_data_featuretest)
    validator.transform(input_data_featuretest)
    assert validator.column_transformer is None


@pytest.mark.parametrize(
    'input_data_featuretest',
    (
        'pandas_categoricalonly_nonan',
        'pandas_mixed_nonan',
    ),
    indirect=True
)
def test_column_transformer_created(input_data_featuretest):
    """
    This test ensures an encoder is created if categorical data is provided
    """
    validator = TabularFeatureValidator()
    validator.fit(input_data_featuretest)
    transformed_X = validator.transform(input_data_featuretest)
    assert validator.column_transformer is not None

    # Make sure that the encoded features are actually encoded. Categorical columns are at
    # the start after transformation. In our fixtures, this is also honored prior encode
    transformed_columns, feature_types = validator._get_columns_to_encode(input_data_featuretest)

    # At least one categorical
    assert 'categorical' in validator.feat_types

    # Numerical if the original data has numerical only columns
    if np.any([pd.api.types.is_numeric_dtype(input_data_featuretest[col]
                                             ) for col in input_data_featuretest.columns]):
        assert 'numerical' in validator.feat_types
    for i, feat_type in enumerate(feature_types):
        if 'numerical' in feat_type:
            np.testing.assert_array_equal(
                transformed_X[:, i],
                input_data_featuretest[input_data_featuretest.columns[i]].to_numpy()
            )
        elif 'categorical' in feat_type:
            np.testing.assert_array_equal(
                transformed_X[:, i],
                # Expect always 0, 1... because we use a ordinal encoder
                np.array([0, 1])
            )
        else:
            raise ValueError(feat_type)


def test_no_new_category_after_fit():
    """
    This test makes sure that we can actually pass new categories to the estimator
    without throwing an error
    """
    # Then make sure we catch categorical extra categories
    x = pd.DataFrame({'A': [1, 2, 3, 4], 'B': [5, 6, 7, 8]}, dtype='category')
    validator = TabularFeatureValidator()
    validator.fit(x)
    x['A'] = x['A'].apply(lambda x: x * x)
    validator.transform(x)


def test_unknown_encode_value():
    x = pd.DataFrame([
        {'a': -41, 'b': -3, 'c': 'a', 'd': -987.2},
        {'a': -21, 'b': -3, 'c': 'a', 'd': -9.2},
        {'a': 0, 'b': -4, 'c': 'b', 'd': -97.2},
        {'a': -51, 'b': -3, 'c': 'a', 'd': 987.2},
        {'a': 500, 'b': -3, 'c': 'a', 'd': -92},
    ])
    x['c'] = x['c'].astype('category')
    validator = TabularFeatureValidator()

    # Make sure that this value is honored
    validator.fit(x)
    x['c'].cat.add_categories(['NA'], inplace=True)
    x.loc[0, 'c'] = 'NA'  # unknown value
    x_t = validator.transform(x)
    # The first row should have a -1 as we added a new categorical there
    expected_row = [-1, -41, -3, -987.2]
    assert expected_row == x_t[0].tolist()

    # Notice how there is only one column 'c' to encode
    assert validator.categories == [list(range(2)) for i in range(1)]


# Actual checks for the features
@pytest.mark.parametrize(
    'openml_id',
    (
        40981,  # Australian
        3,  # kr-vs-kp
        1468,  # cnae-9
        40975,  # car
        40984,  # Segment
    ),
)
@pytest.mark.parametrize('train_data_type', ('numpy', 'pandas', 'list'))
@pytest.mark.parametrize('test_data_type', ('numpy', 'pandas', 'list'))
def test_featurevalidator_new_data_after_fit(openml_id,
                                             train_data_type, test_data_type):

    # List is currently not supported as infer_objects
    # cast list objects to type objects
    if train_data_type == 'list' or test_data_type == 'list':
        pytest.skip()

    validator = TabularFeatureValidator()

    if train_data_type == 'numpy':
        X, y = sklearn.datasets.fetch_openml(data_id=openml_id,
                                             return_X_y=True, as_frame=False)
    elif train_data_type == 'pandas':
        X, y = sklearn.datasets.fetch_openml(data_id=openml_id,
                                             return_X_y=True, as_frame=True)
    else:
        X, y = sklearn.datasets.fetch_openml(data_id=openml_id,
                                             return_X_y=True, as_frame=True)
        X = X.values.tolist()
        y = y.values.tolist()

    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
        X, y, random_state=1)

    validator.fit(X_train)

    transformed_X = validator.transform(X_test)

    # Basic Checking
    assert np.shape(X_test) == np.shape(transformed_X)

    # And then check proper error messages
    if train_data_type == 'pandas':
        old_dtypes = copy.deepcopy(validator.dtypes)
        validator.dtypes = ['dummy' for dtype in X_train.dtypes]
        with pytest.raises(ValueError, match=r"Changing the dtype of the features after fit"):
            transformed_X = validator.transform(X_test)
        validator.dtypes = old_dtypes
        if test_data_type == 'pandas':
            columns = X_test.columns.tolist()
            X_test = X_test[reversed(columns)]
            with pytest.raises(ValueError, match=r"Changing the column order of the features"):
                transformed_X = validator.transform(X_test)


def test_comparator():
    numerical = 'numerical'
    categorical = 'categorical'

    validator = TabularFeatureValidator

    with pytest.raises(ValueError, match=r"The comparator for the column order only accepts .*"):
        dummy = 'dummy'
        feat_type = [numerical, categorical, dummy]
        feat_type = sorted(
            feat_type,
            key=functools.cmp_to_key(validator._comparator)
        )

    feat_type = [numerical, categorical] * 10
    ans = [categorical] * 10 + [numerical] * 10
    feat_type = sorted(
        feat_type,
        key=functools.cmp_to_key(validator._comparator)
    )
    assert ans == feat_type

    feat_type = [numerical] * 10 + [categorical] * 10
    ans = [categorical] * 10 + [numerical] * 10
    feat_type = sorted(
        feat_type,
        key=functools.cmp_to_key(validator._comparator)
    )
    assert ans == feat_type


@pytest.fixture
def input_data_feature_feat_types(request):
    if request.param == 'pandas_categoricalonly':
        return pd.DataFrame([
            {'A': 1, 'B': 2},
            {'A': 3, 'B': 4},
        ], dtype='category'), ['categorical', 'categorical']
    elif request.param == 'pandas_numericalonly':
        return pd.DataFrame([
            {'A': 1, 'B': 2},
            {'A': 3, 'B': 4},
        ], dtype='float'), ['numerical', 'numerical']
    elif request.param == 'pandas_mixed':
        frame = pd.DataFrame([
            {'A': 1, 'B': 2},
            {'A': 3, 'B': 4},
        ], dtype='category')
        frame['B'] = pd.to_numeric(frame['B'])
        return frame, ['categorical', 'numerical']
    elif request.param == 'pandas_string_error':
        frame = pd.DataFrame([
            {'A': 1, 'B': '2'},
            {'A': 3, 'B': '4'},
        ], dtype='category')
        return frame, ['categorical', 'numerical']
    elif request.param == 'pandas_length_error':
        frame = pd.DataFrame([
            {'A': 1, 'B': '2'},
            {'A': 3, 'B': '4'},
        ], dtype='category')
        return frame, ['categorical', 'categorical', 'numerical']
    elif request.param == 'pandas_feat_type_error':
        frame = pd.DataFrame([
            {'A': 1, 'B': '2'},
            {'A': 3, 'B': '4'},
        ], dtype='category')
        return frame, ['not_categorical', 'numerical']
    else:
        ValueError("Unsupported indirect fixture {}".format(request.param))


@pytest.mark.parametrize(
    'input_data_feature_feat_types',
    (
        'pandas_categoricalonly',
        'pandas_numericalonly',
        'pandas_mixed',
    ),
    indirect=True
)
def test_feature_validator_get_columns_to_encode(input_data_feature_feat_types):
    X, feat_types = input_data_feature_feat_types
    validator = TabularFeatureValidator(feat_types=feat_types)
    transformed_columns, val_feat_types = validator.get_columns_to_encode(X)

    assert feat_types == val_feat_types

    for feat_type, col in zip(X.columns, val_feat_types):
        if feat_type.lower() == 'categorical':
            assert col in transformed_columns


@pytest.mark.parametrize(
    'input_data_feature_feat_types',
    (
        'pandas_string_error',
    ),
    indirect=True
)
def test_feature_validator_get_columns_to_encode_error_string(input_data_feature_feat_types):
    """
    Tests the correct error is raised when feat types passed to
    the validator disagree with the column dtypes.

    """
    X, feat_types = input_data_feature_feat_types
    validator = TabularFeatureValidator(feat_types=feat_types)
    with pytest.raises(ValueError, match=r"Passed numerical as the feature type for column: B but "
                                         r"the column is categorical"):
        validator.get_columns_to_encode(X)


@pytest.mark.parametrize(
    'input_data_feature_feat_types',
    (
        'pandas_length_error',
    ),
    indirect=True
)
def test_feature_validator_get_columns_to_encode_error_length(input_data_feature_feat_types):
    """
    Tests the correct error is raised when the length of feat types passed to
    the validator is not the same as the number of features

    """
    X, feat_types = input_data_feature_feat_types
    validator = TabularFeatureValidator(feat_types=feat_types)
    with pytest.raises(ValueError, match=r"Expected number of `feat_types`: .*"):
        validator._validate_feat_types(X)


@pytest.mark.parametrize(
    'input_data_feature_feat_types',
    (
        'pandas_feat_type_error',
    ),
    indirect=True
)
def test_feature_validator_get_columns_to_encode_error_feat_type(input_data_feature_feat_types):
    """
    Tests the correct error is raised when the length of feat types passed to
    the validator is not the same as the number of features

    """
    X, feat_types = input_data_feature_feat_types
    validator = TabularFeatureValidator(feat_types=feat_types)
    with pytest.raises(ValueError, match=r"Expected type of features to be in .*"):
        validator._validate_feat_types(X)
