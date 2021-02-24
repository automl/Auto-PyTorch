import numpy as np

import pandas as pd

import pytest
import scipy

from scipy import sparse

from autoPyTorch.data.time_series_feature_validator import TimeSeriesFeatureValidator


# Fixtures to be used in this class. By default all elements have 100 datapoints
@pytest.fixture
def input_data_featuretest(request):
    if request.param == 'numpy_numericalonly_nonan':
        return np.array([
            [[1.0], [2.0], [3.0]],
            [[-3.0], [-2.0], [-1.0]]
        ])
    else:
        ValueError("Unsupported indirect fixture {}".format(request.param))


# Actual checks for the features
@pytest.mark.parametrize(
    'input_data_featuretest',
    (
        'numpy_numericalonly_nonan',
    ),
    indirect=True
)
def test_featurevalidator_supported_types(input_data_featuretest):
    validator = TimeSeriesFeatureValidator()
    validator.fit(input_data_featuretest, input_data_featuretest)
    transformed_X = validator.transform(input_data_featuretest)
    if sparse.issparse(input_data_featuretest):
        assert sparse.issparse(transformed_X)
    else:
        assert isinstance(transformed_X, np.ndarray)
    assert np.shape(input_data_featuretest) == np.shape(transformed_X)
    assert np.issubdtype(transformed_X.dtype, np.number)
    assert validator._is_fitted


def test_featurevalidator_unsupported_numpy():
    validator = TimeSeriesFeatureValidator()

    with pytest.raises(ValueError, match="Input contains NaN, infinity or a value too large *"):
        validator.fit(X_train=np.array([[[1], [2], [np.nan]], [[4], [5], [6]]]))


def test_features_unsupported_calls_are_raised():
    """
    Makes sure we raise a proper message to the user,
    when providing not supported data input or using the validator in a way that is not
    expected
    """
    validator = TimeSeriesFeatureValidator()

    with pytest.raises(ValueError, match="Time series train data must be given as a numpy array, but got *"):
        validator.fit(
            pd.DataFrame({'x': [1.0, 2.0, 3.0]})
        )

    with pytest.raises(ValueError, match="Time series train data must be given as a numpy array, but got *"):
        validator.fit(
            [1.0, 2.0, 3.0]
        )

    with pytest.raises(ValueError, match="Time series train data must be given as a numpy array, but got *"):
        validator.fit({'input1': 1, 'input2': 2})

    with pytest.raises(ValueError, match="Invalid number of dimensions for time series train data *"):
        validator.fit(X_train=np.array([[1, 2, 3], [4, 5, 6]]))

    with pytest.raises(ValueError, match="Invalid number of dimensions for time series test data *"):
        validator.fit(X_train=np.array([[[1], [2], [3]], [[4], [5], [6]]]),
                      X_test=np.array([[1, 2, 3], [4, 5, 6]]))

    with pytest.raises(ValueError, match="Time series train and test data are expected to have the same shape "
                                         "except for the batch dimension, but got *"):
        validator.fit(X_train=np.array([[[1], [2], [3]], [[4], [5], [6]]]),
                      X_test=np.array([[[1], [2], [3], [4]], [[4], [5], [6], [7]]]))

    with pytest.raises(ValueError, match=r"Cannot call transform on a validator that is not fit"):
        validator.transform(np.array([[1, 2, 3], [4, 5, 6]]))
