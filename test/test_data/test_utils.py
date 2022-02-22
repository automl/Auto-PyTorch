import numpy as np

from pandas.testing import assert_frame_equal

import pytest

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

from autoPyTorch.data.utils import (
    megabytes,
    reduce_dataset_size_if_too_large,
    reduce_precision,
    validate_dataset_compression_arg
)
from autoPyTorch.utils.common import subsampler


@pytest.mark.parametrize('openmlid', [2, 40984])
@pytest.mark.parametrize('as_frame', [True, False])
def test_data_validation_for_classification(openmlid, as_frame, n_samples):
    X, _ = fetch_openml(data_id=openmlid, return_X_y=True, as_frame=as_frame)
    X = subsampler(data=X, x=range(n_samples))
    X_train, X_test = train_test_split(
        X, test_size=0.33, random_state=0)
    X_converted, precision = reduce_dataset_size_if_too_large(X.copy(), memory_allocation=0)
    np.allclose(X, X_converted) if not as_frame else assert_frame_equal(X, X_converted, check_dtype=False)
    assert megabytes(X_converted) < megabytes(X)
    if as_frame:
        assert isinstance(precision, dict)
        assert isinstance(list(precision.values())[0], type)
    else:
        assert isinstance(precision, type)


def test_validate_dataset_compression_arg():

    data_compression_args = validate_dataset_compression_arg({}, 10)
    # check whether the function uses default args
    # to fill in case args is empty
    assert data_compression_args is not None

    # assert memory allocation is an integer after validation
    assert isinstance(data_compression_args['memory_allocation'], int)

    # check whether the function raises an error
    # in case an unknown key is in args
    with pytest.raises(ValueError, match=r'Unknown key in dataset_compression, .*'):
        validate_dataset_compression_arg({'not_there': 1}, 1)

    # check whether the function raises an error
    # in case memory_allocation is not int or float is in args
    with pytest.raises(ValueError, match=r"key 'memory_allocation' must be an `int` or `float`.*"):
        validate_dataset_compression_arg({'memory_allocation': 'not int'}, 1)

    # check whether the function raises an error
    # in case memory_allocation is an int greater than memory limit
    with pytest.raises(ValueError, match=r"key 'memory_allocation' if int must be in.*"):
        validate_dataset_compression_arg({'memory_allocation': 1}, 0)

    # check whether the function raises an error
    # in case memory_allocation is a float greater than 1
    with pytest.raises(ValueError, match=r"key 'memory_allocation' if float must be in.*"):
        validate_dataset_compression_arg({'memory_allocation': 1.5}, 0)

    # check whether the function raises an error
    # in case an unknown method is passed in args
    with pytest.raises(ValueError, match=r"key 'methods' can only contain .*"):
        validate_dataset_compression_arg({'methods': 'unknown'}, 1)

    # check whether the function raises an error
    # in case an unknown key is in args
    with pytest.raises(ValueError, match=r'Unknown type for `dataset_compression` .*'):
        validate_dataset_compression_arg(1, 1)


def test_error_raised_reduce_precision():
    # check whether the function raises an error
    # in case X is not an expected type
    with pytest.raises(ValueError, match=r'Unrecognised data type of X, expected data type to .*'):
        reduce_precision(X='not expected')
