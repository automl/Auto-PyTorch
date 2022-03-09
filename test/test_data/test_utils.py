import warnings
from test.test_data.utils import convert, dtype, size
from typing import Mapping

import numpy as np

import pandas as pd

import pytest

from scipy.sparse import csr_matrix

from sklearn.datasets import fetch_openml

from autoPyTorch.constants import (
    BINARY,
    CLASSIFICATION_TASKS,
    CONTINUOUS,
    CONTINUOUSMULTIOUTPUT,
    MULTICLASS,
    MULTICLASSMULTIOUTPUT,
    TABULAR_CLASSIFICATION,
    TABULAR_REGRESSION
)
from autoPyTorch.data.utils import (
    default_dataset_compression_arg,
    get_dataset_compression_mapping,
    megabytes,
    reduce_dataset_size_if_too_large,
    reduce_precision,
    subsample,
    validate_dataset_compression_arg
)
from autoPyTorch.utils.common import subsampler


@pytest.mark.parametrize('openmlid', [2, 40984])
@pytest.mark.parametrize('as_frame', [True, False])
def test_reduce_dataset_if_too_large(openmlid, as_frame, n_samples):
    X, y = fetch_openml(data_id=openmlid, return_X_y=True, as_frame=as_frame)
    X = subsampler(data=X, x=range(n_samples))
    y = subsampler(data=y, x=range(n_samples))

    X_converted, y_converted = reduce_dataset_size_if_too_large(
        X.copy(),
        y=y.copy(),
        is_classification=True,
        random_state=1,
        memory_allocation=0.001)

    assert X_converted.shape[0] < X.shape[0]
    assert y_converted.shape[0] < y.shape[0]

    assert megabytes(X_converted) < megabytes(X)


@pytest.mark.parametrize("X", [np.asarray([[1, 1, 1]] * 30)])
@pytest.mark.parametrize("x_type", [list, np.ndarray, csr_matrix, pd.DataFrame])
@pytest.mark.parametrize(
    "y, task, output",
    [
        (np.asarray([0] * 15 + [1] * 15), TABULAR_CLASSIFICATION, BINARY),
        (np.asarray([0] * 10 + [1] * 10 + [2] * 10), TABULAR_CLASSIFICATION, MULTICLASS),
        (np.asarray([[1, 0, 1]] * 30), TABULAR_CLASSIFICATION, MULTICLASSMULTIOUTPUT),
        (np.asarray([1.0] * 30), TABULAR_REGRESSION, CONTINUOUS),
        (np.asarray([[1.0, 1.0, 1.0]] * 30), TABULAR_REGRESSION, CONTINUOUSMULTIOUTPUT),
    ],
)
@pytest.mark.parametrize("y_type", [list, np.ndarray, pd.DataFrame, pd.Series])
@pytest.mark.parametrize("random_state", [0])
@pytest.mark.parametrize("sample_size", [0.25, 0.5, 5, 10])
def test_subsample_validity(X, x_type, y, y_type, random_state, sample_size, task, output):
    """Asserts the validity of the function with all valid types
    We want to make sure that `subsample` works correctly with all the types listed
    as x_type and y_type.
    We also want to make sure it works with all kinds of target types.
    The output should maintain the types, and subsample the correct amount.
    (test adapted from autosklearn)
    """
    assert len(X) == len(y)  # Make sure our test data is correct

    if y_type == pd.Series and output in [
        MULTICLASSMULTIOUTPUT,
        CONTINUOUSMULTIOUTPUT,
    ]:
        # We can't have a pd.Series with multiple values as it's 1 dimensional
        pytest.skip("Can't have pd.Series as y when task is n-dimensional")

    # Convert our data to its given x_type or y_type
    X = convert(X, x_type)
    y = convert(y, y_type)

    # Subsample the data, ignoring any warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        X_sampled, y_sampled = subsample(
            X,
            y=y,
            random_state=random_state,
            sample_size=sample_size,
            is_classification=task in CLASSIFICATION_TASKS,
        )

    # Check that the types of X remain the same after subsampling
    if isinstance(X, pd.DataFrame):
        # Dataframe can have multiple types, one per column
        assert list(dtype(X_sampled)) == list(dtype(X))
    else:
        assert dtype(X_sampled) == dtype(X)

    # Check that the types of y remain the same after subsampling
    if isinstance(y, pd.DataFrame):
        assert list(dtype(y_sampled)) == list(dtype(y))
    else:
        assert dtype(y_sampled) == dtype(y)

    # check the right amount of samples were taken
    if sample_size < 1:
        assert size(X_sampled) == int(sample_size * size(X))
    else:
        assert size(X_sampled) == sample_size


def test_validate_dataset_compression_arg():

    data_compression_args = validate_dataset_compression_arg({}, 10)
    # check whether the function uses default args
    # to fill in case args is empty
    assert data_compression_args is not None

    # assert memory allocation is a float after validation
    assert isinstance(data_compression_args['memory_allocation'], float)

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


def _verify_dataset_compression_mapping(mapping, expected_mapping):
    assert isinstance(mapping, Mapping)
    assert 'methods' in mapping
    assert 'memory_allocation' in mapping
    assert mapping == expected_mapping


@pytest.mark.parametrize('memory_limit', [2048])
def test_get_dataset_compression_mapping(memory_limit):
    """
    Tests the functionalities of `get_dataset_compression_mapping`
    """
    dataset_compression_mapping = get_dataset_compression_mapping(
        dataset_compression=True,
        memory_limit=memory_limit)
    # validation converts the memory allocation from float to integer based on the memory limit
    expected_mapping = validate_dataset_compression_arg(default_dataset_compression_arg, memory_limit)
    _verify_dataset_compression_mapping(dataset_compression_mapping, expected_mapping)

    mapping = {'memory_allocation': 0.01, 'methods': ['precision']}
    dataset_compression_mapping = get_dataset_compression_mapping(
        dataset_compression=mapping,
        memory_limit=memory_limit
    )
    expected_mapping = validate_dataset_compression_arg(mapping, memory_limit)
    _verify_dataset_compression_mapping(dataset_compression_mapping, expected_mapping)

    dataset_compression_mapping = get_dataset_compression_mapping(
        dataset_compression=False,
        memory_limit=memory_limit
    )
    assert dataset_compression_mapping is None


def test_unsupported_errors():
    """
    Checks if errors are raised when unsupported data is passed to reduce
    """
    X = np.array([
        ['a', 'b', 'c', 'a', 'b', 'c'],
        ['a', 'b', 'd', 'r', 'b', 'c']])
    with pytest.raises(ValueError, match=r'X.dtype = .*'):
        reduce_dataset_size_if_too_large(X, is_classification=True, random_state=1, memory_allocation=0)

    X = [[1, 2], [2, 3]]
    with pytest.raises(ValueError, match=r'Unrecognised data type of X, expected data type to be in .*'):
        reduce_dataset_size_if_too_large(X, is_classification=True, random_state=1, memory_allocation=0)
