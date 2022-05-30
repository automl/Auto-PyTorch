import numpy as np

import pandas as pd

import pytest

from scipy import sparse

from autoPyTorch.data.time_series_target_validator import TimeSeriesTargetValidator


def test_forecasting_target_transform():
    validator = TimeSeriesTargetValidator(is_classification=False)
    series_length = 10
    y = np.ones(series_length)
    validator.fit(y)
    y_transformed_0 = validator.transform(y)
    assert isinstance(y_transformed_0, pd.DataFrame)
    assert np.all(y_transformed_0.index.values == np.zeros(series_length, dtype=np.int64))

    index_1 = np.full(series_length, 1)
    y_transformed_1 = validator.transform(y, index_1)
    assert np.all(y_transformed_1.index.values == index_1)

    index_2 = pd.Index([f"a{i}" for i in range(series_length)])
    y_transformed_2 = validator.transform(y, index_2)
    assert np.all(y_transformed_2.index.values == index_2)

    index_3 = [('a', 'a')] * (series_length // 3) + \
              [('a', 'b')] * (series_length // 3) + \
              [('b', 'a')] * (series_length - series_length // 3 * 2)
    index_3 = pd.MultiIndex.from_tuples(index_3)
    y_transformed_3 = validator.transform(y, index_3)
    assert isinstance(y_transformed_3.index, pd.MultiIndex)
    assert np.all(y_transformed_3.index == index_3)


def test_forecasting_target_handle_exception():
    validator = TimeSeriesTargetValidator(is_classification=False)
    target_sparse = sparse.csr_matrix(np.array([1, 1, 1]))
    with pytest.raises(NotImplementedError, match=r"Sparse Target is unsupported for forecasting task!"):
        # sparse matrix is unsupported for nan filling
        validator.fit(target_sparse)

    series_length = 10
    y = np.ones(series_length)
    validator.fit(y)
    with pytest.raises(ValueError, match=r"Index must have length as the input targets!"):
        validator.transform(y, np.asarray([1, 2, 3]))


def test_forecasting_target_missing_values():
    """
    Makes sure we raise a proper message to the user,
    when providing not supported data input
    """
    validator1 = TimeSeriesTargetValidator(is_classification=False)
    target_1 = np.array([np.nan, 1, 2])
    validator1.fit(target_1)
    assert validator1.transform(target_1).isnull().values.sum() == 1
