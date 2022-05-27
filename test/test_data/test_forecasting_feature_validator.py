import numpy as np

import pandas as pd

import pytest

from autoPyTorch.data.time_series_feature_validator import TimeSeriesFeatureValidator


# Actual checks for the features
@pytest.mark.parametrize(
    'input_data_forecastingfeaturetest',
    (
        'numpy_nonan',
        'numpy_with_static',
        'numpy_with_seq_length',
        'pandas_wo_seriesid',
        'pandas_w_seriesid',
        'pandas_only_seriesid',
        'pandas_without_seriesid',
        'pandas_with_static_features',
        'pandas_multi_seq',
        'pandas_multi_seq_w_idx',
        'pandas_with_static_features_multi_series',
    ),
    indirect=True
)
def test_forecasting_validator_supported_types(input_data_forecastingfeaturetest):
    data, series_idx, seq_lengths = input_data_forecastingfeaturetest
    validator = TimeSeriesFeatureValidator()
    validator.fit(data, data, series_idx, seq_lengths)

    if series_idx is not None:
        index = pd.MultiIndex.from_frame(pd.DataFrame(data[series_idx]))
    elif seq_lengths is not None:
        index = np.arange(len(seq_lengths)).repeat(seq_lengths)
    else:
        index = None
    if series_idx is not None and np.all(series_idx == data.columns):
        assert validator.only_contain_series_idx is True
        return

    transformed_X = validator.transform(data, index)
    assert isinstance(transformed_X, pd.DataFrame)
    if series_idx is None and seq_lengths is None:
        if not (isinstance(data, pd.DataFrame) and len(data.index.unique() > 1)):
            assert np.all(transformed_X.index == 0)
    else:
        if series_idx is not None:
            assert series_idx not in transformed_X
        else:
            if seq_lengths is not None:
                for i, group in enumerate(transformed_X.groupby(transformed_X.index)):
                    assert len(group[1]) == seq_lengths[i]
    # static features
    all_columns = transformed_X.columns
    all_columns_are_unique = {col: True for col in all_columns}
    for group in transformed_X.groupby(transformed_X.index):
        for col in group[1].columns:
            unique = np.unique(group[1][col])
            all_columns_are_unique[col] = all_columns_are_unique[col] & len(unique) == 1
    for key, value in all_columns_are_unique.items():
        if key in validator.static_features:
            assert value is True
        else:
            assert value is False
    assert validator._is_fitted


def test_forecasting_validator_get_reordered_columns():
    df = pd.DataFrame([
        {'category': 'one', 'int': 1, 'float': 1.0, 'bool': True},
        {'category': 'two', 'int': 2, 'float': 2.0, 'bool': False},
    ])

    for col in df.columns:
        df[col] = df[col].astype(col)

    validator = TimeSeriesFeatureValidator()
    validator.fit(df)
    reorder_cols = validator.get_reordered_columns()
    assert reorder_cols == ['category', 'bool', 'int', 'float']


def test_forecasting_validator_handle_exception():
    df = pd.DataFrame([
        {'A': 1, 'B': 2},
        {'A': np.NAN, 'B': 3},

    ])
    validator = TimeSeriesFeatureValidator()
    with pytest.raises(ValueError, match=r"All Series ID must be contained in the training column"):
        validator.fit(df, series_idx=['B', 'C'])
    with pytest.raises(ValueError, match=r'NaN should not exit in Series ID!'):
        validator.fit(df, series_idx=['A'])
    valirator2 = TimeSeriesFeatureValidator()
    valirator2.fit(df)
    with pytest.raises(ValueError, match=r'Given index must have length as the input features!'):
        valirator2.transform(df, index=[0] * 5)
