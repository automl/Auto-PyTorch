import numpy as np

import pandas as pd

import pytest

from autoPyTorch.data.time_series_forecasting_validator import TimeSeriesForecastingInputValidator


def test_uni_variant_validator_only_y():
    validator = TimeSeriesForecastingInputValidator(is_classification=False)
    y_train = [[0.] * 5, [1.] * 10]
    y_test = [[0.] * 3, [1.] * 3]
    validator.fit(X_train=None, y_train=y_train, X_test=None, y_test=y_test)
    assert validator.start_times == [pd.Timestamp('1900-01-01')] * len(y_train)

    assert validator._is_fitted
    assert validator._is_uni_variant
    assert validator.feature_validator.num_features == 0
    assert len(validator.feature_validator.numerical_columns) == 0
    assert len(validator.feature_validator.categorical_columns) == 0
    assert validator.feature_validator._is_fitted is False
    assert len(validator.feature_shapes) == 0
    assert len(validator.feature_names) == 0

    x_transformed, y_transformed, sequence_lengths = validator.transform(None, y_train)
    assert x_transformed is None
    assert isinstance(y_transformed, pd.DataFrame)
    assert np.all(sequence_lengths == [5, 10])
    assert y_transformed.index.tolist() == sum([[i] * l_seq for i, l_seq in enumerate(sequence_lengths)], [])


@pytest.mark.parametrize(
    'input_data_forecastingfeaturetest',
    (
        'pandas_only_seriesid',
    ),
    indirect=True
)
def test_uni_variant_validator_with_series_id(input_data_forecastingfeaturetest):
    data, series_idx, seq_lengths = input_data_forecastingfeaturetest
    validator = TimeSeriesForecastingInputValidator(is_classification=False)
    start_times = [pd.Timestamp('2000-01-01')]
    x = [data]
    y = [list(range(len(data)))]
    validator.fit(x, y, start_times=start_times, series_idx=series_idx)
    assert validator._is_uni_variant is True
    assert validator.start_times == start_times
    x_transformed, y_transformed, sequence_lengths = validator.transform(x, y)
    assert x_transformed is None
    # for uni_variant validator, setting X as None should not cause any issue
    with pytest.raises(ValueError, match=r"X must be given as series_idx!"):
        _ = validator.transform(None, y)


@pytest.mark.parametrize(
    'input_data_forecastingfeaturetest',
    (
        'pandas_w_seriesid',
    ),
    indirect=True
)
def test_multi_variant_validator_with_series_id(input_data_forecastingfeaturetest):
    data, series_idx, seq_lengths = input_data_forecastingfeaturetest
    validator = TimeSeriesForecastingInputValidator(is_classification=False)
    start_times = [pd.Timestamp('2000-01-01')]
    x = [data]
    y = [list(range(len(data)))]
    validator.fit(x, y, start_times=start_times, series_idx=series_idx)
    x_transformed, y_transformed, sequence_lengths = validator.transform(x, y)
    assert series_idx not in x_transformed


@pytest.mark.parametrize(
    'input_data_forecastingfeaturetest',
    (
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
def test_transform_pds(input_data_forecastingfeaturetest):
    data, series_idx, _ = input_data_forecastingfeaturetest
    validator = TimeSeriesForecastingInputValidator(is_classification=False)
    # start_times = [pd.Timestamp('2000-01-01')]
    start_times = None
    x = data
    y = pd.DataFrame(range(len(data)))
    validator.fit(x, y, start_times=start_times, series_idx=series_idx)

    x_transformed, y_transformed, sequence_lengths = validator.transform(x, y)
    assert np.all(sequence_lengths == y_transformed.index.value_counts(sort=False).values)

    if x_transformed is not None:
        assert series_idx not in x_transformed
        assert np.all(sequence_lengths == x_transformed.index.value_counts(sort=False).values)
    if series_idx is not None:
        for seq_len, group in zip(sequence_lengths, data.groupby(series_idx)):
            assert seq_len == len(group[1])


def test_forecasting_validator():
    df = pd.DataFrame([
        {'category': 'one', 'int': 1, 'float': 1.0, 'bool': True},
        {'category': 'two', 'int': 2, 'float': 2.0, 'bool': False},
    ])

    for col in df.columns:
        df[col] = df[col].astype(col)

    x = [df, df]
    y = [[1., 2.], [1., 2.]]

    validator = TimeSeriesForecastingInputValidator()
    validator.fit(x, y, start_times=[pd.Timestamp('1900-01-01')] * 2)
    feature_names = ['category', 'bool', 'int', 'float']
    assert validator._is_uni_variant is False
    assert validator.feature_names == feature_names

    for fea_name in feature_names:
        assert fea_name in validator.feature_shapes
        assert validator.feature_shapes[fea_name] == 1

    x_transformed, y_transformed, sequence_lengths = validator.transform(x, y)
    assert isinstance(x_transformed, pd.DataFrame)
    assert isinstance(y_transformed, pd.DataFrame)
    assert np.all(x_transformed.index == y_transformed.index)
    assert len(x_transformed) == sum(sequence_lengths)

    # y is only allowed to be None if validate_for_future_features is True
    _ = validator.transform(x, None, validate_for_future_features=True)
    with pytest.raises(ValueError, match=r"Targets must be given!"):
        validator.transform(x)
    with pytest.raises(ValueError, match=r"Multi Variant dataset requires X as input!"):
        validator.transform(None, y)


def test_forecasting_handle_exception():
    validator = TimeSeriesForecastingInputValidator()
    # if X and y has different lengths
    X = [np.ones(3), np.ones(3)]
    y = [[1], ]
    with pytest.raises(ValueError, match="Inconsistent number of sequences for features and targets"):
        validator.fit(X, y)

    y = [[1], [1]]
    # test data must have the same shapes as they are attached to the tails of the datasets
    with pytest.raises(ValueError, match="Inconsistent number of test datapoints for features and targets"):
        validator.fit(X, y, X_test=X, y_test=y)
