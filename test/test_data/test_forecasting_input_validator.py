import numpy as np
import pytest
import pandas as pd
from autoPyTorch.data.time_series_forecasting_validator import TimeSeriesForecastingInputValidator

"""
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
            'pandas_with_static_features_multi_series',
    ),
    indirect=True
)
"""

def test_uni_variant_validator():
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
