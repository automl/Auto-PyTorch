from test.test_pipeline.components.preprocessing.forecasting.base import \
    ForecastingPipeline

import numpy as np

import pytest

from sklearn.compose import ColumnTransformer

from autoPyTorch.pipeline.components.preprocessing.time_series_preprocessing.TimeSeriesTransformer import (
    TimeSeriesFeatureTransformer,
    TimeSeriesTargetTransformer
)


@pytest.mark.parametrize("fit_dictionary_forecasting", ['uni_variant_wo_missing',
                                                        'uni_variant_w_missing',
                                                        'multi_variant_wo_missing',
                                                        'multi_variant_w_missing',
                                                        'multi_variant_w_missing_only_cat',
                                                        'multi_variant_w_missing_only_num',
                                                        ], indirect=True)
def test_time_series_preprocess(fit_dictionary_forecasting):
    pipeline = ForecastingPipeline(dataset_properties=fit_dictionary_forecasting['dataset_properties'])
    pipeline = pipeline.fit(fit_dictionary_forecasting)
    X = pipeline.transform(fit_dictionary_forecasting)

    assert 'time_series_target_transformer' in X.keys()
    target_transformer = X['time_series_target_transformer']

    # check if transformer is of expected type
    # In this case we expect the tabular transformer not the actual column transformer
    # as the later is not callable and runs into error in the compose transform
    assert isinstance(target_transformer, TimeSeriesTargetTransformer)

    targets = target_transformer.preprocessor.fit_transform(X['y_train'])
    assert isinstance(targets, np.ndarray)

    targets_2 = target_transformer(X['y_train'])
    assert np.allclose(targets, targets_2)

    assert isinstance(target_transformer.get_target_transformer(), ColumnTransformer)

    if not X['dataset_properties']['uni_variant']:
        assert 'time_series_feature_transformer' in X.keys()
        time_series_feature_transformer = X['time_series_feature_transformer']
        assert isinstance(time_series_feature_transformer, TimeSeriesFeatureTransformer)

        features = time_series_feature_transformer.preprocessor.fit_transform(X['X_train'])
        assert isinstance(features, np.ndarray)

        features_2 = time_series_feature_transformer(X['X_train'])
        assert np.allclose(features, features_2)

        assert isinstance(time_series_feature_transformer.get_column_transformer(), ColumnTransformer)

        # Make sure no columns are unintentionally dropped after preprocessing
        if len(fit_dictionary_forecasting['dataset_properties']["numerical_columns"]) == 0:
            categorical_pipeline = time_series_feature_transformer.preprocessor.named_transformers_[
                'categorical_pipeline'
            ]
            categorical_data = categorical_pipeline.transform(X['X_train'])
            assert features.shape[1] == categorical_data.shape[1]
        elif len(fit_dictionary_forecasting['dataset_properties']["categorical_columns"]) == 0:
            numerical_pipeline = time_series_feature_transformer.preprocessor.named_transformers_['numerical_pipeline']
            numerical_data = numerical_pipeline.transform(X['X_train'])
            assert features.shape[1] == numerical_data.shape[1]
