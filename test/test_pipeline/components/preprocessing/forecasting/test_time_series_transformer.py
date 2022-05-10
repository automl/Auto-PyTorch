from test.test_pipeline.components.preprocessing.forecasting.base import TimeSeriesTransformer

import numpy as np

import pytest

from scipy.sparse import csr_matrix

from sklearn.compose import ColumnTransformer

from autoPyTorch.pipeline.components.preprocessing.time_series_preprocessing.TimeSeriesTransformer import (
    TimeSeriesTransformer
)


@pytest.mark.parametrize("get_fit_dictionary_forecasting", ['uni_variant_wo_missing',
                                                            'uni_variant_w_missing',
                                                            'multi_variant_wo_missing',
                                                            'multi_variant_w_missing'], indirect=True)
class TimeSeriesForecstingTransformer:
    def test_tabular_preprocess(self, get_fit_dictionary_forecasting):
        pipeline = TimeSeriesTransformer(dataset_properties=get_fit_dictionary_forecasting['dataset_properties'])
        pipeline = pipeline.fit(get_fit_dictionary_forecasting)
        X = pipeline.transform(get_fit_dictionary_forecasting)
        column_transformer = X['tabular_transformer']

        # check if transformer was added to fit dictionary
        assert 'tabular_transformer' in X.keys()
        # check if transformer is of expected type
        # In this case we expect the tabular transformer not the actual column transformer
        # as the later is not callable and runs into error in the compose transform
        assert isinstance(column_transformer, TimeSeriesTransformer)

        data = column_transformer.preprocessor.fit_transform(X['X_train'])
        assert isinstance(data, np.ndarray)

        # Make sure no columns are unintentionally dropped after preprocessing
        if len(get_fit_dictionary_forecasting['dataset_properties']["numerical_columns"]) == 0:
            categorical_pipeline = column_transformer.preprocessor.named_transformers_['categorical_pipeline']
            categorical_data = categorical_pipeline.transform(X['X_train'])
            assert data.shape[1] == categorical_data.shape[1]
        elif len(get_fit_dictionary_forecasting['dataset_properties']["categorical_columns"]) == 0:
            numerical_pipeline = column_transformer.preprocessor.named_transformers_['numerical_pipeline']
            numerical_data = numerical_pipeline.transform(X['X_train'])
            assert data.shape[1] == numerical_data.shape[1]
