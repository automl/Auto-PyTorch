import numpy as np

import pytest

from sklearn.pipeline import Pipeline

from autoPyTorch.pipeline.components.preprocessing.time_series_preprocessing.TimeSeriesTransformer import \
    TimeSeriesTransformer

from test.test_pipeline.components.base import TimeSeriesPipeline


@pytest.mark.parametrize("fit_dictionary_time_series", ['classification_numerical_only'], indirect=True)
class TestTimeSeriesTransformer:
    def test_time_series_preprocess(self, fit_dictionary_time_series):
        pipeline = TimeSeriesPipeline(dataset_properties=fit_dictionary_time_series['dataset_properties'])
        pipeline = pipeline.fit(fit_dictionary_time_series)
        X = pipeline.transform(fit_dictionary_time_series)
        transformer = X['time_series_transformer']

        # check if transformer was added to fit dictionary
        assert 'time_series_transformer' in X.keys()
        # check if transformer is of expected type
        # In this case we expect the time series transformer not the actual implementation behind it
        # as the later is not callable and runs into error in the compose transform
        assert isinstance(transformer, TimeSeriesTransformer)
        assert isinstance(transformer.preprocessor, Pipeline)

        data = transformer.preprocessor.fit_transform(X['X_train'])
        assert isinstance(data, np.ndarray)
