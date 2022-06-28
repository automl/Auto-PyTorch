import unittest

import numpy as np
from numpy.testing import assert_array_equal

import pandas as pd

from sklearn.base import BaseEstimator
from sklearn.compose import make_column_transformer

from autoPyTorch.pipeline.components.preprocessing.time_series_preprocessing.encoding.NoEncoder import \
    TimeSeriesNoEncoder
from autoPyTorch.pipeline.components.preprocessing.time_series_preprocessing.encoding.OneHotEncoder import \
    TimeSeriesOneHotEncoder


class TestEncoders(unittest.TestCase):
    def setUp(self) -> None:
        data = np.array([[1, 'male', 1],
                         [1, 'female', 2],
                         [1, 'unknown', 2],
                         [2, 'male', 2],
                         [2, 'female', 2]])
        feature_names = ("feature_n1", "feature_c", "feature_n2")

        self.data = pd.DataFrame(data, columns=feature_names)

        categorical_columns = [1]
        numerical_columns = [0, 2]
        self.train_indices = np.array([0, 1, 2])
        self.test_indices = np.array([3, 4])

        self.dataset_properties = {
            'categorical_columns': categorical_columns,
            'numerical_columns': numerical_columns,
            'categories': [['female', 'male', 'unknown']],
            'feature_names': feature_names,
            'feature_shapes': {fea: 1 for fea in feature_names}
        }

    def test_one_hot_encoder_no_unknown(self):
        X = {
            'X_train': self.data.iloc[self.train_indices],
            'dataset_properties': self.dataset_properties
        }
        encoder_component = TimeSeriesOneHotEncoder()
        encoder_component.fit(X)
        X = encoder_component.transform(X)
        encoder = X['encoder']['categorical']

        # check if the fit dictionary X is modified as expected
        self.assertIsInstance(X['encoder'], dict)
        self.assertIsInstance(encoder, BaseEstimator)
        self.assertIsNone(X['encoder']['numerical'])

        # make column transformer with returned encoder to fit on data
        column_transformer = make_column_transformer((encoder, X['dataset_properties']['categorical_columns']),
                                                     remainder='passthrough')
        column_transformer = column_transformer.fit(X['X_train'])
        transformed = column_transformer.transform(self.data.iloc[self.test_indices])
        # check if the transform is correct

        assert_array_equal(transformed.tolist(), [[0.0, 1.0, 0.0, '2', '2'], [1.0, 0.0, 0.0, '2', '2']])

        dataset_properties = X['dataset_properties']

        idx_cat = 0
        for i, fea_name in enumerate(dataset_properties['feature_names']):
            if i in dataset_properties['categorical_columns']:
                self.assertEqual(dataset_properties['feature_shapes'][fea_name],
                                 len(dataset_properties['categories'][idx_cat]))
                idx_cat += 1
            else:
                assert dataset_properties['feature_shapes'][fea_name] == 1

    def test_none_encoder(self):
        X = {
            'X_train': self.data.iloc[self.train_indices],
            'dataset_properties': self.dataset_properties
        }

        encoder_component = TimeSeriesNoEncoder()
        encoder_component.fit(X)
        X = encoder_component.transform(X)

        # check if the fit dictionary X is modified as expected
        self.assertIsInstance(X['encoder'], dict)
        self.assertIsNone(X['encoder']['categorical'])
        self.assertIsNone(X['encoder']['numerical'])

        dataset_properties = X['dataset_properties']
        for i, fea_name in enumerate(dataset_properties['feature_names']):
            self.assertEqual(dataset_properties['feature_shapes'][fea_name], 1)
