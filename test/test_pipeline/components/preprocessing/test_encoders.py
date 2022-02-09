import unittest

import numpy as np
from numpy.testing import assert_array_equal

from sklearn.base import BaseEstimator
from sklearn.compose import make_column_transformer

from autoPyTorch.pipeline.components.preprocessing.tabular_preprocessing.encoding.NoEncoder import NoEncoder
from autoPyTorch.pipeline.components.preprocessing.tabular_preprocessing.encoding.OneHotEncoder import OneHotEncoder


class TestEncoders(unittest.TestCase):

    def test_one_hot_encoder_no_unknown(self):
        data = np.array([[1, 'male'],
                         [1, 'female'],
                         [3, 'female'],
                         [2, 'male'],
                         [2, 'female']])

        categorical_columns = [1]
        numerical_columns = [0]
        train_indices = np.array([0, 2, 3])
        test_indices = np.array([1, 4])

        dataset_properties = {
            'categorical_columns': categorical_columns,
            'numerical_columns': numerical_columns,
            'categories': [['female', 'male']]
        }
        X = {
            'X_train': data[train_indices],
            'dataset_properties': dataset_properties
        }
        encoder_component = OneHotEncoder()
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
        transformed = column_transformer.transform(data[test_indices])

        # check if the transform is correct
        assert_array_equal(transformed, [['1.0', '0.0', 1], ['1.0', '0.0', 2]])

    def test_none_encoder(self):

        data = np.array([[1, 'male'],
                         [1, 'female'],
                         [3, 'unknown'],
                         [2, 'female'],
                         [2, 'male']])

        categorical_columns = [1]
        numerical_columns = [0]
        train_indices = np.array([0, 2, 3])

        dataset_properties = {
            'categorical_columns': categorical_columns,
            'numerical_columns': numerical_columns,
            'categories': [['female', 'male', 'unknown']]
        }
        X = {
            'X_train': data[train_indices],
            'dataset_properties': dataset_properties
        }
        encoder_component = NoEncoder()
        encoder_component.fit(X)
        X = encoder_component.transform(X)

        # check if the fit dictionary X is modified as expected
        self.assertIsInstance(X['encoder'], dict)
        self.assertIsNone(X['encoder']['categorical'])
        self.assertIsNone(X['encoder']['numerical'])
