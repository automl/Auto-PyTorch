import unittest

import numpy as np
from numpy.testing import assert_allclose

from sklearn.base import BaseEstimator

from autoPyTorch.pipeline.components.preprocessing.time_series_preprocessing.scaling.MaxAbsScaler import MaxAbsScaler
from autoPyTorch.pipeline.components.preprocessing.time_series_preprocessing.scaling.MinMaxScaler import MinMaxScaler
from autoPyTorch.pipeline.components.preprocessing.time_series_preprocessing.scaling.NoScaler import NoScaler
from autoPyTorch.pipeline.components.preprocessing.time_series_preprocessing.scaling.StandardScaler import \
    StandardScaler


class TestMinMaxScaler(unittest.TestCase):

    def test_minmax_scaler(self):
        data = np.array([
            [[1], [2], [3]],
            [[7], [8], [9]],
            [[10], [11], [12]]
        ])

        dataset_properties = {'categorical_features': [],
                              'numerical_features': [0]}

        X = {
            'X_train': data,
            'dataset_properties': dataset_properties
        }
        scaler_component = MinMaxScaler()

        scaler_component = scaler_component.fit(X)
        X = scaler_component.transform(X)
        scaler = X['scaler']['numerical']

        # check if the fit dictionary X is modified as expected
        self.assertIsInstance(X['scaler'], dict)
        self.assertIsInstance(scaler, BaseEstimator)
        self.assertIsNone(X['scaler']['categorical'])

        # make column transformer with returned encoder to fit on data
        scaler = scaler.fit(X["X_train"])
        transformed = scaler.transform(X["X_train"])
        assert_allclose(transformed,
                        np.array([
                            [[0], [0.5], [1]],
                            [[0], [0.5], [1]],
                            [[0], [0.5], [1]],
                        ]))


class TestMaxAbsScaler(unittest.TestCase):

    def test_maxabs_scaler(self):
        data = np.array([
            [[-10], [2], [3]],
            [[-7], [8], [9]],
            [[-8], [11], [12]]
        ])

        dataset_properties = {'categorical_features': [],
                              'numerical_features': [0]}

        X = {
            'X_train': data,
            'dataset_properties': dataset_properties
        }
        scaler_component = MaxAbsScaler()

        scaler_component = scaler_component.fit(X)
        X = scaler_component.transform(X)
        scaler = X['scaler']['numerical']

        # check if the fit dictionary X is modified as expected
        self.assertIsInstance(X['scaler'], dict)
        self.assertIsInstance(scaler, BaseEstimator)
        self.assertIsNone(X['scaler']['categorical'])

        # make column transformer with returned encoder to fit on data
        scaler = scaler.fit(X["X_train"])
        transformed = scaler.transform(X["X_train"])
        assert_allclose(transformed,
                        np.array([
                            [[-1], [0.2], [0.3]],
                            [[-7 / 9], [8 / 9], [1]],
                            [[-8 / 12], [11 / 12], [1]],
                        ]))


class TestStandardScaler(unittest.TestCase):

    def test_standard_scaler(self):
        data = np.array([
            [[1], [2], [3], [4], [5]],
            [[7], [8], [9], [10], [11]],
            [[10], [11], [12], [13], [14]]
        ])

        dataset_properties = {'categorical_features': [],
                              'numerical_features': [0]}

        X = {
            'X_train': data,
            'dataset_properties': dataset_properties
        }
        scaler_component = StandardScaler()

        scaler_component = scaler_component.fit(X)
        X = scaler_component.transform(X)
        scaler = X['scaler']['numerical']

        # check if the fit dictionary X is modified as expected
        self.assertIsInstance(X['scaler'], dict)
        self.assertIsInstance(scaler, BaseEstimator)
        self.assertIsNone(X['scaler']['categorical'])

        # make column transformer with returned encoder to fit on data
        scaler = scaler.fit(X["X_train"])
        transformed = scaler.transform(X["X_train"])
        assert_allclose(transformed,
                        np.array([
                            [[-1.41421356], [-0.70710678], [0.], [0.70710678], [1.41421356]],
                            [[-1.41421356], [-0.70710678], [0.], [0.70710678], [1.41421356]],
                            [[-1.41421356], [-0.70710678], [0.], [0.70710678], [1.41421356]],
                        ]))


class TestNoneScaler(unittest.TestCase):

    def test_none_scaler(self):
        data = np.array([
            [[1], [2], [3]],
            [[7], [8], [9]],
            [[10], [11], [12]]
        ])

        dataset_properties = {'categorical_features': [],
                              'numerical_features': [0]}

        X = {
            'X_train': data,
            'dataset_properties': dataset_properties
        }
        scaler_component = NoScaler()

        scaler_component = scaler_component.fit(X)
        X = scaler_component.transform(X)

        # check if the fit dictionary X is modified as expected
        self.assertIsInstance(X['scaler'], dict)
        self.assertIsNone(X['scaler']['categorical'])
        self.assertIsNone(X['scaler']['numerical'])
