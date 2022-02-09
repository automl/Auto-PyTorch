import unittest

import numpy as np
from numpy.testing import assert_allclose

from sklearn.base import BaseEstimator
from sklearn.compose import make_column_transformer

from autoPyTorch.pipeline.components.preprocessing.tabular_preprocessing.scaling.MinMaxScaler import MinMaxScaler
from autoPyTorch.pipeline.components.preprocessing.tabular_preprocessing.scaling.NoScaler import NoScaler
from autoPyTorch.pipeline.components.preprocessing.tabular_preprocessing.scaling.Normalizer import Normalizer
from autoPyTorch.pipeline.components.preprocessing.tabular_preprocessing.scaling.PowerTransformer import \
    PowerTransformer
from autoPyTorch.pipeline.components.preprocessing.tabular_preprocessing.scaling.QuantileTransformer import \
    QuantileTransformer
from autoPyTorch.pipeline.components.preprocessing.tabular_preprocessing.scaling.RobustScaler import RobustScaler
from autoPyTorch.pipeline.components.preprocessing.tabular_preprocessing.scaling.StandardScaler import StandardScaler


class TestNormalizer(unittest.TestCase):

    def test_l2_norm(self):
        data = np.array([[1, 2, 3],
                         [7, 8, 9],
                         [4, 5, 6],
                         [11, 12, 13],
                         [17, 18, 19],
                         [14, 15, 16]])
        train_indices = np.array([0, 2, 5])
        test_indices = np.array([1, 4, 3])
        categorical_columns = list()
        numerical_columns = [0, 1, 2]
        dataset_properties = {'categorical_columns': categorical_columns,
                              'numerical_columns': numerical_columns, }
        X = {
            'X_train': data[train_indices],
            'dataset_properties': dataset_properties
        }
        scaler_component = Normalizer(norm='mean_squared')

        scaler_component = scaler_component.fit(X)
        X = scaler_component.transform(X)
        scaler = X['scaler']['numerical']

        # check if the fit dictionary X is modified as expected
        self.assertIsInstance(X['scaler'], dict)
        self.assertIsInstance(scaler, BaseEstimator)
        self.assertIsNone(X['scaler']['categorical'])

        # make column transformer with returned encoder to fit on data
        column_transformer = make_column_transformer((scaler, X['dataset_properties']['numerical_columns']),
                                                     remainder='passthrough')
        column_transformer = column_transformer.fit(X['X_train'])
        transformed = column_transformer.transform(data[test_indices])

        assert_allclose(transformed, np.array([[0.50257071, 0.57436653, 0.64616234],
                                               [0.54471514, 0.5767572, 0.60879927],
                                               [0.5280169, 0.57601843, 0.62401997]]))

    def test_l1_norm(self):
        data = np.array([[1, 2, 3],
                         [7, 8, 9],
                         [4, 5, 6],
                         [11, 12, 13],
                         [17, 18, 19],
                         [14, 15, 16]])
        train_indices = np.array([0, 2, 5])
        test_indices = np.array([1, 4, 3])
        categorical_columns = list()
        numerical_columns = [0, 1, 2]
        dataset_properties = {'categorical_columns': categorical_columns,
                              'numerical_columns': numerical_columns, }
        X = {
            'X_train': data[train_indices],
            'dataset_properties': dataset_properties
        }
        scaler_component = Normalizer(norm='mean_abs')

        scaler_component = scaler_component.fit(X)
        X = scaler_component.transform(X)
        scaler = X['scaler']['numerical']

        # check if the fit dictionary X is modified as expected
        self.assertIsInstance(X['scaler'], dict)
        self.assertIsInstance(scaler, BaseEstimator)
        self.assertIsNone(X['scaler']['categorical'])

        # make column transformer with returned encoder to fit on data
        column_transformer = make_column_transformer((scaler, X['dataset_properties']['numerical_columns']),
                                                     remainder='passthrough')
        column_transformer = column_transformer.fit(X['X_train'])
        transformed = column_transformer.transform(data[test_indices])

        assert_allclose(transformed, np.array([[0.29166667, 0.33333333, 0.375],
                                               [0.31481481, 0.33333333, 0.35185185],
                                               [0.30555556, 0.33333333, 0.36111111]]))

    def test_max_norm(self):
        data = np.array([[1, 2, 3],
                         [7, 8, 9],
                         [4, 5, 6],
                         [11, 12, 13],
                         [17, 18, 19],
                         [14, 15, 16]])
        train_indices = np.array([0, 2, 5])
        test_indices = np.array([1, 4, 3])
        categorical_columns = list()
        numerical_columns = [0, 1, 2]
        dataset_properties = {'categorical_columns': categorical_columns,
                              'numerical_columns': numerical_columns, }
        X = {
            'X_train': data[train_indices],
            'dataset_properties': dataset_properties
        }
        scaler_component = Normalizer(norm='max')

        scaler_component = scaler_component.fit(X)
        X = scaler_component.transform(X)
        scaler = X['scaler']['numerical']

        # check if the fit dictionary X is modified as expected
        self.assertIsInstance(X['scaler'], dict)
        self.assertIsInstance(scaler, BaseEstimator)
        self.assertIsNone(X['scaler']['categorical'])

        # make column transformer with returned encoder to fit on data
        column_transformer = make_column_transformer((scaler, X['dataset_properties']['numerical_columns']),
                                                     remainder='passthrough')
        column_transformer = column_transformer.fit(X['X_train'])
        transformed = column_transformer.transform(data[test_indices])

        assert_allclose(transformed, np.array([[0.77777778, 0.88888889, 1],
                                               [0.89473684, 0.94736842, 1],
                                               [0.84615385, 0.92307692, 1]]))


class TestMinMaxScaler(unittest.TestCase):

    def test_minmax_scaler(self):
        data = np.array([[1, 2, 3],
                         [7, 8, 9],
                         [4, 5, 6],
                         [11, 12, 13],
                         [17, 18, 19],
                         [14, 15, 16]])
        train_indices = np.array([0, 2, 5])
        test_indices = np.array([1, 4, 3])
        categorical_columns = list()
        numerical_columns = [0, 1, 2]
        dataset_properties = {'categorical_columns': categorical_columns,
                              'numerical_columns': numerical_columns, }
        X = {
            'X_train': data[train_indices],
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
        column_transformer = make_column_transformer((scaler, X['dataset_properties']['numerical_columns']),
                                                     remainder='passthrough')
        column_transformer = column_transformer.fit(X['X_train'])
        transformed = column_transformer.transform(data[test_indices])

        assert_allclose(transformed, np.array([[0.46153846, 0.46153846, 0.46153846],
                                               [1.23076923, 1.23076923, 1.23076923],
                                               [0.76923077, 0.76923077, 0.76923077]]))


class TestStandardScaler(unittest.TestCase):

    def test_standard_scaler(self):
        data = np.array([[1, 2, 3],
                         [7, 8, 9],
                         [4, 5, 6],
                         [11, 12, 13],
                         [17, 18, 19],
                         [14, 15, 16]])
        train_indices = np.array([0, 2, 5])
        test_indices = np.array([1, 4, 3])
        categorical_columns = list()
        numerical_columns = [0, 1, 2]
        dataset_properties = {'categorical_columns': categorical_columns,
                              'numerical_columns': numerical_columns,
                              'issparse': False}
        X = {
            'X_train': data[train_indices],
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
        column_transformer = make_column_transformer((scaler, X['dataset_properties']['numerical_columns']),
                                                     remainder='passthrough')
        column_transformer = column_transformer.fit(X['X_train'])
        transformed = column_transformer.transform(data[test_indices])

        assert_allclose(transformed, np.array([[0.11995203, 0.11995203, 0.11995203],
                                               [1.91923246, 1.91923246, 1.91923246],
                                               [0.8396642, 0.8396642, 0.8396642]]))


class TestNoneScaler(unittest.TestCase):

    def test_none_scaler(self):
        data = np.array([[1, 2, 3],
                         [7, 8, 9],
                         [4, 5, 6],
                         [11, 12, 13],
                         [17, 18, 19],
                         [14, 15, 16]])
        train_indices = np.array([0, 2, 5])
        categorical_columns = list()
        numerical_columns = [0, 1, 2]
        dataset_properties = {'categorical_columns': categorical_columns,
                              'numerical_columns': numerical_columns, }
        X = {
            'X_train': data[train_indices],
            'dataset_properties': dataset_properties
        }
        scaler_component = NoScaler()

        scaler_component = scaler_component.fit(X)
        X = scaler_component.transform(X)

        # check if the fit dictionary X is modified as expected
        self.assertIsInstance(X['scaler'], dict)
        self.assertIsNone(X['scaler']['categorical'])
        self.assertIsNone(X['scaler']['numerical'])


def test_power_transformer():
    data = np.array([[1, 2, 3],
                    [7, 8, 9],
                    [4, 5, 6],
                    [11, 12, 13],
                    [17, 18, 19],
                    [14, 15, 16]])
    train_indices = np.array([0, 2, 5])
    test_indices = np.array([1, 4, 3])
    categorical_columns = list()
    numerical_columns = [0, 1, 2]
    dataset_properties = {'categorical_columns': categorical_columns,
                          'numerical_columns': numerical_columns,
                          'issparse': False}
    X = {
        'X_train': data[train_indices],
        'dataset_properties': dataset_properties
    }
    scaler_component = PowerTransformer()

    scaler_component = scaler_component.fit(X)
    X = scaler_component.transform(X)
    scaler = X['scaler']['numerical']

    # check if the fit dictionary X is modified as expected
    assert isinstance(X['scaler'], dict)
    assert isinstance(scaler, BaseEstimator)
    assert X['scaler']['categorical'] is None

    # make column transformer with returned encoder to fit on data
    column_transformer = make_column_transformer((scaler, X['dataset_properties']['numerical_columns']),
                                                 remainder='passthrough')
    column_transformer = column_transformer.fit(X['X_train'])
    transformed = column_transformer.transform(data[test_indices])

    assert_allclose(transformed, np.array([[0.531648, 0.522782, 0.515394],
                                           [1.435794, 1.451064, 1.461685],
                                           [0.993609, 1.001055, 1.005734]]), rtol=1e-06)


def test_robust_scaler():
    data = np.array([[1, 2, 3],
                    [7, 8, 9],
                    [4, 5, 6],
                    [11, 12, 13],
                    [17, 18, 19],
                    [14, 15, 16]])
    train_indices = np.array([0, 2, 5])
    test_indices = np.array([1, 4, 3])
    categorical_columns = list()
    numerical_columns = [0, 1, 2]
    dataset_properties = {'categorical_columns': categorical_columns,
                          'numerical_columns': numerical_columns,
                          'issparse': False}
    X = {
        'X_train': data[train_indices],
        'dataset_properties': dataset_properties
    }
    scaler_component = RobustScaler()

    scaler_component = scaler_component.fit(X)
    X = scaler_component.transform(X)
    scaler = X['scaler']['numerical']

    # check if the fit dictionary X is modified as expected
    assert isinstance(X['scaler'], dict)
    assert isinstance(scaler, BaseEstimator)
    assert X['scaler']['categorical'] is None

    # make column transformer with returned encoder to fit on data
    column_transformer = make_column_transformer((scaler, X['dataset_properties']['numerical_columns']),
                                                 remainder='passthrough')
    column_transformer = column_transformer.fit(X['X_train'])
    transformed = column_transformer.transform(data[test_indices])

    assert_allclose(transformed, np.array([[100, 100, 100],
                                           [433.33333333, 433.33333333, 433.33333333],
                                           [233.33333333, 233.33333333, 233.33333333]]))


class TestQuantileTransformer():
    def test_quantile_transformer_uniform(self):
        data = np.array([[1, 2, 3],
                         [7, 8, 9],
                         [4, 5, 6],
                         [11, 12, 13],
                         [17, 18, 19],
                         [14, 15, 16]])
        train_indices = np.array([0, 2, 5])
        test_indices = np.array([1, 4, 3])
        categorical_columns = list()
        numerical_columns = [0, 1, 2]
        dataset_properties = {'categorical_columns': categorical_columns,
                              'numerical_columns': numerical_columns,
                              'issparse': False}
        X = {
            'X_train': data[train_indices],
            'dataset_properties': dataset_properties
        }
        scaler_component = QuantileTransformer(output_distribution='uniform')

        scaler_component = scaler_component.fit(X)
        X = scaler_component.transform(X)
        scaler = X['scaler']['numerical']

        # check if the fit dictionary X is modified as expected
        assert isinstance(X['scaler'], dict)
        assert isinstance(scaler, BaseEstimator)
        assert X['scaler']['categorical'] is None

        # make column transformer with returned encoder to fit on data
        column_transformer = make_column_transformer((scaler, X['dataset_properties']['numerical_columns']),
                                                     remainder='passthrough')
        column_transformer = column_transformer.fit(X['X_train'])
        transformed = column_transformer.transform(data[test_indices])

        assert_allclose(transformed, np.array([[0.65, 0.65, 0.65],
                                               [1, 1, 1],
                                               [0.85, 0.85, 0.85]]), rtol=1e-06)

    def test_quantile_transformer_normal(self):
        data = np.array([[1, 2, 3],
                         [7, 8, 9],
                         [4, 5, 6],
                         [11, 12, 13],
                         [17, 18, 19],
                         [14, 15, 16]])
        train_indices = np.array([0, 2, 5])
        test_indices = np.array([1, 4, 3])
        categorical_columns = list()
        numerical_columns = [0, 1, 2]
        dataset_properties = {'categorical_columns': categorical_columns,
                              'numerical_columns': numerical_columns,
                              'issparse': False}
        X = {
            'X_train': data[train_indices],
            'dataset_properties': dataset_properties
        }
        scaler_component = QuantileTransformer(output_distribution='normal')

        scaler_component = scaler_component.fit(X)
        X = scaler_component.transform(X)
        scaler = X['scaler']['numerical']

        # check if the fit dictionary X is modified as expected
        assert isinstance(X['scaler'], dict)
        assert isinstance(scaler, BaseEstimator)
        assert X['scaler']['categorical'] is None

        # make column transformer with returned encoder to fit on data
        column_transformer = make_column_transformer((scaler, X['dataset_properties']['numerical_columns']),
                                                     remainder='passthrough')
        column_transformer = column_transformer.fit(X['X_train'])
        transformed = column_transformer.transform(data[test_indices])

        assert_allclose(transformed, np.array([[0.38532, 0.38532, 0.38532],
                                               [5.199338, 5.199338, 5.199338],
                                               [1.036433, 1.036433, 1.036433]]), rtol=1e-05)
