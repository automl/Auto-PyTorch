import unittest

import numpy as np
from numpy.testing import assert_array_equal

from sklearn.base import BaseEstimator, clone
from sklearn.compose import make_column_transformer

from autoPyTorch.pipeline.components.preprocessing.tabular_preprocessing.imputation.SimpleImputer import SimpleImputer


class TestSimpleImputer(unittest.TestCase):

    def test_get_config_space(self):
        dataset_properties = dict(categorical_columns=[0, 1],
                                  numerical_columns=[1, 2])
        config = SimpleImputer.get_hyperparameter_search_space(dataset_properties).sample_configuration()
        estimator = SimpleImputer(**config)
        estimator_clone = clone(estimator)
        estimator_clone_params = estimator_clone.get_params()

        # Make sure all keys are copied properly
        for k, v in estimator.get_params().items():
            self.assertIn(k, estimator_clone_params)

        # Make sure the params getter of estimator are honored
        klass = estimator.__class__
        new_object_params = estimator.get_params(deep=False)
        for name, param in new_object_params.items():
            new_object_params[name] = clone(param, safe=False)
        new_object = klass(**new_object_params)
        params_set = new_object.get_params(deep=False)

        for name in new_object_params:
            param1 = new_object_params[name]
            param2 = params_set[name]
            self.assertEqual(param1, param2)

    def test_mean_imputation(self):
        data = np.array([['1.0', np.nan, 3],
                         [np.nan, 8, 9],
                         ['4.0', 5, np.nan],
                         [np.nan, 2, 3],
                         ['7.0', np.nan, 9],
                         ['4.0', np.nan, np.nan]], dtype=object)
        numerical_columns = [1, 2]
        categorical_columns = [0]
        train_indices = np.array([0, 2, 3])
        test_indices = np.array([1, 4, 5])
        dataset_properties = {
            'categorical_columns': categorical_columns,
            'numerical_columns': numerical_columns,
        }
        X = {
            'X_train': data[train_indices],
            'dataset_properties': dataset_properties
        }
        imputer_component = SimpleImputer(numerical_strategy='mean')

        imputer_component = imputer_component.fit(X)
        X = imputer_component.transform(X)
        categorical_imputer = X['imputer']['categorical']
        numerical_imputer = X['imputer']['numerical']

        # check if the fit dictionary X is modified as expected
        self.assertIsInstance(X['imputer'], dict)
        self.assertIsInstance(categorical_imputer, BaseEstimator)
        self.assertIsInstance(numerical_imputer, BaseEstimator)

        # make column transformer with returned encoder to fit on data
        column_transformer = make_column_transformer((categorical_imputer,
                                                      X['dataset_properties']['categorical_columns']),
                                                     (numerical_imputer,
                                                      X['dataset_properties']['numerical_columns']),
                                                     remainder='passthrough')
        column_transformer = column_transformer.fit(X['X_train'])
        transformed = column_transformer.transform(data[test_indices])

        assert_array_equal(transformed.astype(str), np.array([[1.0, 8.0, 9.0],
                                                             [7.0, 3.5, 9.0],
                                                             [4.0, 3.5, 3.0]], dtype=str))

    def test_median_imputation(self):
        data = np.array([['1.0', np.nan, 3],
                         [np.nan, 8, 9],
                         ['4.0', 5, np.nan],
                         [np.nan, 2, 3],
                         ['7.0', np.nan, 9],
                         ['4.0', np.nan, np.nan]], dtype=object)
        numerical_columns = [1, 2]
        categorical_columns = [0]
        train_indices = np.array([0, 2, 3])
        test_indices = np.array([1, 4, 5])
        dataset_properties = {
            'categorical_columns': categorical_columns,
            'numerical_columns': numerical_columns,
        }
        X = {
            'X_train': data[train_indices],
            'dataset_properties': dataset_properties
        }
        imputer_component = SimpleImputer(numerical_strategy='median')

        imputer_component = imputer_component.fit(X)
        X = imputer_component.transform(X)
        categorical_imputer = X['imputer']['categorical']
        numerical_imputer = X['imputer']['numerical']

        # check if the fit dictionary X is modified as expected
        self.assertIsInstance(X['imputer'], dict)
        self.assertIsInstance(categorical_imputer, BaseEstimator)
        self.assertIsInstance(numerical_imputer, BaseEstimator)

        # make column transformer with returned encoder to fit on data
        column_transformer = make_column_transformer(
            (categorical_imputer, X['dataset_properties']['categorical_columns']),
            (numerical_imputer, X['dataset_properties']['numerical_columns']),
            remainder='passthrough'
        )
        column_transformer = column_transformer.fit(X['X_train'])
        transformed = column_transformer.transform(data[test_indices])

        assert_array_equal(transformed.astype(str), np.array([[1.0, 8.0, 9.0],
                                                             [7.0, 3.5, 9.0],
                                                             [4.0, 3.5, 3.0]], dtype=str))

    def test_frequent_imputation(self):
        data = np.array([['1.0', np.nan, 3],
                         [np.nan, 8, 9],
                         ['4.0', 5, np.nan],
                         [np.nan, 2, 3],
                         ['7.0', np.nan, 9],
                         ['4.0', np.nan, np.nan]], dtype=object)
        numerical_columns = [1, 2]
        categorical_columns = [0]
        train_indices = np.array([0, 2, 3])
        test_indices = np.array([1, 4, 5])
        dataset_properties = {
            'categorical_columns': categorical_columns,
            'numerical_columns': numerical_columns,
        }
        X = {
            'X_train': data[train_indices],
            'dataset_properties': dataset_properties
        }
        imputer_component = SimpleImputer(numerical_strategy='most_frequent',
                                          categorical_strategy='most_frequent')

        imputer_component = imputer_component.fit(X)
        X = imputer_component.transform(X)
        categorical_imputer = X['imputer']['categorical']
        numerical_imputer = X['imputer']['numerical']

        # check if the fit dictionary X is modified as expected
        self.assertIsInstance(X['imputer'], dict)
        self.assertIsInstance(categorical_imputer, BaseEstimator)
        self.assertIsInstance(numerical_imputer, BaseEstimator)

        # make column transformer with returned encoder to fit on data
        column_transformer = make_column_transformer(
            (categorical_imputer, X['dataset_properties']['categorical_columns']),
            (numerical_imputer, X['dataset_properties']['numerical_columns']),
            remainder='passthrough'
        )
        column_transformer = column_transformer.fit(X['X_train'])
        transformed = column_transformer.transform(data[test_indices])

        assert_array_equal(transformed.astype(str), np.array([[1.0, 8, 9],
                                                             [7.0, 2, 9],
                                                             [4.0, 2, 3]], dtype=str))

    def test_constant_imputation(self):
        data = np.array([['1.0', np.nan, 3],
                         [np.nan, 8, 9],
                         ['4.0', 5, np.nan],
                         [np.nan, 2, 3],
                         ['7.0', np.nan, 9],
                         ['4.0', np.nan, np.nan]], dtype=object)
        numerical_columns = [1, 2]
        categorical_columns = [0]
        train_indices = np.array([0, 2, 3])
        test_indices = np.array([1, 4, 5])
        dataset_properties = {
            'categorical_columns': categorical_columns,
            'numerical_columns': numerical_columns,
        }
        X = {
            'X_train': data[train_indices],
            'dataset_properties': dataset_properties
        }
        imputer_component = SimpleImputer(numerical_strategy='constant_zero',
                                          categorical_strategy='constant_!missing!')

        imputer_component = imputer_component.fit(X)
        X = imputer_component.transform(X)
        categorical_imputer = X['imputer']['categorical']
        numerical_imputer = X['imputer']['numerical']

        # check if the fit dictionary X is modified as expected
        self.assertIsInstance(X['imputer'], dict)
        self.assertIsInstance(categorical_imputer, BaseEstimator)
        self.assertIsInstance(numerical_imputer, BaseEstimator)

        # make column transformer with returned encoder to fit on data
        column_transformer = make_column_transformer(
            (categorical_imputer, X['dataset_properties']['categorical_columns']),
            (numerical_imputer, X['dataset_properties']['numerical_columns']),
            remainder='passthrough'
        )
        column_transformer = column_transformer.fit(X['X_train'])
        transformed = column_transformer.transform(data[test_indices])
        assert_array_equal(transformed.astype(str), np.array([['-1', 8, 9],
                                                             [7.0, '0', 9],
                                                             [4.0, '0', '0']], dtype=str))


if __name__ == '__main__':
    unittest.main()
