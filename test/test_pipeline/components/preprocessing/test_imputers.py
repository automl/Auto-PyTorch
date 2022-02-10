import unittest

import numpy as np
from numpy.testing import assert_array_equal

import pytest

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
        data = np.array([[1.0, np.nan, 3],
                         [np.nan, 8, 9],
                         [4.0, 5, np.nan],
                         [np.nan, 2, 3],
                         [7.0, np.nan, 9],
                         [4.0, np.nan, np.nan]])
        numerical_columns = [0, 1, 2]
        categorical_columns = []
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
        self.assertIsNone(categorical_imputer)
        self.assertIsInstance(numerical_imputer, BaseEstimator)

        # make column transformer with returned encoder to fit on data
        column_transformer = make_column_transformer((numerical_imputer,
                                                      X['dataset_properties']['numerical_columns']),
                                                     remainder='passthrough')
        column_transformer = column_transformer.fit(X['X_train'])
        transformed = column_transformer.transform(data[test_indices])

        assert_array_equal(transformed, np.array([[2.5, 8, 9],
                                                  [7, 3.5, 9],
                                                  [4, 3.5, 3]]))

    def test_median_imputation(self):
        data = np.array([[1.0, np.nan, 7],
                         [np.nan, 9, 10],
                         [10.0, 7, 7],
                         [9.0, np.nan, 11],
                         [9.0, 9, np.nan],
                         [np.nan, 5, 6],
                         [12.0, np.nan, 8],
                         [9.0, np.nan, np.nan]])
        numerical_columns = [0, 1, 2]
        categorical_columns = []
        train_indices = np.array([0, 2, 3, 4, 7])
        test_indices = np.array([1, 5, 6])
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
        self.assertIsNone(categorical_imputer)
        self.assertIsInstance(numerical_imputer, BaseEstimator)

        # make column transformer with returned encoder to fit on data
        column_transformer = make_column_transformer((numerical_imputer,
                                                      X['dataset_properties']['numerical_columns']),
                                                     remainder='passthrough')
        column_transformer = column_transformer.fit(X['X_train'])
        transformed = column_transformer.transform(data[test_indices])

        assert_array_equal(transformed, np.array([[9, 9, 10],
                                                  [9, 5, 6],
                                                  [12, 8, 8]]))

    def test_frequent_imputation(self):
        data = np.array([[1.0, np.nan, 7],
                         [np.nan, 9, 10],
                         [10.0, 7, 7],
                         [9.0, np.nan, 11],
                         [9.0, 9, np.nan],
                         [np.nan, 5, 6],
                         [12.0, np.nan, 8],
                         [9.0, np.nan, np.nan]])
        numerical_columns = [0, 1, 2]
        categorical_columns = []
        train_indices = np.array([0, 2, 4, 5, 7])
        test_indices = np.array([1, 3, 6])
        dataset_properties = {
            'categorical_columns': categorical_columns,
            'numerical_columns': numerical_columns,
        }
        X = {
            'X_train': data[train_indices],
            'dataset_properties': dataset_properties
        }
        imputer_component = SimpleImputer(numerical_strategy='most_frequent')

        imputer_component = imputer_component.fit(X)
        X = imputer_component.transform(X)
        categorical_imputer = X['imputer']['categorical']
        numerical_imputer = X['imputer']['numerical']

        # check if the fit dictionary X is modified as expected
        self.assertIsInstance(X['imputer'], dict)
        self.assertIsNone(categorical_imputer)
        self.assertIsInstance(numerical_imputer, BaseEstimator)

        # make column transformer with returned encoder to fit on data
        column_transformer = make_column_transformer((numerical_imputer,
                                                      X['dataset_properties']['numerical_columns']),
                                                     remainder='passthrough')
        column_transformer = column_transformer.fit(X['X_train'])
        transformed = column_transformer.transform(data[test_indices])

        assert_array_equal(transformed, np.array([[9, 9, 10],
                                                  [9, 5, 11],
                                                  [12, 5, 8]]))

    def test_constant_imputation(self):
        data = np.array([[1.0, np.nan, 3],
                         [np.nan, 8, 9],
                         [4.0, 5, np.nan],
                         [np.nan, 2, 3],
                         [7.0, np.nan, 9],
                         [4.0, np.nan, np.nan]])
        numerical_columns = [0, 1, 2]
        categorical_columns = []
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
        imputer_component = SimpleImputer(numerical_strategy='constant_zero')

        imputer_component = imputer_component.fit(X)
        X = imputer_component.transform(X)
        categorical_imputer = X['imputer']['categorical']
        numerical_imputer = X['imputer']['numerical']

        # check if the fit dictionary X is modified as expected
        self.assertIsInstance(X['imputer'], dict)
        self.assertIsNone(categorical_imputer)
        self.assertIsInstance(numerical_imputer, BaseEstimator)

        # make column transformer with returned encoder to fit on data
        column_transformer = make_column_transformer((numerical_imputer,
                                                      X['dataset_properties']['numerical_columns']),
                                                     remainder='passthrough')
        column_transformer = column_transformer.fit(X['X_train'])
        transformed = column_transformer.transform(data[test_indices])
        assert_array_equal(transformed, np.array([[0, 8, 9],
                                                  [7, 0, 9],
                                                  [4, 0, 0]]))

    def test_imputation_without_dataset_properties_raises_error(self):
        """Tests SimpleImputer checks for dataset properties when querying for
        HyperparameterSearchSpace, even though the arg is marked `Optional`.

        Expects:
            * Should raise a ValueError that no dataset_properties were passed
        """
        with pytest.raises(ValueError):
            SimpleImputer.get_hyperparameter_search_space()


if __name__ == '__main__':
    unittest.main()
