import unittest

import numpy as np
from numpy.testing import assert_array_equal

import pandas as pd

import pytest

from sklearn.base import BaseEstimator, clone
from sklearn.compose import make_column_transformer

from sktime.transformations.series.impute import Imputer as SKTImpute

from autoPyTorch.pipeline.components.preprocessing.time_series_preprocessing.imputation.TimeSeriesImputer import (
    TimeSeriesFeatureImputer,
    TimeSeriesTargetImputer
)


class TestTimeSeriesFeatureImputer(unittest.TestCase):
    def setUp(self) -> None:
        data = np.array([[1.0, np.nan, 3],
                         [np.nan, 8, 9],
                         [4.0, 5, np.nan],
                         [np.nan, 2, 3],
                         [7.0, np.nan, 9],
                         [4.0, np.nan, np.nan]])
        numerical_columns = [0, 1, 2]
        categorical_columns = []
        train_indices = np.array([0, 1, 2])
        self.test_indices = np.array([3, 4, 5])
        dataset_properties = {
            'categorical_columns': categorical_columns,
            'numerical_columns': numerical_columns,
        }
        self.X = {
            'X_train': data[train_indices],
            'dataset_properties': dataset_properties
        }
        self.data = data
        self.dataset_properties = dataset_properties

    def test_get_config_space(self):
        dataset_properties = dict(categorical_columns=[0, 1],
                                  numerical_columns=[1, 2],
                                  features_have_missing_values=True)
        config = TimeSeriesFeatureImputer.get_hyperparameter_search_space(dataset_properties).sample_configuration()
        estimator = TimeSeriesFeatureImputer(**config)
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

        dataset_properties['features_have_missing_values'] = False
        cs = TimeSeriesFeatureImputer.get_hyperparameter_search_space(dataset_properties)
        self.assertEqual(len(cs.get_hyperparameters()), 0)

        with self.assertRaises(ValueError):
            TimeSeriesFeatureImputer.get_hyperparameter_search_space()

    def test_drift_imputation(self):
        imputer_component = TimeSeriesFeatureImputer(imputation_strategy='drift')
        data = pd.DataFrame(self.data)

        imputer_component = imputer_component.fit(self.X)
        X = imputer_component.transform(self.X)
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
        transformed = column_transformer.transform(data.iloc[self.test_indices])

        skt_imputer = SKTImpute(method='drift', random_state=imputer_component.random_state)
        skt_imputer.fit(X['X_train'])

        self.assertTrue(np.allclose(transformed, skt_imputer.transform(data.iloc[self.test_indices]).values))

    def test_linear_imputation(self):
        imputer_component = TimeSeriesFeatureImputer(imputation_strategy='linear')

        imputer_component = imputer_component.fit(self.X)
        X = imputer_component.transform(self.X)
        numerical_imputer = X['imputer']['numerical']

        # make column transformer with returned encoder to fit on data
        column_transformer = make_column_transformer((numerical_imputer,
                                                      X['dataset_properties']['numerical_columns']),
                                                     remainder='passthrough')
        column_transformer = column_transformer.fit(X['X_train'])
        transformed = column_transformer.transform(self.data[self.test_indices])

        skt_imputer = SKTImpute(method='linear', random_state=imputer_component.random_state)
        skt_imputer.fit(X['X_train'])

        assert_array_equal(transformed, skt_imputer.transform(self.data[self.test_indices]))

    def test_nearest_imputation(self):
        data = np.array([[1.0, np.nan, 7],
                         [np.nan, 9, 10],
                         [10.0, 7, 7],
                         [9.0, np.nan, 11],
                         [9.0, 9, np.nan],
                         [np.nan, 5, 6],
                         [12.0, np.nan, 8],
                         [9.0, 7.0, np.nan]])
        numerical_columns = [0, 1, 2]
        categorical_columns = []
        train_indices = np.array([0, 1, 2, 3, 4])
        test_indices = np.array([5, 6, 7])
        dataset_properties = {
            'categorical_columns': categorical_columns,
            'numerical_columns': numerical_columns,
        }
        X = {
            'X_train': data[train_indices],
            'dataset_properties': dataset_properties
        }
        imputer_component = TimeSeriesFeatureImputer(imputation_strategy='nearest')

        imputer_component = imputer_component.fit(X)
        X = imputer_component.transform(X)
        numerical_imputer = X['imputer']['numerical']

        # make column transformer with returned encoder to fit on data
        column_transformer = make_column_transformer((numerical_imputer,
                                                      X['dataset_properties']['numerical_columns']),
                                                     remainder='passthrough')
        column_transformer = column_transformer.fit(X['X_train'])
        transformed = column_transformer.transform(data[test_indices])

        skt_imputer = SKTImpute(method='nearest', random_state=imputer_component.random_state)
        skt_imputer.fit(X['X_train'])

        assert_array_equal(transformed, skt_imputer.transform(data[test_indices]))

    def test_constant_imputation(self):
        imputer_component = TimeSeriesFeatureImputer(imputation_strategy='constant_zero')

        imputer_component = imputer_component.fit(self.X)
        X = imputer_component.transform(self.X)
        numerical_imputer = X['imputer']['numerical']

        # make column transformer with returned encoder to fit on data
        column_transformer = make_column_transformer((numerical_imputer,
                                                      X['dataset_properties']['numerical_columns']),
                                                     remainder='passthrough')
        column_transformer = column_transformer.fit(X['X_train'])
        transformed = column_transformer.transform(self.data[self.test_indices])
        assert_array_equal(transformed, np.array([[0, 2, 3],
                                                  [7, 0, 9],
                                                  [4, 0, 0]]))

    def test_bfill_imputation(self):
        imputer_component = TimeSeriesFeatureImputer(imputation_strategy='bfill')

        imputer_component = imputer_component.fit(self.X)
        X = imputer_component.transform(self.X)
        numerical_imputer = X['imputer']['numerical']

        # make column transformer with returned encoder to fit on data
        column_transformer = make_column_transformer((numerical_imputer,
                                                      X['dataset_properties']['numerical_columns']),
                                                     remainder='passthrough')
        column_transformer = column_transformer.fit(X['X_train'])
        transformed = column_transformer.transform(self.data[self.test_indices])

        assert_array_equal(transformed, np.array([[7., 2, 3],
                                                  [7, 2., 9],
                                                  [4, 2., 9.]]))

    def test_ffill_imputation(self):
        imputer_component = TimeSeriesFeatureImputer(imputation_strategy='ffill')

        imputer_component = imputer_component.fit(self.X)
        X = imputer_component.transform(self.X)
        numerical_imputer = X['imputer']['numerical']

        # make column transformer with returned encoder to fit on data
        column_transformer = make_column_transformer((numerical_imputer,
                                                      X['dataset_properties']['numerical_columns']),
                                                     remainder='passthrough')
        column_transformer = column_transformer.fit(X['X_train'])
        transformed = column_transformer.transform(self.data[self.test_indices])
        assert_array_equal(transformed, np.array([[7, 2, 3],
                                                  [7, 2, 9],
                                                  [4, 2, 9]]))


class TestTimeSeriesTargetImputer(unittest.TestCase):
    def test_get_config_space(self):
        dataset_properties = dict(categorical_columns=[0, 1],
                                  numerical_columns=[1, 2])
        config = TimeSeriesTargetImputer.get_hyperparameter_search_space(dataset_properties).sample_configuration()
        estimator = TimeSeriesFeatureImputer(**config)
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

        dataset_properties = dict(targets_have_missing_values=False)
        cs = TimeSeriesTargetImputer.get_hyperparameter_search_space(dataset_properties)
        self.assertEqual(len(cs.get_hyperparameters()), 0)

        with pytest.raises(ValueError):
            TimeSeriesTargetImputer.get_hyperparameter_search_space()

    def test_ffill_imputation(self):
        y = np.array([1.0, np.nan, 8, 9, 4.0, 5, np.nan]).reshape([-1, 1])
        numerical_columns = [0, 1, 2]
        categorical_columns = []
        dataset_properties = {
            'categorical_columns': categorical_columns,
            'numerical_columns': numerical_columns,
        }
        self.X = {
            'y_train': y,
            'dataset_properties': dataset_properties
        }
        self.dataset_properties = dataset_properties

        imputer_component = TimeSeriesTargetImputer(imputation_strategy='ffill')

        imputer_component = imputer_component.fit(self.X)

        imputer_component = imputer_component.fit(self.X)
        X = imputer_component.transform(self.X)
        numerical_imputer = X['target_imputer']['target_numerical']

        # check if the fit dictionary X is modified as expected
        self.assertIsInstance(X['target_imputer'], dict)
        self.assertIsInstance(numerical_imputer, BaseEstimator)

        # make column transformer with returned encoder to fit on data
        column_transformer = make_column_transformer((numerical_imputer, [0]),
                                                     remainder='passthrough')
        column_transformer = column_transformer.fit(X['y_train'])
        transformed = column_transformer.transform(y)
        assert_array_equal(transformed, np.array([[1.], [1.], [8.], [9.], [4.], [5.], [5.]]))


if __name__ == '__main__':
    unittest.main()
