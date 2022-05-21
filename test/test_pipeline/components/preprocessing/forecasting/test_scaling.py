import unittest

import numpy as np
import pandas as pd
from numpy.testing import assert_allclose

from sklearn.base import BaseEstimator
from sklearn.compose import make_column_transformer

from autoPyTorch.pipeline.components.preprocessing.time_series_preprocessing.scaling.base_scaler import BaseScaler
from autoPyTorch.pipeline.components.preprocessing.time_series_preprocessing.scaling.utils import TimeSeriesScaler


class TestScaling(unittest.TestCase):
    def setUp(self) -> None:
        data_seq_1 = np.array([[1, 2, 3],
                               [0, 2, 3],
                               [2, 2, 3],
                               ])

        data_seq_2 = np.array([[0, 1, 1],
                               [0, 1, 2],
                               [0, 1, 4],
                               [0, 1, 6]
                               ])

        columns = ['f1', 's', 'f2']
        self.raw_data = [data_seq_1, data_seq_2]
        self.data = pd.DataFrame(np.concatenate([data_seq_1, data_seq_2]), columns=columns, index=[0] * 3 + [1] * 4)
        self.static_features = ('s',)
        self.static_features_column = (1, )

        categorical_columns = list()
        numerical_columns = [0, 1, 2]

        self.dataset_properties = {'categorical_columns': categorical_columns,
                                   'numerical_columns': numerical_columns,
                                   'static_features': self.static_features,
                                   'is_small_preprocess': True}

    def test_base_and_standard_scaler(self):
        scaler_component = BaseScaler(scaling_mode='standard')
        X = {
            'X_train': self.data,
            'dataset_properties': self.dataset_properties
        }

        scaler_component = scaler_component.fit(dict(dataset_properties=self.dataset_properties))
        X = scaler_component.transform(X)

        scaler: TimeSeriesScaler = X['scaler']['numerical']

        # check if the fit dictionary X is modified as expected
        self.assertIsInstance(X['scaler'], dict)
        self.assertIsInstance(scaler, BaseEstimator)
        self.assertIsNone(X['scaler']['categorical'])

        # make column transformer with returned encoder to fit on data
        column_transformer = make_column_transformer((scaler, X['dataset_properties']['numerical_columns']),
                                                     remainder='passthrough')
        column_transformer = column_transformer.fit(X['X_train'])
        transformer = column_transformer.named_transformers_['timeseriesscaler']

        self.assertTrue(np.allclose(transformer.loc.values, np.asarray([[1.0, 1.428571, 3.00],
                                                                        [0.0, 1.428571, 3.25]])))

        self.assertTrue(np.allclose(transformer.scale.values, np.asarray([[1.0, 0.534522, 1.000000],
                                                                          [1.0, 0.534522, 2.217356]])))
        transformed = column_transformer.transform(self.data)

        self.assertTrue(np.allclose(transformed, np.asarray([[0., 1.06904497, 0.],
                                                             [-1., 1.06904497, 0.],
                                                             [1., 1.06904497, 0.],
                                                             [0., -0.80178373, -1.01472214],
                                                             [0., -0.80178373, -0.56373452],
                                                             [0., -0.80178373, 0.33824071],
                                                             [0., -0.80178373, 1.24021595]])))


        # second column is static features, those it need to be the mean and std value across all sequences
        scaler.dataset_is_small_preprocess = False
        scaler.static_features = self.static_features_column
        scaler = scaler.fit(self.raw_data[0])

        self.assertTrue(np.allclose(scaler.loc, np.asarray([[1., 2., 3]])))
        self.assertTrue(np.allclose(scaler.scale, np.asarray([[1., 2., 3.]])))

        transformed_test = scaler.transform(self.raw_data[0])
        self.assertIsInstance(transformed_test, np.ndarray)
        # should have the same value as the second part of transformed except for the static values
        self.assertTrue(np.allclose(transformed_test[:, [0, -1]], transformed[:len(self.raw_data[0]), [0, -1]]))
        self.assertTrue(np.all(transformed_test[:, 1] == 0.))

    def test_min_max(self):
        scaler = TimeSeriesScaler(mode='min_max',
                                  static_features=self.static_features
                                  )

        scaler = scaler.fit(self.data)
        self.assertTrue(np.allclose(scaler.loc.values, np.asarray([[0, 1, 3],
                                                                   [0, 1, 1]])))

        self.assertTrue(np.allclose(scaler.scale.values, np.asarray([[2, 1, 1],
                                                                     [1, 1, 5]])))

        transformed_data = scaler.transform(self.data).values
        self.assertTrue(np.allclose(transformed_data, np.asarray([[0.5, 1., 0.],
                                                                  [0., 1., 0.],
                                                                  [1., 1., 0.],
                                                                  [0., 0., 0.],
                                                                  [0., 0., 0.2],
                                                                  [0., 0., 0.6],
                                                                  [0., 0., 1.]])))

        scaler.dataset_is_small_preprocess = False
        scaler.static_features = self.static_features_column
        scaler = scaler.fit(self.raw_data[0])

        self.assertTrue(np.allclose(scaler.loc, np.asarray([[0., 2., 3.]])))
        self.assertTrue(np.allclose(scaler.scale, np.asarray([[2., 2., 3.]])))

        idx_start = 0
        for i, raw_data in enumerate(self.raw_data):
            idx_end = idx_start + len(raw_data)
            scaler = scaler.fit(raw_data)

            transformed_test = scaler.transform(self.raw_data[i])
            self.assertIsInstance(transformed_test, np.ndarray)
            # should have the same value as the second part of transformed except for the static values
            self.assertTrue(np.allclose(transformed_test[:, [0, -1]], transformed_data[idx_start:idx_end, [0, -1]]))
            self.assertTrue(np.all(transformed_test[:, 1] == 0.))
            idx_start = idx_end

    def test_max_abs_scaler(self):
        scaler = TimeSeriesScaler(mode='max_abs',
                                  static_features=self.static_features
                                  )

        scaler = scaler.fit(self.data)

        self.assertIsNone(scaler.loc)

        self.assertTrue(np.allclose(scaler.scale.values, np.asarray([[2, 2, 3],
                                                                     [1, 2, 6]])))

        transformed_data = scaler.transform(self.data).values

        self.assertTrue(np.allclose(transformed_data, np.asarray([[0.5, 1., 1.],
                                                                  [0., 1., 1.],
                                                                  [1., 1., 1.],
                                                                  [0., 0.5, 0.16666667],
                                                                  [0., 0.5, 0.33333333],
                                                                  [0., 0.5, 0.66666667],
                                                                  [0., 0.5, 1.]])))

        scaler.dataset_is_small_preprocess = False
        scaler.static_features = self.static_features_column
        scaler = scaler.fit(self.raw_data[0])
        self.assertIsNone(scaler.loc)
        self.assertTrue(np.allclose(scaler.scale, np.asarray([[2., 2., 3.]])))

        idx_start = 0
        for i, raw_data in enumerate(self.raw_data):
            idx_end = idx_start + len(raw_data)
            scaler = scaler.fit(raw_data)

            transformed_test = scaler.transform(self.raw_data[i])
            self.assertIsInstance(transformed_test, np.ndarray)
            # should have the same value as the second part of transformed except for the static values
            self.assertTrue(np.allclose(transformed_test[:, [0, -1]], transformed_data[idx_start:idx_end, [0, -1]]))
            self.assertTrue(np.all(transformed_test[:, 1] == 1.))
            idx_start = idx_end

    def test_mean_abs_scaler(self):
        scaler = TimeSeriesScaler(mode='mean_abs',
                                  static_features=self.static_features
                                  )

        scaler = scaler.fit(self.data)
        transformed_data = scaler.transform(self.data).values

        self.assertTrue(np.allclose(transformed_data, np.asarray([[1., 1.33333333, 1.],
                                                                  [0., 1.33333333, 1.],
                                                                  [2., 1.33333333, 1.],
                                                                  [0., 0.66666667, 0.30769231],
                                                                  [0., 0.66666667, 0.61538462],
                                                                  [0., 0.66666667, 1.23076923],
                                                                  [0., 0.66666667, 1.84615385]])))
        self.assertIsNone(scaler.loc)

        self.assertTrue(np.allclose(scaler.scale.values, np.asarray([[1., 1.5, 3.],
                                                                     [1., 1.5, 3.25]])))
        scaler.dataset_is_small_preprocess = False
        scaler.static_features = self.static_features_column
        scaler = scaler.fit(self.raw_data[0])

        self.assertIsNone(scaler.loc)
        self.assertTrue(np.allclose(scaler.scale, np.asarray([[1., 2., 3.]])))

        idx_start = 0
        for i, raw_data in enumerate(self.raw_data):
            idx_end = idx_start + len(raw_data)
            scaler = scaler.fit(raw_data)

            transformed_test = scaler.transform(self.raw_data[i])
            self.assertIsInstance(transformed_test, np.ndarray)
            # should have the same value as the second part of transformed except for the static values
            self.assertTrue(np.allclose(transformed_test[:, [0, -1]], transformed_data[idx_start:idx_end, [0, -1]]))
            self.assertTrue(np.all(transformed_test[:, 1] == 1.))
            idx_start = idx_end

    def test_no_scaler(self):
        scaler = TimeSeriesScaler(mode='none',
                                  static_features=self.static_features
                                  )

        scaler = scaler.fit(self.data)
        transformed_data = scaler.transform(self.data).values

        self.assertTrue(np.allclose(transformed_data, self.data.values))
        self.assertIsNone(scaler.loc)
        self.assertIsNone(scaler.scale)

        scaler.dataset_is_small_preprocess = False
        scaler.static_features = self.static_features_column
        scaler = scaler.fit(self.raw_data[0])

        idx_start = 0
        for i, raw_data in enumerate(self.raw_data):
            idx_end = idx_start + len(raw_data)
            scaler = scaler.fit(raw_data)

            transformed_test = scaler.transform(self.raw_data[i])
            self.assertIsInstance(transformed_test, np.ndarray)
            # should have the same value as the second part of transformed except for the static values
            self.assertTrue(np.allclose(transformed_test[:, [0, -1]], transformed_data[idx_start:idx_end, [0, -1]]))

            self.assertIsNone(scaler.loc)
            self.assertIsNone(scaler.scale)
            idx_start = idx_end

        with self.assertRaises(ValueError):
            scaler = TimeSeriesScaler(mode='random',
                                      static_features=self.static_features
                                      )
            _ = scaler.fit(self.data)
