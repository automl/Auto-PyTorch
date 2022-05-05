import numpy as np
import torch
import pandas as pd
import pytest
import unittest
from autoPyTorch.datasets.time_series_dataset import TimeSeriesForecastingDataset, TimeSeriesSequence
from gluonts.time_feature import Constant as ConstantTransform, DayOfMonth


class ZeroTransformer:
    def __call__(self, x: np.ndarray):
        return np.zeros_like(x)


class TestTimeSeriesSequence(unittest.TestCase):
    def setUp(self) -> None:
        rng = np.random.RandomState(1)
        self.data_length = 10
        self.n_prediction_steps = 3

        n_features = 5

        self.x_data = rng.rand(self.data_length, n_features)
        self.y = rng.rand(self.data_length, 1)

        self.x_test_data = rng.rand(self.n_prediction_steps, 5)
        self.y_test = rng.rand(self.n_prediction_steps, 1)
        self.time_feature_transform = [DayOfMonth(), ConstantTransform(10.0)]
        self.known_future_features_index = [0, 2]
        self.seq_uni = TimeSeriesSequence(X=None, Y=self.y, Y_test=self.y_test,
                                          n_prediction_steps=self.n_prediction_steps,
                                          time_feature_transform=self.time_feature_transform)
        self.seq_multi = TimeSeriesSequence(X=self.x_data,
                                            Y=self.y,
                                            X_test=self.x_test_data,
                                            Y_test=self.y_test, n_prediction_steps=self.n_prediction_steps,
                                            time_feature_transform=self.time_feature_transform,
                                            freq="1M")
        self.seq_multi_with_future = TimeSeriesSequence(X=self.x_data,
                                                        Y=self.y,
                                                        X_test=self.x_test_data,
                                                        Y_test=self.y_test, n_prediction_steps=self.n_prediction_steps,
                                                        time_feature_transform=self.time_feature_transform,
                                                        known_future_features_index=self.known_future_features_index,
                                                        freq="1M")

    def test_sequence_uni_variant_base(self):
        self.assertEqual(len(self.seq_uni), self.data_length - self.n_prediction_steps)
        idx = 6
        data, target = self.seq_uni[idx]
        self.assertTrue(isinstance(data['past_targets'], torch.Tensor))
        self.assertEqual(len(data['past_targets']), idx + 1)
        self.assertEqual(data['decoder_lengths'], self.n_prediction_steps)
        self.assertEqual(self.seq_uni.start_time, pd.Timestamp('1900-01-01'))
        mase_coefficient_1 = data['mase_coefficient']
        self.assertEqual(mase_coefficient_1.size, 1)
        # all data is observed
        self.assertTrue(data['past_observed_targets'].all())

        self.assertTrue(np.allclose(data['past_targets'].numpy(),
                                    self.y[:idx + 1]))
        self.assertTrue(np.allclose(target['future_targets'].numpy(),
                                    self.y[idx + 1:1 + idx + self.n_prediction_steps]))

        self.assertTrue(target['future_observed_targets'].all())

        self.assertTrue(self.seq_uni[-2][0]["past_targets"].size, self.data_length - self.n_prediction_steps - 2 + 1)

    def test_get_val_seq_and_test_targets(self):
        val_seq = self.seq_uni.get_val_seq_set(-1)
        self.assertEqual(len(val_seq), len(self.seq_uni))

        val_seq = self.seq_uni.get_val_seq_set(5)
        self.assertEqual(len(val_seq), 5 + 1)

        test_targets = self.seq_uni.get_test_target(-1)
        self.assertTrue(np.all(self.y[-self.n_prediction_steps:] == test_targets))

        test_targets = self.seq_uni.get_test_target(5)
        self.assertTrue(np.all(self.y[5 + 1: 5 + 1 + self.n_prediction_steps] == test_targets))

    def test_uni_get_update_time_faetures(self):
        self.seq_uni.update_attribute(transform_time_features=True)

        data, target = self.seq_uni[3]
        past_features = data["past_features"]
        future_features = data["future_features"]

        self.assertEqual(len(self.seq_uni._cached_time_features), len(self.y))
        self.assertTrue(list(past_features.shape) == [3 + 1, len(self.time_feature_transform)])
        self.assertTrue(list(future_features.shape) == [self.n_prediction_steps, len(self.time_feature_transform)])
        self.assertTrue(torch.all(past_features[:, 1] == 10.))
        self.assertTrue(torch.all(future_features[:, 1] == 10.))

    def test_uni_to_test_set(self):
        # For test set, its length should equal to y's length
        self.seq_uni.is_test_set = True
        self.assertEqual(len(self.seq_uni), len(self.y))

        self.seq_uni.transform_time_features = True

        data, target = self.seq_uni[-1]
        self.assertTrue(target is None)
        self.assertEqual(len(data["past_targets"]), len(self.y))
        self.assertEqual(len(data["past_features"]), len(self.y))
        self.assertEqual(len(self.seq_uni._cached_time_features), len(self.y) + self.n_prediction_steps)

    def test_observed_values(self):
        y_with_nan = self.seq_uni.Y.copy()
        y_with_nan[[3, -2]] = np.nan
        seq_1 = TimeSeriesSequence(X=None, Y=y_with_nan, n_prediction_steps=self.n_prediction_steps)
        data, target = seq_1[-1]
        self.assertFalse(data["past_observed_targets"][3])
        self.assertTrue(target["future_observed_targets"][2])

    def test_compute_mase_coefficient(self):
        seq_2 = TimeSeriesSequence(X=None, Y=self.y, n_prediction_steps=self.n_prediction_steps, is_test_set=True)
        self.assertNotEqual(self.seq_uni.mase_coefficient, seq_2.mase_coefficient)

    def test_sequence_multi_variant_base(self):
        data, _ = self.seq_multi[-1]
        self.assertEqual(list(data["past_features"].shape), [len(self.seq_multi), self.x_data.shape[-1]])
        self.assertTrue(data['future_features'] is None)

        data, _ = self.seq_multi[-1]

    def test_multi_known_future_variant(self):
        data, _ = self.seq_multi_with_future[-1]
        num_future_var = len(self.known_future_features_index)
        future_features = data['future_features']
        self.assertEqual(list(future_features.shape), [self.n_prediction_steps, num_future_var])
        self.assertTrue(np.allclose(
            future_features.numpy(),
            self.x_data[-self.n_prediction_steps:, self.known_future_features_index])
        )

    def test_multi_transform_features(self):
        self.seq_multi_with_future.transform_time_features = True
        num_future_var = len(self.known_future_features_index)

        data, _ = self.seq_multi_with_future[-1]
        past_features = data["past_features"]
        self.assertEqual(list(past_features.shape),
                         [len(self.seq_multi_with_future), self.x_data.shape[-1] + len(self.time_feature_transform)])

        self.assertTrue(np.allclose(
            past_features[:, -len(self.time_feature_transform):].numpy(),
            self.seq_multi_with_future._cached_time_features[:-self.n_prediction_steps]
        ))

        future_features = data["future_features"]
        self.assertEqual(list(future_features.shape),
                         [self.n_prediction_steps, num_future_var + len(self.time_feature_transform)])

        self.assertTrue(np.allclose(
            future_features[:, -len(self.time_feature_transform):].numpy(),
            self.seq_multi_with_future._cached_time_features[-self.n_prediction_steps:]
        ))

    def test_multi_to_test_set(self):
        self.seq_multi_with_future.is_test_set = True
        self.assertEqual(len(self.seq_multi_with_future.X), len(self.x_data) + len(self.x_test_data))
        data, _ = self.seq_multi_with_future[-1]

        self.assertTrue(np.allclose(data["past_features"].numpy(), self.x_data))
        self.assertTrue(
            np.allclose(data["future_features"].numpy(), self.x_test_data[:, self.known_future_features_index])
        )

        self.seq_multi_with_future.is_test_set = False
        self.assertEqual(len(self.seq_multi_with_future.X), len(self.x_data))

        seq_2 = self.seq_multi_with_future.get_val_seq_set(6)
        self.assertEqual(len(seq_2), 6 + 1)

    def test_transformation(self):
        self.seq_multi.update_transform(ZeroTransformer(), train=True)
        data, _ = self.seq_multi[-1]
        self.assertTrue(torch.all(data['past_features'][:, :-len(self.time_feature_transform)] == 0.))

        self.seq_multi.update_transform(ZeroTransformer(), train=False)
        data, _ = self.seq_multi.__getitem__(-1, False)
        self.assertTrue(torch.all(data['past_features'][:, :-len(self.time_feature_transform)] == 0.))

    def test_exception(self):
        seq_1 = TimeSeriesSequence(X=self.x_data, Y=self.y, X_test=None,
                                   known_future_features_index=self.known_future_features_index)
        with self.assertRaises(ValueError):
            seq_1.is_test_set = True

        seq_2 = TimeSeriesSequence(X=self.x_data, Y=self.y, X_test=None,
                                   is_test_set=True)

        with self.assertRaises(ValueError):
            seq_2.get_val_seq_set(5)

        with self.assertRaises(ValueError):
            seq_2.get_test_target(5)
