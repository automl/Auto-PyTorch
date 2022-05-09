from typing import List, Callable, Tuple

import numpy as np
import torch
import pandas as pd
import pytest
import unittest
from gluonts.time_feature import Constant as ConstantTransform, DayOfMonth
from autoPyTorch.datasets.time_series_dataset import (
    TimeSeriesForecastingDataset,
    TimeSeriesSequence,
    extract_feature_index
)
from autoPyTorch.datasets.resampling_strategy import (
    CrossValTypes,
    HoldoutValTypes
)
from autoPyTorch.utils.pipeline import get_dataset_requirements


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

        self.seq_uni.cache_time_features()
        val_seq = self.seq_uni.get_val_seq_set(5)
        self.assertEqual(len(val_seq), 5 + 1)
        self.assertEqual(len(val_seq._cached_time_features), 5 + 1 + self.n_prediction_steps)

        test_targets = self.seq_uni.get_test_target(-1)
        self.assertTrue(np.all(self.y[-self.n_prediction_steps:] == test_targets))

        test_targets = self.seq_uni.get_test_target(5)
        self.assertTrue(np.all(self.y[5 + 1: 5 + 1 + self.n_prediction_steps] == test_targets))

    def test_uni_get_update_time_features(self):
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
        self.seq_uni.transform_time_features = True
        self.seq_uni.cache_time_features()
        # For test set, its length should equal to y's length
        self.seq_uni.is_test_set = True
        self.assertEqual(len(self.seq_uni), len(self.y))

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


@pytest.mark.parametrize("get_fit_dictionary_forecasting", ['uni_variant_wo_missing',
                                                            'uni_variant_w_missing',
                                                            'multi_variant_wo_missing',
                                                            'uni_variant_w_missing'], indirect=True)
def test_dataset_properties(backend, get_fit_dictionary_forecasting):
    # The fixture creates a datamanager by itself
    datamanager: TimeSeriesForecastingDataset = backend.load_datamanager()
    info = {'task_type': datamanager.task_type,
            'numerical_features': datamanager.numerical_features,
            'categorical_features': datamanager.categorical_features,
            'output_type': datamanager.output_type,
            'numerical_columns': datamanager.numerical_columns,
            'categorical_columns': datamanager.categorical_columns,
            'target_columns': (1,),
            'issparse': False}

    dataset_properties = datamanager.get_dataset_properties(get_dataset_requirements(info))
    assert dataset_properties['n_prediction_steps'] == datamanager.n_prediction_steps
    assert dataset_properties['sp'] == datamanager.seasonality
    assert dataset_properties['freq'] == datamanager.freq
    assert isinstance(dataset_properties['input_shape'], Tuple)
    assert isinstance(dataset_properties['time_feature_transform'], List)
    for item in dataset_properties['time_feature_transform']:
        assert isinstance(item, Callable)
    assert dataset_properties['uni_variant'] == (get_fit_dictionary_forecasting['X_train'] is None)
    assert dataset_properties['targets_have_missing_values'] == \
           get_fit_dictionary_forecasting['y_train'].isnull().values.any()
    if get_fit_dictionary_forecasting['X_train'] is not None:
        assert dataset_properties['features_have_missing_values'] == \
               get_fit_dictionary_forecasting['X_train'].isnull().values.any()


def test_freq_valeus():
    freq = '1H'
    n_prediction_steps = 12

    seasonality, freq, freq_value = TimeSeriesForecastingDataset.compute_freq_values(freq, n_prediction_steps)
    assert seasonality == 24
    assert freq == '1H'
    assert freq_value == 24

    n_prediction_steps = 36
    seasonality, freq, freq_value = TimeSeriesForecastingDataset.compute_freq_values(freq, n_prediction_steps)
    assert seasonality == 24
    assert freq_value == 168

    freq = [2, 3, 4]
    n_prediction_steps = 10
    seasonality, freq, freq_value = TimeSeriesForecastingDataset.compute_freq_values(freq, n_prediction_steps)
    assert seasonality == 2
    assert freq == '1Y'
    assert freq_value == 4


def test_target_normalization():
    Y = [[1, 2], [3, 4, 5]]
    dataset = TimeSeriesForecastingDataset(None, Y, normalize_y=True)

    assert np.allclose(dataset.y_mean.values, np.vstack([np.mean(y) for y in Y]))
    assert np.allclose(dataset.y_std.values, np.vstack([np.std(y, ddof=1) for y in Y]))
    assert np.allclose(dataset.train_tensors[1].values.flatten(),
                       np.hstack([(y - np.mean(y))/np.std(y, ddof=1) for y in Y]))


@pytest.mark.parametrize("get_fit_dictionary_forecasting", ['uni_variant_wo_missing'], indirect=True)
def test_dataset_index(backend, get_fit_dictionary_forecasting):
    datamanager: TimeSeriesForecastingDataset = backend.load_datamanager()
    assert np.allclose(datamanager[5][0]['past_targets'][-1].numpy(), 5.0)
    assert np.allclose(datamanager[50][0]['past_targets'][-1].numpy(), 1005.0)
    assert np.allclose(datamanager[150][0]['past_targets'][-1].numpy(), 2050.0)
    assert np.allclose(datamanager[-1][0]['past_targets'][-1].numpy(), 9134.0)

    assert datamanager.get_time_series_seq(50) == datamanager.datasets[1]

    # test for validation indices
    val_indices = datamanager.splits[0][1]
    val_set = [datamanager.get_validation_set(val_idx) for val_idx in val_indices]
    val_targets = np.concatenate([val_seq[-1][1]['future_targets'].numpy() for val_seq in val_set])
    assert np.allclose(val_targets, datamanager.get_test_target(val_indices))


@pytest.mark.parametrize("get_fit_dictionary_forecasting", ['multi_variant_wo_missing'], indirect=True)
def test_update_dataset(backend, get_fit_dictionary_forecasting):
    datamanager: TimeSeriesForecastingDataset = backend.load_datamanager()
    X = datamanager.train_tensors[0]
    for col in X.columns:
        X[col] = X.index
    datamanager.replace_data(X, None)
    for i, data in enumerate(datamanager.datasets):
        assert np.allclose(data.X, np.ones_like(data.X) * i)

    datamanager.update_transform(ZeroTransformer(), train=True)
    assert np.allclose(datamanager[0][0]['past_features'].numpy(), np.zeros(len(X.columns)))
    assert datamanager.transform_time_features is False

    datamanager.transform_time_features = True
    for dataset in datamanager.datasets:
        assert dataset.transform_time_features is True
    seq_lengths = datamanager.sequence_lengths_train
    new_test_seq = datamanager.generate_test_seqs()
    for seq_len, test_seq in zip(seq_lengths, new_test_seq):
        # seq_len is len(y) - n_prediction_steps, here we expand X_test with another n_prediction_steps
        assert test_seq.X.shape[0] - seq_len == 2 * datamanager.n_prediction_steps


def test_splits():

    y = [np.arange(100 + i * 10) for i in range(10)]
    resampling_strategy_args = {'num_splits': 5}
    dataset = TimeSeriesForecastingDataset(None, y,
                                           resampling_strategy=CrossValTypes.time_series_ts_cross_validation,
                                           resampling_strategy_args=resampling_strategy_args,
                                           n_prediction_steps=10,
                                           freq='1M')
    assert len(dataset.splits) == 5
    assert dataset.splits[0][1][0] == (100 - 10 - 1)
    for split in dataset.splits:
        # We need to ensure that the training indices only interrupt at where the validation sets start, e.g.,
        #  the tail of each sequence
        assert len(np.unique(split[0] - np.arange(len(split[0])))) == len(y)
        assert np.all(split[1][1:] - split[1][:-1] == [100 + i * 10 for i in range(9)])
        assert len(split[1]) == len(y)

    y = [np.arange(100) for _ in range(10)]
    resampling_strategy_args = {'num_splits': 5,
                                'n_repeats': 2}
    dataset = TimeSeriesForecastingDataset(None, y,
                                           resampling_strategy=CrossValTypes.time_series_ts_cross_validation,
                                           resampling_strategy_args=resampling_strategy_args,
                                           n_prediction_steps=10,
                                           freq='1M')
    assert len(dataset.splits) == 5
    for split in dataset.splits:
        assert len(split[1]) == len(y) * 1

    y = [np.arange(40) for _ in range(10)]
    resampling_strategy_args = {'num_splits': 5}
    dataset = TimeSeriesForecastingDataset(None, y,
                                           resampling_strategy=CrossValTypes.time_series_ts_cross_validation,
                                           resampling_strategy_args=resampling_strategy_args,
                                           n_prediction_steps=10,
                                           freq='1M')
    # the length of each sequence does not support 5 splitions
    assert len(dataset.splits) == 3

    # datasets with long but little sequence
    y = [np.arange(4000) for _ in range(2)]
    dataset = TimeSeriesForecastingDataset(None, y,
                                           resampling_strategy=CrossValTypes.time_series_ts_cross_validation,
                                           n_prediction_steps=10,
                                           freq='1M')
    # the length of each sequence does not support 5 splits
    assert len(dataset.splits) == 2
    for split in dataset.splits:
        assert len(split[1]) == len(y) * 50

    resampling_strategy = CrossValTypes.time_series_cross_validation

    y = [np.arange(40) for _ in range(10)]
    resampling_strategy_args = {'num_splits': 5,
                                'n_repeats': 5}

    resampling_strategy, resampling_strategy_args = TimeSeriesForecastingDataset.get_split_strategy(
        [60] * 10,
        10,
        25,
        CrossValTypes.time_series_ts_cross_validation,
        resampling_strategy_args=resampling_strategy_args,
    )
    assert resampling_strategy_args['num_splits'] == 3
    assert resampling_strategy_args['n_repeats'] == 1

    resampling_strategy, resampling_strategy_args = TimeSeriesForecastingDataset.get_split_strategy(
        [15] * 10,
        10,
        25,
        CrossValTypes.time_series_cross_validation,
    )
    assert resampling_strategy == HoldoutValTypes.time_series_hold_out_validation

    resampling_strategy_args = {'num_splits': 5,
                                'n_repeats': 5}
    resampling_strategy, resampling_strategy_args = TimeSeriesForecastingDataset.get_split_strategy(
        [60] * 10,
        10,
        25,
        CrossValTypes.time_series_cross_validation,
        resampling_strategy_args=resampling_strategy_args,
    )
    assert resampling_strategy_args['num_splits'] == 4
    assert resampling_strategy_args['n_repeats'] == 1

    y = [np.arange(60) for _ in range(10)]
    dataset = TimeSeriesForecastingDataset(None, y,
                                           resampling_strategy=CrossValTypes.time_series_cross_validation,
                                           resampling_strategy_args=resampling_strategy_args,
                                           n_prediction_steps=10,
                                           freq='1M')
    assert len(dataset.splits) == 4

    refit_set = dataset.create_refit_set()
    assert len(refit_set.splits[0][0]) == len(refit_set)


def test_extract_time_features():
    feature_shapes = {'b': 5, 'a': 3, 'c': 7, 'd': 12}
    feature_names = ['a', 'b', 'c', 'd']
    queried_features = ('b', 'd')
    feature_index = extract_feature_index(feature_shapes, feature_names, queried_features)
    feature_index2 = []
    idx_tracker = 0
    for fea_name in feature_names:
        feature_s = feature_shapes[fea_name]
        if fea_name in queried_features:
            feature_index2.append(list(range(idx_tracker, idx_tracker + feature_s)))
        idx_tracker += feature_s

    assert feature_index == tuple(sum(feature_index2, []))
    # the value should not be relevant with the order of queried_features
    assert feature_index == extract_feature_index(feature_shapes, feature_names, ('d', 'b'))
