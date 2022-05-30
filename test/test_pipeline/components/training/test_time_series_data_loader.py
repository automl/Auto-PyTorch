import copy
import unittest
import unittest.mock
from typing import List
from unittest import mock

import numpy as np

import pandas as pd

import torch

import torchvision

from autoPyTorch.datasets.resampling_strategy import HoldOutFuncs, HoldoutValTypes
from autoPyTorch.datasets.time_series_dataset import TimeSeriesForecastingDataset, TimeSeriesSequence
from autoPyTorch.pipeline.components.training.data_loader.time_series_forecasting_data_loader import (
    TimeSeriesForecastingDataLoader
)
from autoPyTorch.pipeline.components.training.data_loader.time_series_util import (
    PadSequenceCollector,
    SequentialSubSetSampler,
    TestSequenceDataset,
    TimeSeriesSampler,
    pad_sequence_with_minimal_length
)
from autoPyTorch.utils.common import HyperparameterSearchSpace


class TestTimeSeriesForecastingDataLoader(unittest.TestCase):
    def setUp(self) -> None:
        feature_names = ['f1']
        feature_shapes = {'f1': 1}
        known_future_features = ('f1',)
        freq = '1Y'
        n_prediction_steps = 3

        sequence_lengths_train = [10, 20, 30, 40, 50, 60, 70, 80, 90, 1000]

        backend = unittest.mock.Mock()
        n_repeats = 2

        with mock.patch('autoPyTorch.datasets.time_series_dataset.TimeSeriesForecastingDataset') as MockDataSet:
            mockdataset = MockDataSet.return_value
            mockdataset.holdout_validators = HoldOutFuncs.get_holdout_validators(
                HoldoutValTypes.time_series_hold_out_validation
            )
            datasets = []
            mockdataset.sequence_lengths_train = sequence_lengths_train
            for seq_len in sequence_lengths_train:
                mock_ser = mock.MagicMock()
                mock_ser.__len__.return_value = seq_len
                datasets.append(mock_ser)
            mockdataset.datasets = datasets
            mockdataset.n_prediction_steps = n_prediction_steps

        split = TimeSeriesForecastingDataset.create_holdout_val_split(mockdataset,
                                                                      HoldoutValTypes.time_series_hold_out_validation,
                                                                      0.1,
                                                                      n_repeats=n_repeats)

        with mock.patch('autoPyTorch.datasets.time_series_dataset.TimeSeriesForecastingDataset') as MockDataSet:
            dataset = MockDataSet.return_value

            dataset.__len__.return_value = sum(sequence_lengths_train)
            datamanager = unittest.mock.MagicMock()
            datamanager.get_dataset.return_value = dataset
            datamanager.feature_names = ['f1']
            datamanager.splits.__getitem__.return_value = split

        dataset_properties = dict(feature_names=feature_names,
                                  feature_shapes=feature_shapes,
                                  known_future_features=known_future_features,
                                  freq=freq,
                                  is_small_preprocess=True,
                                  uni_variant=False,
                                  time_feature_transform=True,
                                  sequence_lengths_train=sequence_lengths_train,
                                  n_prediction_steps=n_prediction_steps,
                                  n_repeats=n_repeats)

        self.n_prediction_steps = n_prediction_steps

        backend.load_datamanager.return_value = datamanager
        self.fit_dictionary = {
            'dataset_properties': dataset_properties,
            'lagged_value': [1, 2, 3],
            'X_train': pd.DataFrame([0.] * sum(sequence_lengths_train)),
            'y_train': pd.DataFrame([0.] * sum(sequence_lengths_train)),
            'train_indices': split[0],
            'test_indices': split[1],
            'working_dir': '/tmp',
            'backend': backend,
            'split_id': 0,
        }

    def test_get_set_config_space(self):
        """
        Makes sure that the configuration space of the base data loader
        is properly working"""
        loader = TimeSeriesForecastingDataLoader()

        dataset_properties = {'seq_length_max': 70}
        cs = loader.get_hyperparameter_search_space(dataset_properties)
        self.assertEqual(cs.get_hyperparameter('window_size').upper, 50)

        dataset_properties = {'seq_length_max': 25}
        cs = loader.get_hyperparameter_search_space(dataset_properties)
        self.assertEqual(cs.get_hyperparameter('window_size').upper, 25)
        self.assertEqual(cs.get_hyperparameter('window_size').default_value, 25)

        dataset_properties = {'seq_length_max': 20}
        cs = loader.get_hyperparameter_search_space(dataset_properties)
        self.assertEqual(cs.get_hyperparameter('window_size').upper, 20)
        self.assertEqual(cs.get_hyperparameter('window_size').lower, 1)

        dataset_properties = {'seq_length_max': 10}
        cs = loader.get_hyperparameter_search_space(dataset_properties)
        self.assertEqual(cs.get_hyperparameter('window_size').upper, 10)
        self.assertEqual(cs.get_hyperparameter('window_size').lower, 1)

        cs = loader.get_hyperparameter_search_space(dataset_properties,
                                                    window_size=HyperparameterSearchSpace(hyperparameter='window_size',
                                                                                          value_range=(2, 5),
                                                                                          default_value=3))

        self.assertEqual(cs.get_hyperparameter('window_size').upper, 5)
        self.assertEqual(cs.get_hyperparameter('window_size').lower, 2)

        for _ in range(5):
            sample = cs.sample_configuration()
            self.assertTrue(
                ('backcast_period' in sample) ^ ('window_size' in sample)
            )

    def test_base_fit(self):
        """ Makes sure that fit and transform work as intended """
        fit_dictionary = copy.copy(self.fit_dictionary)

        # Mock child classes requirements
        loader = TimeSeriesForecastingDataLoader()
        loader.build_transform = unittest.mock.Mock()
        loader._check_transform_requirements = unittest.mock.Mock()

        loader.fit(fit_dictionary)

        # Fit means that we created the data loaders
        self.assertIsInstance(loader.train_data_loader, torch.utils.data.DataLoader)
        self.assertIsInstance(loader.val_data_loader, torch.utils.data.DataLoader)

        # Transforms adds this fit dictionaries
        transformed_fit_dictionary = loader.transform(fit_dictionary)
        self.assertIn('train_data_loader', transformed_fit_dictionary)
        self.assertIn('val_data_loader', transformed_fit_dictionary)

        self.assertEqual(transformed_fit_dictionary['train_data_loader'],
                         loader.train_data_loader)
        self.assertEqual(transformed_fit_dictionary['val_data_loader'],
                         loader.val_data_loader)
        self.assertEqual(transformed_fit_dictionary['window_size'], loader.window_size)

    def test_build_transform_small_preprocess_true(self):
        """
        Makes sure a proper composition is created
        """
        loader = TimeSeriesForecastingDataLoader()
        fit_dictionary = copy.deepcopy(self.fit_dictionary)
        fit_dictionary['dataset_properties']['is_small_preprocess'] = True
        for thing in ['imputer', 'scaler', 'encoder']:
            fit_dictionary[thing] = [unittest.mock.Mock()]

        compose = loader.build_transform(fit_dictionary, mode='train')

        self.assertIsInstance(compose, torchvision.transforms.Compose)

        # No preprocessing needed here as it was done before
        self.assertEqual(len(compose.transforms), 1)

    def test_build_transform_small_preprocess_false(self):
        """
        Makes sure a proper composition is created
        """
        loader = TimeSeriesForecastingDataLoader()
        fit_dictionary = copy.deepcopy(self.fit_dictionary)
        fit_dictionary['dataset_properties']['is_small_preprocess'] = False
        fit_dictionary['preprocess_transforms'] = [unittest.mock.Mock()]

        compose = loader.build_transform(fit_dictionary, mode='train')

        self.assertIsInstance(compose, torchvision.transforms.Compose)

        # We expect the expand_transformer and Mock
        self.assertEqual(len(compose.transforms), 2)

    def test_adjust_window_size(self):
        window_size = 2
        n_prediction_steps = 5
        backcast_period = 3
        time_series_dataloader = TimeSeriesForecastingDataLoader(batch_size=1,
                                                                 window_size=window_size,
                                                                 n_prediction_steps=n_prediction_steps)
        self.assertEqual(time_series_dataloader.window_size, window_size)

        time_series_dataloader = TimeSeriesForecastingDataLoader(batch_size=1,
                                                                 backcast=True,
                                                                 backcast_period=backcast_period,
                                                                 window_size=window_size,
                                                                 n_prediction_steps=n_prediction_steps)
        self.assertEqual(time_series_dataloader.window_size, backcast_period * n_prediction_steps)

        sample_interval = 3
        self.assertEqual(time_series_dataloader.adjust_window_size(sample_interval),
                         (backcast_period * n_prediction_steps) // sample_interval)

    @mock.patch("autoPyTorch.pipeline.components.training.data_loader.time_series_util.TimeSeriesSampler.__init__",
                spec=True)
    def test_compute_expected_num_instances_per_seq(self, sampler_mock_init):
        sampler_mock_init.return_value = None
        batch_size = 5
        window_size = 5
        num_batches_per_epoch = 4
        time_series_dataloader = TimeSeriesForecastingDataLoader(batch_size=batch_size,
                                                                 window_size=window_size,
                                                                 num_batches_per_epoch=num_batches_per_epoch)
        fit_dictionary = copy.copy(self.fit_dictionary)
        time_series_dataloader = time_series_dataloader.fit(fit_dictionary)

        self.assertEqual(time_series_dataloader.window_size, window_size)
        self.assertEqual(time_series_dataloader.known_future_features_index, (0,))

        sampler = time_series_dataloader.sampler_train
        self.assertIsInstance(sampler, TimeSeriesSampler)
        train_split = fit_dictionary['train_indices']
        self.assertEqual(len(train_split), len(sampler_mock_init.call_args[1]['indices']))

        train_seq_length = fit_dictionary['dataset_properties']['sequence_lengths_train']

        seq_lengths = []
        for train_seq_len in train_seq_length:
            n_train_seq = len(
                HoldOutFuncs.time_series_hold_out_validation(
                    None, None,
                    np.arange(train_seq_len),
                    n_prediction_steps=fit_dictionary['dataset_properties']['n_prediction_steps'],
                    n_repeats=fit_dictionary['dataset_properties']['n_repeats'])[0])
            if n_train_seq > 0:
                seq_lengths.append(n_train_seq)
        self.assertTrue(np.all(seq_lengths == sampler_mock_init.call_args[1]['seq_lengths']))

        num_instances_per_seqs_full = sampler_mock_init.call_args[1]['num_instances_per_seqs']
        unique_num_instances_per_seqs = np.unique(num_instances_per_seqs_full)
        self.assertEqual(len(unique_num_instances_per_seqs), 1)

        self.assertAlmostEqual(unique_num_instances_per_seqs.item(),
                               num_batches_per_epoch * batch_size / len(seq_lengths))

        self.assertEqual(sampler_mock_init.call_args[1]['min_start'],
                         fit_dictionary['dataset_properties']['n_prediction_steps'])

        num_instances_dataset = sum(train_seq_length)
        seq_train_length = seq_lengths
        min_start = fit_dictionary['dataset_properties']['n_prediction_steps']

        fraction_seq = 0.3
        num_instances_per_seqs_frac_seq = time_series_dataloader.compute_expected_num_instances_per_seq(
            num_instances_dataset,
            seq_train_length,
            min_start, fraction_seq)
        instances_to_be_sampled = np.where(num_instances_per_seqs_frac_seq)[0]
        self.assertEqual(len(instances_to_be_sampled), int(np.ceil(fraction_seq * len(seq_train_length))))
        self.assertAlmostEqual(np.unique(num_instances_per_seqs_frac_seq[instances_to_be_sampled]),
                               unique_num_instances_per_seqs)

        fraction_samples_per_seq = 0.3
        num_instances_per_seqs_frac_per_seq = time_series_dataloader.compute_expected_num_instances_per_seq(
            num_instances_dataset,
            seq_train_length,
            min_start,
            fraction_samples_per_seq=fraction_samples_per_seq)
        self.assertTrue(np.allclose(num_instances_per_seqs_frac_per_seq,
                                    fraction_samples_per_seq * num_instances_per_seqs_full))

        time_series_dataloader.sample_strategy = 'LengthUniform'

        seq_lengths_reduced = np.asarray(seq_lengths) - min_start
        seq_lengths_reduced = np.where(seq_lengths_reduced <= 0, 0, seq_lengths_reduced)

        num_instances_per_seqs_full = time_series_dataloader.compute_expected_num_instances_per_seq(
            num_instances_dataset,
            seq_train_length,
            min_start)

        self.assertTrue(
            np.allclose(num_instances_per_seqs_full,
                        batch_size * num_batches_per_epoch * seq_lengths_reduced / np.sum(seq_lengths_reduced))
        )

        fraction_seq = 0.3
        num_instances_per_seqs_frac_seq = time_series_dataloader.compute_expected_num_instances_per_seq(
            num_instances_dataset,
            seq_train_length,
            min_start, fraction_seq)
        instances_to_be_sampled = np.where(num_instances_per_seqs_frac_seq)[0]

        self.assertTrue(np.allclose(np.unique(num_instances_per_seqs_frac_seq[instances_to_be_sampled]),
                                    num_instances_per_seqs_full[instances_to_be_sampled]))

        fraction_samples_per_seq = 0.3
        num_instances_per_seqs_frac_per_seq = time_series_dataloader.compute_expected_num_instances_per_seq(
            num_instances_dataset,
            seq_train_length,
            min_start,
            fraction_samples_per_seq=fraction_samples_per_seq)
        self.assertTrue(np.allclose(num_instances_per_seqs_frac_per_seq,
                                    fraction_samples_per_seq * num_instances_per_seqs_full))

    @mock.patch("autoPyTorch.pipeline.components.training.data_loader.time_series_util.TestSequenceDataset.__init__",
                spec=True)
    def test_get_loader(self, loader_init_mock):
        loader_init_mock.return_value = None
        batch_size = 5
        window_size = 5
        num_batches_per_epoch = 4
        time_series_dataloader = TimeSeriesForecastingDataLoader(batch_size=batch_size,
                                                                 window_size=window_size,
                                                                 num_batches_per_epoch=num_batches_per_epoch)
        fit_dictionary = copy.copy(self.fit_dictionary)
        time_series_dataloader.fit(fit_dictionary)
        x_test = TimeSeriesSequence(X=np.array([1, 2, 3, 4, 5]),
                                    Y=np.array([1, 2, 3, 4, 5]),
                                    X_test=np.array([1, 2, 3]))
        test_loader = time_series_dataloader.get_loader(X=copy.deepcopy(x_test))
        self.assertIsInstance(test_loader, torch.utils.data.DataLoader)
        self.assertIsInstance(test_loader.dataset, TestSequenceDataset)
        test_set = loader_init_mock.call_args[0][0]
        self.assertIsInstance(test_set, List)
        self.assertEqual(len(test_set), 1)

        x_test = [x_test, x_test]
        _ = time_series_dataloader.get_loader(X=copy.deepcopy(x_test))
        test_set = loader_init_mock.call_args[0][0]
        self.assertEqual(len(test_set), len(x_test))

        for seq in test_set:
            self.assertIsInstance(seq, TimeSeriesSequence)
            self.assertTrue(seq.is_test_set)
            self.assertEqual(seq.freq, time_series_dataloader.freq)

        class DummyEncoder:
            def fit(self, data):
                return self

            def transform(self, data: pd.DataFrame):
                return np.concatenate([data.values, data.values], axis=-1)

        transform = DummyEncoder()
        time_series_dataloader.feature_preprocessor = transform
        x_test_copy = copy.deepcopy(x_test)
        _ = time_series_dataloader.get_loader(X=x_test_copy)

        test_set = loader_init_mock.call_args[0][0]
        for seq_raw, seq in zip(x_test, test_set):
            self.assertTrue(seq.X.shape[-1] == 2 * seq_raw.X.shape[-1])

        # ensure that we do not transform the dataset twice
        _ = time_series_dataloader.get_loader(X=x_test_copy)
        test_set = loader_init_mock.call_args[0][0]
        for seq_raw, seq in zip(x_test, test_set):
            self.assertTrue(seq.X.shape[-1] == 2 * seq_raw.X.shape[-1])


class TestTimeSeriesUtil(unittest.TestCase):
    def test_test_seq_length(self):
        x_test = TimeSeriesSequence(X=np.array([0, 1, 2, 3, 4]),
                                    Y=np.array([1, 2, 3, 4, 5]),
                                    X_test=np.array([1, 2, 3]),
                                    n_prediction_steps=3,
                                    is_test_set=True)
        x_test = [x_test, x_test]
        test_set = TestSequenceDataset(x_test)
        self.assertEqual(len(test_set), len(x_test))
        self.assertTrue(np.allclose(test_set[0][0]['past_targets'].numpy(), x_test[0].Y))

    def test_pad_sequence_with_minimal_length(self):
        sequences = [torch.ones([10, 1]),
                     torch.ones([3, 1]),
                     torch.ones([17, 1])]
        pad_seq_1 = pad_sequence_with_minimal_length(sequences, 5)
        self.assertEqual(list(pad_seq_1.shape), [3, 17, 1])
        self.assertTrue(torch.all(pad_seq_1[0] == torch.tensor([0.] * 7 + [1.] * 10).unsqueeze(-1)))

        pad_seq_2 = pad_sequence_with_minimal_length(sequences, 5, batch_first=False)
        self.assertEqual(list(pad_seq_2.shape), [17, 3, 1])
        self.assertTrue(torch.all(pad_seq_2[:, 0] == torch.tensor([0.] * 7 + [1.] * 10).unsqueeze(-1)))

        pad_seq_3 = pad_sequence_with_minimal_length(sequences, 5, padding_value=0.5)
        self.assertTrue(torch.all(pad_seq_3[0] == torch.tensor([0.5] * 7 + [1.] * 10).unsqueeze(-1)))

        pad_seq_4 = pad_sequence_with_minimal_length(sequences, 5, 10)
        self.assertEqual(list(pad_seq_4.shape), [3, 10, 1])
        self.assertTrue(torch.all(pad_seq_4[0] == torch.ones(10).unsqueeze(-1)))
        self.assertTrue(torch.all(pad_seq_4[1] == torch.tensor([0] * 7 + [1.] * 3).unsqueeze(-1)))
        self.assertTrue(torch.all(pad_seq_4[2] == torch.ones(10).unsqueeze(-1)))

        pad_seq_5 = pad_sequence_with_minimal_length(sequences, 20)
        self.assertEqual(list(pad_seq_5.shape), [3, 20, 1])
        self.assertTrue(torch.all(pad_seq_5[0] == torch.tensor([0] * 10 + [1.] * 10).unsqueeze(-1)))
        self.assertTrue(torch.all(pad_seq_5[1] == torch.tensor([0] * 17 + [1.] * 3).unsqueeze(-1)))
        self.assertTrue(torch.all(pad_seq_5[2] == torch.tensor([0] * 3 + [1.] * 17).unsqueeze(-1)))

        sequences = [torch.ones(3, dtype=torch.bool),
                     torch.ones(15, dtype=torch.bool)]
        pad_seq_6 = pad_sequence_with_minimal_length(sequences, 5)
        self.assertTrue(pad_seq_6.dtype == torch.bool)
        self.assertTrue(torch.all(pad_seq_6[0] == torch.tensor([False] * 12 + [True] * 3, dtype=torch.bool)))

    def test_pad_sequence_controller(self):
        window_size = 3
        seq_max_length = 5
        target_padding_value = 0.5
        pad_seq_controller = PadSequenceCollector(window_size=window_size,
                                                  sample_interval=1,
                                                  target_padding_value=target_padding_value,
                                                  seq_max_length=seq_max_length)
        n_prediction_steps = 2
        seq = TimeSeriesSequence(np.arange(10).astype(np.float), np.arange(10).astype(np.float),
                                 n_prediction_steps=n_prediction_steps)
        features_padded = pad_seq_controller([seq[0][0], seq[-1][0]])
        past_targets = features_padded['past_targets']
        past_features = features_padded['past_features']
        self.assertEqual(list(past_targets.shape), [2, seq_max_length])
        self.assertEqual(list(past_features.shape), [2, seq_max_length, 1])
        self.assertTrue(features_padded['past_observed_targets'].dtype == torch.bool)
        self.assertTrue(features_padded['decoder_lengths'].dtype == torch.int64)

        self.assertTrue(torch.all(torch.ones(seq_max_length - 1) * target_padding_value == past_targets[0, :-1]))
        self.assertTrue(torch.all(torch.zeros(seq_max_length - 1) == past_features[0, :-1]))

        targets_padded = pad_seq_controller([seq[0][1], seq[-1][1]])
        self.assertTrue(list(targets_padded['future_targets']), [2, n_prediction_steps])

        features_padded = pad_seq_controller([seq[0][0], seq[0][0]])
        self.assertEqual(list(features_padded['past_targets'].shape), [2, window_size])

        pad_seq_controller.sample_interval = 2
        features_padded = pad_seq_controller([seq[0][0], seq[-1][0]])
        self.assertEqual(list(features_padded['past_targets'].shape), [2, 3])

        self.assertTrue(torch.all(
            pad_seq_controller([{'x': 0}, {'x': 1}])['x'] == torch.Tensor([0, 1]))
        )
        self.assertTrue(torch.all(
            pad_seq_controller([{'x': np.array(0)}, {'x': np.array(1)}])['x'] == torch.Tensor([0, 1]))
        )

    def test_time_series_sampler(self):
        indices = np.arange(100)
        seq_lengths = [5, 10, 15, 20, 50]
        num_instances_per_seqs = [3.3, 1.3, 0.0, 10, 20.1]

        sampler = TimeSeriesSampler(indices, seq_lengths, num_instances_per_seqs, min_start=2)
        self.assertEqual(sampler.num_instances, int(np.round(np.sum(num_instances_per_seqs))))
        # The first sequence does not contain enough data to allow 3.3 sequences, so it only has 1 interval
        # For the others, Interval should be np.floor(n_inst) + 1 (resulting in  np.floor(n_inst) intervals)

        self.assertEqual(list(map(len, sampler.seq_intervals_int)), [1, 2, 1, 10, 21])
        self.assertTrue(torch.equal(sampler.seq_intervals_decimal, torch.tensor([[2, 5],
                                                                                 [7, 11],
                                                                                 [17, 30],
                                                                                 [32, 33],
                                                                                 [52, 54]])))
        self.assertTrue(
            torch.allclose(sampler.num_expected_ins_decimal,
                           torch.Tensor(
                               [3.3000e+00, 3.0000e-01, 1.0000e-08, 1.0000e-08, 1.0000e-01]).type(torch.float64))
        )

        for i in range(5):
            samples = torch.stack(list(sampler)).sort()[0].numpy()
            for seq_intervals_int in sampler.seq_intervals_int:
                if len(seq_intervals_int) > 1:
                    for i in range(len(seq_intervals_int) - 1):
                        self.assertTrue(
                            len(np.where((seq_intervals_int[i] < samples) & (samples < seq_intervals_int[i + 1]))) == 1
                        )

    def test_sequential_sub_set_sampler(self):
        n_samples = 5
        n_indices = np.arange(100)
        sampler = SequentialSubSetSampler(n_indices, n_samples)
        self.assertEqual(len(sampler), n_samples)
        self.assertEqual(len(list(sampler)), n_samples)

        sampler = SequentialSubSetSampler(n_indices, 150)
        self.assertEqual(len(sampler), len(n_indices))
        self.assertEqual(len(list(sampler)), len(n_indices))
