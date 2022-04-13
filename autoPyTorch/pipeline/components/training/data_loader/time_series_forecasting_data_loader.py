from typing import Any, Dict, Optional, Union, Sequence, List, Iterator, Sized
import warnings
from functools import partial

from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformIntegerHyperparameter, CategoricalHyperparameter, Constant
from ConfigSpace.conditions import EqualsCondition

import numpy as np

import torch
from torch.utils.data.sampler import SubsetRandomSampler

import torchvision

from autoPyTorch.datasets.time_series_dataset import TimeSeriesForecastingDataset, TimeSeriesSequence
from autoPyTorch.utils.common import (
    HyperparameterSearchSpace,
    custom_collate_fn,
    add_hyperparameter,
    get_hyperparameter
)

from autoPyTorch.pipeline.components.training.data_loader.feature_data_loader import FeatureDataLoader
from autoPyTorch.pipeline.components.training.data_loader.time_series_util import (
    TestSequenceDataset,
    PadSequenceCollector,
    TimeSeriesSampler,
    SequentialSubSetSampler,
    ExpandTransformTimeSeries
)


class TimeSeriesForecastingDataLoader(FeatureDataLoader):
    """This class is an interface to read time sequence data

    It gives the possibility to read various types of mapped
    datasets as described in:
    https://pytorch.org/docs/stable/data.html
    """

    def __init__(self,
                 batch_size: int = 64,
                 backcast: bool = False,
                 backcast_period: int = 2,
                 window_size: int = 1,
                 num_batches_per_epoch: Optional[int] = 50,
                 n_prediction_steps: int = 1,
                 sample_strategy='seq_uniform',
                 transform_time_features=False,
                 random_state: Optional[np.random.RandomState] = None) -> None:
        """
        initialize a dataloader
        Args:
            batch_size: batch size
            sequence_length: length of each sequence
            sample_interval: sample interval ,its value is the interval of the resolution

            num_batches_per_epoch: how
            n_prediction_steps: how many steps to predict in advance
        """
        super().__init__(batch_size=batch_size, random_state=random_state)
        self.backcast = backcast
        self.backcast_period = backcast_period
        if not backcast:
            self.window_size: int = window_size
        else:
            self.window_size: int = backcast_period * n_prediction_steps
        self.n_prediction_steps = n_prediction_steps
        self.sample_interval = 1
        # length of the tail, for instance if a sequence_length = 2, sample_interval =2, n_prediction = 2,
        # the time sequence should look like: [X, y, X, y, y] [test_data](values in tail is marked with X)
        # self.subseq_length = self.sample_interval * (self.window_size - 1) + 1
        self.sample_strategy = sample_strategy
        self.subseq_length = self.window_size
        self.num_batches_per_epoch = num_batches_per_epoch if num_batches_per_epoch is not None else np.inf
        self.padding_collector = None

        self.static_features = None
        self.known_future_features = None
        self._is_uni_variant = False

        self.transform_time_features = transform_time_features
        self.freq = "1Y"
        self.time_feature_transform = []

    def fit(self, X: Dict[str, Any], y: Any = None) -> torch.utils.data.DataLoader:
        """
        Fits a component by using an input dictionary with pre-requisites

        Args:
            X (X: Dict[str, Any]): Dependencies needed by current component to perform fit
            y (Any): not used. To comply with sklearn API

        Returns:
            A instance of self
        """
        self.check_requirements(X, y)

        # Incorporate the transform to the dataset
        datamanager = X['backend'].load_datamanager()  # type: TimeSeriesForcecastingDataset

        self.n_prediction_steps = datamanager.n_prediction_steps
        if self.backcast:
            self.window_size = self.backcast_period * self.n_prediction_steps

        self.static_features = datamanager.static_features
        self.known_future_features = datamanager.known_future_features

        # this value corresponds to budget type resolution
        sample_interval = X.get('sample_interval', 1)
        padding_value = X.get('required_padding_value', 0.0)


        if sample_interval > 1:
            # for lower resolution, window_size should be smaller
            self.window_size = (self.window_size - 1) // sample_interval + 1

        max_lagged_value = max(X['dataset_properties'].get('lagged_value', [np.inf]))
        max_lagged_value += self.window_size + self.n_prediction_steps


        self.padding_collector = PadSequenceCollector(self.window_size, sample_interval, padding_value, max_lagged_value)

        # this value corresponds to budget type num_sequence
        fraction_seq = X.get('fraction_seq', 1.0)
        # this value corresponds to budget type num_sample_per_seq
        fraction_samples_per_seq = X.get('fraction_samples_per_seq', 1.0)
        self.sample_interval = sample_interval

        # TODO, consider bucket setting
        self.train_transform = self.build_transform(X, mode='train')
        self.val_transform = self.build_transform(X, mode='val')
        self.test_transform = self.build_transform(X, mode='test')
        datamanager.update_transform(
            self.train_transform,
            train=True,
        )
        datamanager.update_transform(
            self.val_transform,
            train=False,
        )
        datamanager.transform_time_features = self.transform_time_features
        if X['dataset_properties']["is_small_preprocess"]:
            # This parameter indicates that the data has been pre-processed for speed
            # Overwrite the datamanager with the pre-processes data
            datamanager.replace_data(X['X_train'], X['X_test'] if 'X_test' in X else None)
            self.dataset_small_preprocess = True
        else:
            self.dataset_small_preprocess = False

        self._is_uni_variant = X['dataset_properties']['uni_variant']

        self.freq = X['dataset_properties']['freq']
        self.time_feature_transform = X['dataset_properties']['time_feature_transform']

        train_dataset = datamanager.get_dataset(split_id=X['split_id'], train=True)
        val_dataset = datamanager.get_dataset(split_id=X['split_id'], train=False)

        train_split, test_split = datamanager.splits[X['split_id']]

        num_instances_dataset = np.size(train_split)
        num_instances_train = self.num_batches_per_epoch * self.batch_size

        # get the length of each sequence of training data (after split), as we know that validation sets are always
        # place on the tail of the series, the discontinuity only happens if a new series is concated.
        # for instance, if we have a train indices is experssed as [0, 1, 2 ,3, 7 ,8 ].
        #  A new sequence must start from the index 7. We could then split each unique values to represent the length
        # of each split

        # TODO consider min_start as a hp (multiple of self.n_prediction_steps?)
        min_start = self.n_prediction_steps

        dataset_seq_length_train_all = X['dataset_properties']['sequence_lengths_train']
        if np.sum(dataset_seq_length_train_all) == len(train_split):
            # this works if we want to fit the entire datasets
            seq_train_length = np.array(dataset_seq_length_train_all)
        else:
            _, seq_train_length = np.unique(train_split - np.arange(len(train_split)), return_counts=True)
        # create masks for masking
        seq_idx_inactivate = np.where(self.random_state.rand(seq_train_length.size) > fraction_seq)[0]
        if len(seq_idx_inactivate) == seq_train_length.size:
            seq_idx_inactivate = self.random_state.choice(seq_idx_inactivate, len(seq_idx_inactivate)-1, replace=False )
        # this budget will reduce the number of samples inside each sequence, e.g., the samples becomes more sparse

        """
        num_instances_per_seqs = np.ceil(
            np.ceil(num_instances_train / (num_instances_dataset - min_start) * seq_train_length) *
            fraction_samples_per_seq
        )
        """
        if self.sample_strategy == 'LengthUniform':
            available_seq_length = seq_train_length - min_start
            available_seq_length = np.where(available_seq_length <= 1, 1, available_seq_length)
            num_instances_per_seqs = num_instances_train / num_instances_dataset * available_seq_length
        elif self.sample_strategy == 'SeqUniform':
            num_seq_train = len(seq_train_length)
            num_instances_per_seqs = np.repeat(num_instances_train / num_seq_train, num_seq_train)
        else:
            raise NotImplementedError(f'Unsupported sample strategy: {self.sample_strategy}')

        num_instances_per_seqs[seq_idx_inactivate] = 0
        num_instances_per_seqs *= fraction_samples_per_seq

        # num_instances_per_seqs = num_instances_per_seqs.astype(seq_train_length.dtype)
        # at least one element of each sequence should be selected

        # TODO consider the case where num_instances_train is greater than num_instances_dataset,
        # In which case we simply iterate through all the datasets

        sampler_indices_train = np.arange(num_instances_dataset)

        self.sampler_train = TimeSeriesSampler(indices=sampler_indices_train, seq_lengths=seq_train_length,
                                               num_instances_per_seqs=num_instances_per_seqs,
                                               min_start=min_start)

        self.train_data_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=min(self.batch_size, len(sampler_indices_train)),
            shuffle=False,
            num_workers=X.get('num_workers', 0),
            pin_memory=X.get('pin_memory', True),
            drop_last=X.get('drop_last', True),
            collate_fn=partial(custom_collate_fn, x_collector=self.padding_collector),
            sampler=self.sampler_train,
        )

        num_samples_val = int(np.sum(num_instances_per_seqs)) // 5
        if num_samples_val > len(val_dataset):
            sampler_val = None
        else:
            sampler_val = SequentialSubSetSampler(data_source=val_dataset,
                                                  num_samples=num_samples_val)

        self.val_data_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=min(1000, len(val_dataset)),
            shuffle=False,
            num_workers=X.get('num_workers', 0),
            pin_memory=X.get('pin_memory', True),
            drop_last=X.get('drop_last', False),
            collate_fn=partial(custom_collate_fn, x_collector=self.padding_collector),
            sampler=sampler_val
        )
        return self

    def transform(self, X: Dict) -> Dict:
        X.update({"window_size": self.window_size,
                  'transform_time_features': self.transform_time_features})
        return super().transform(X)

    def build_transform(self, X: Dict[str, Any], mode: str) -> torchvision.transforms.Compose:
        """
        Method to build a transformation that can pre-process input data

        Args:
            X (X: Dict[str, Any]): Dependencies needed by current component to perform fit
            mode (str): train/val/test

        Returns:
            A composition of transformations
        """

        if mode not in ['train', 'val', 'test']:
            raise ValueError("Unsupported mode provided {}. ".format(mode))

        candidate_transformations = []  # type: List[Callable]

        # if 'test' in mode or not X['dataset_properties']['is_small_preprocess']:
        #    candidate_transformations.extend(X['preprocess_transforms'])

        candidate_transformations.append(ExpandTransformTimeSeries())
        if "test" in mode or not X['dataset_properties']['is_small_preprocess']:
            if "preprocess_transforms" in X:
                candidate_transformations.extend(X['preprocess_transforms'])

        # We transform to tensor under dataset
        return torchvision.transforms.Compose(candidate_transformations)

    def get_loader(self, X: Union[np.ndarray, TimeSeriesSequence], y: Optional[np.ndarray] = None,
                   batch_size: int = np.inf,
                   ) -> torch.utils.data.DataLoader:
        """
        Creates a data loader object from the provided data,
        applying the transformations meant to validation objects
        This is a lazy loaded test set, each time only one piece of series
        """
        if isinstance(X, TimeSeriesSequence):
            X = [X]
        if isinstance(X, List):
            for x_seq in X:
                if not isinstance(x_seq, TimeSeriesSequence):
                    raise NotImplementedError('Test Set must be a TimeSeriesSequence or a'
                                              ' list of time series objects!')
                if x_seq.freq != self.freq:
                    # WE need to recompute the cached time features (However, this should not happen)
                    x_seq._cached_time_features = None

                x_seq.update_transform(self.test_transform, train=False)
                x_seq.update_attribute(freq=self.freq,
                                       transform_time_features=self.transform_time_features,
                                       time_feature_transform=self.time_feature_transform,
                                       static_features=self.static_features,
                                       known_future_features=self.known_future_features,
                                       )
                if self.transform_time_features:
                    x_seq.compute_time_features()

                x_seq.freq = self.freq
                x_seq.update_transform(self.test_transform, train=False)
        else:
            raise NotImplementedError('Unsupported data type for time series data loader!')

        dataset = X
        dataset_test = TestSequenceDataset(dataset, train=False)

        return torch.utils.data.DataLoader(
            dataset_test,
            batch_size=min(batch_size, len(dataset)),
            shuffle=False,
            collate_fn=partial(custom_collate_fn, x_collector=self.padding_collector),
        )

    def get_train_data_loader(self) -> torch.utils.data.DataLoader:
        """Returns a data loader object for the train data

        Returns:
            torch.utils.data.DataLoader: A train data loader
        """
        assert self.train_data_loader is not None, "No train data loader fitted"
        return self.train_data_loader

    def get_val_data_loader(self) -> torch.utils.data.DataLoader:
        """Returns a data loader object for the validation data

        Returns:
            torch.utils.data.DataLoader: A validation data loader
        """
        assert self.val_data_loader is not None, "No val data loader fitted"
        return self.val_data_loader

    def get_test_data_loader(self) -> torch.utils.data.DataLoader:
        """Returns a data loader object for the test data

        Returns:
            torch.utils.data.DataLoader: A validation data loader
        """
        return self.test_data_loader

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties: Optional[Dict] = None,
                                        batch_size: HyperparameterSearchSpace =
                                        HyperparameterSearchSpace(hyperparameter="batch_size",
                                                                  value_range=(32, 320),
                                                                  default_value=64),
                                        window_size: HyperparameterSearchSpace =
                                        HyperparameterSearchSpace(hyperparameter='window_size',
                                                                  value_range=(20, 50),
                                                                  default_value=30),
                                        num_batch_per_epoch: HyperparameterSearchSpace =
                                        HyperparameterSearchSpace(hyperparameter="num_batches_per_epoch",
                                                                  value_range=(30, 100),
                                                                  default_value=50),
                                        sample_strategy: HyperparameterSearchSpace =
                                        HyperparameterSearchSpace(hyperparameter="sample_strategy",
                                                                  value_range=('LengthUniform', 'SeqUniform'),
                                                                  default_value='SeqUniform'),
                                        backcast: HyperparameterSearchSpace =
                                        HyperparameterSearchSpace(hyperparameter='backcast',
                                                                  value_range=(True, False),
                                                                  default_value=False),
                                        backcast_period: HyperparameterSearchSpace =
                                        HyperparameterSearchSpace(hyperparameter='backcast_period',
                                                                  value_range=(1, 7),
                                                                  default_value=2),
                                        transform_time_features: HyperparameterSearchSpace =
                                        HyperparameterSearchSpace(hyperparameter='transform_time_features',
                                                                  value_range=(True, False),
                                                                  default_value=False)
                                        ) -> ConfigurationSpace:
        """
        hyperparameter search space for forecasting dataloader. Forecasting dataloader construct the window size in two
        ways: either window_size is directly assigned or it is computed by backcast_period * n_prediction_steps
        (introduced by nbeats:
        Oreshkin et al., N-BEATS: Neural basis expansion analysis for interpretable time series forecasting, ICLR 2020
        https://arxiv.org/abs/1905.10437)
        Currently back_cast_period is only activate when back_cast is activate
        Args:
            dataset_properties (Optional[Dict]): dataset properties
            batch_size (int): batch size
            window_size (int): window size, (if activate) this value directly determines the window_size of the
                               data loader
            num_batch_per_epoch (int): how many batches are trained at each iteration
            sample_strategy(str): how samples are distributed. if it is LengthUnifrom, then every single data point
                                  has the same probability to be sampled, in which case longer sequence will occupy more
                                  samples. If it is SeqUniform, then every sequence has the same probability to be
                                  sampled regardless of their length
            backcast (bool): if back_cast module is activate (in which case window size is a
            multiple of n_prediction_steps)
            backcast_period (int): activate if backcast is activate, the window size is then computed with
                                   backcast_period * n_prediction_steps
            transform_time_features (bool) if time feature trasnformation is applied

        Returns:
            cs: Configuration Space

        """
        cs = ConfigurationSpace()
        add_hyperparameter(cs, batch_size, UniformIntegerHyperparameter)
        add_hyperparameter(cs, num_batch_per_epoch, UniformIntegerHyperparameter)
        add_hyperparameter(cs, sample_strategy, CategoricalHyperparameter)

        seq_length_max = dataset_properties.get('seq_length_max', np.inf)

        if seq_length_max <= window_size.value_range[1]:
            if seq_length_max <= window_size.value_range[0]:
                warnings.warn('The base window_size is larger than the maximal sequence length in the dataset,'
                              'we simply set it as a constant value with maximal sequence length')
                window_size = HyperparameterSearchSpace(hyperparameter=window_size.hyperparameter, 
                                                        value_range=(seq_length_max, ),
                                                        default_value=seq_length_max)
                window_size = get_hyperparameter(window_size, Constant)
            else:
                window_size_value_range = window_size.value_range
                window_size = HyperparameterSearchSpace(hyperparameter='window_size',
                                                        value_range=(window_size_value_range[0], seq_length_max),
                                                        default_value=min(window_size.default_value, seq_length_max))
                window_size = get_hyperparameter(window_size, UniformIntegerHyperparameter)
        else:
            window_size = get_hyperparameter(window_size, UniformIntegerHyperparameter)

        backcast = get_hyperparameter(backcast, CategoricalHyperparameter)
        backcast_period = get_hyperparameter(backcast_period, UniformIntegerHyperparameter)

        cs.add_hyperparameters([window_size, backcast, backcast_period])

        window_size_cond = EqualsCondition(window_size, backcast, False)
        backcast_period_cond = EqualsCondition(backcast_period, backcast, True)
        cs.add_conditions([window_size_cond, backcast_period_cond])

        time_feature_transform = dataset_properties.get('time_feature_transform', [])
        if time_feature_transform:
            add_hyperparameter(cs, transform_time_features, CategoricalHyperparameter)

        return cs

    def __str__(self) -> str:
        """ Allow a nice understanding of what components where used """
        string = self.train_data_loader.__class__.__name__
        return string
