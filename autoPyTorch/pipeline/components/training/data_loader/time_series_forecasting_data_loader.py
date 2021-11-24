from typing import Any, Dict, Optional, Tuple, Union, Sequence, List

from torch.utils.data.sampler import SubsetRandomSampler

from autoPyTorch.pipeline.components.training.data_loader.time_series_data_loader import TimeSeriesDataLoader


from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import (
    UniformIntegerHyperparameter, Constant
)

import numpy as np

import torch

import torchvision

import warnings


from autoPyTorch.datasets.base_dataset import TransformSubset
from autoPyTorch.datasets.time_series_dataset import TimeSeriesForecastingDataset, TimeSeriesSequence
from autoPyTorch.utils.common import custom_collate_fn
from autoPyTorch.pipeline.components.training.data_loader.feature_data_loader import FeatureDataLoader
from autoPyTorch.pipeline.components.preprocessing.time_series_preprocessing.TimeSeriesTransformer import \
    TimeSeriesTransformer


class TimeSeriesSampler(SubsetRandomSampler):
    def __init__(self,
                 indices: Sequence[int],
                 seq_lengths: Sequence[int],
                 num_instances_per_seqs: List[int],
                 min_start: int = 0,
                 generator: Optional[torch.Generator] = None) -> None:
        """
        A sampler designed for time series sequence. For the sake of efficiency, it will not sample each possible
        sequences from indices. Instead, it samples 'num_instances_per_seqs' for each sequence. This sampler samples
        the instances in a Latin-Hypercube likewise way: we divide each sequence in to num_instances_per_seqs interval
        and  randomly sample one instance from each interval.

        Parameters
        ----------
        indices: Sequence[int]
            The set of all the possible indices that can be sampled from
        seq_lengths: Sequence[int]
            lengths of each sequence, applied to unsqueeze indices
        num_instances_per_seqs: List[int]
            how many instances are sampled in each sequence
        min_start: int
            the how many first instances we want to skip (the first few sequences need to be padded with 0)
        generator: Optional[torch.Generator]
            pytorch generator to control the randomness
        """
        super(TimeSeriesSampler, self).__init__(indices, generator)
        if len(seq_lengths) != len(num_instances_per_seqs):
            raise ValueError(f'the lengths of seq_lengths must equal the lengths of num_instances_per_seqs.'
                             f'However, they are {len(seq_lengths)} versus {len(num_instances_per_seqs)}')
        if np.sum(seq_lengths) != len(indices):
            raise ValueError(f'the sum of sequence length must correspond to the number of indices. '
                             f'However, they are {np.sum(seq_lengths)} versus {len(indices)}')
        seq_intervals = []
        idx_tracker = 0
        for seq_idx, (num_instances, seq_length) in enumerate(zip(num_instances_per_seqs, seq_lengths)):
            idx_end = idx_tracker + seq_length
            idx_start = idx_tracker + min_start
            interval = np.linspace(idx_start, idx_end, num_instances + 1, endpoint=True, dtype=np.int)
            seq_intervals.append(interval)
        self.seq_lengths = seq_lengths
        self.num_instances = np.sum(num_instances_per_seqs)
        self.seq_intervals = seq_intervals

    def __iter__(self):
        samples = torch.ones(self.num_instances, dtype=torch.int)
        idx_samples_start = 0
        idx_seq_tracker = 0
        for idx_seq, (interval, seq_length) in enumerate(zip(self.seq_intervals, self.seq_lengths)):
            num_samples = len(interval) - 1
            idx_samples_end = idx_samples_start + num_samples

            samples_shift = torch.rand(num_samples, generator=self.generator) * (interval[1:] - interval[:-1])
            samples_seq = torch.floor(samples_shift + interval[:-1]).int() + idx_seq_tracker
            samples[idx_samples_start: idx_samples_end] = samples_seq

            idx_samples_start = idx_samples_end
            idx_seq_tracker += seq_length

        return (samples[i] for i in torch.randperm(self.num_instances, generator=self.generator))

    def __len__(self):
        return self.num_instances


class ExpandTransformTimeSeries(object):
    """Expand Dimensionality so tabular transformations see
       a 2d Array, unlike the ExpandTransform defined under tabular dataset, the dimension is expanded
       along the last axis
    """
    def __call__(self, data: np.ndarray) -> np.ndarray:
        if len(data.shape) <= 1:
            data = np.expand_dims(data, axis=-1)
        return data


class SequenceBuilder(object):
    """build a time sequence token from the given time sequence
    it requires two hyperparameters: sample_interval and window size
    let's assume we have a time sequence
    x = [0 1 2 3 4 5 6 7 8 9 10].with window_size=3 and sample resolution=2
    then the extracted time series is [6, 8, 10] (or x[-5,-3,-1])
    if window_size=3 and sample_resolution=3
    then the extracted token is [4, 7, 10] (or x[-7,-4,-1])

    Parameters
    ----------
    sample_interval : int, default=1
        sample resolution

    window_size : int, default=1
        sliding window size
    """
    def __init__(self, sample_interval: int = 1, window_size: int = 1, subseq_length=1, padding_value=0.):
        """
        initialization
        Args:
            sample_interval: int: sample resolution
            window_size: int: the size of the sliding window
        """
        self.sample_interval = sample_interval
        self.window_size = window_size
        # assuming that subseq_length is 10, e.g., we can only start from -10. sample_interval = -4
        # we will sample the following indices: [-9,-5,-1]
        self.first_indices = -(self.sample_interval * ((subseq_length - 1) // self.sample_interval) + 1)
        self.padding_value = padding_value

    def __call__(self, data: np.ndarray) -> np.ndarray:
        sample_indices = np.arange(self.first_indices, 0, step=self.sample_interval)
        if sample_indices[0] < -1 * len(data):
            # we need to pad with 0
            valid_indices = sample_indices[np.where(sample_indices >= -len(data))[0]]

            data_values = data[valid_indices]
            if data.ndim == 1:
                padding_vector = np.full([len(sample_indices) - len(valid_indices)], self.padding_value)
                return np.hstack([padding_vector, data_values])
            else:
                padding_vector = np.full([len(sample_indices) - len(valid_indices), data.shape[-1]], self.padding_value)
                return np.vstack([padding_vector, data_values])
        else:
            return data[sample_indices]


class TimeSeriesForecastingDataLoader(FeatureDataLoader):
    """This class is an interface to read time sequence data

    It gives the possibility to read various types of mapped
    datasets as described in:
    https://pytorch.org/docs/stable/data.html
    """
    def __init__(self,
                 batch_size: int = 64,
                 window_size: int = 1,
                 #sample_interval: int = 1,
                 upper_sequence_length: int = np.iinfo(np.int32).max,
                 num_batches_per_epoch: Optional[int] = 50,
                 n_prediction_steps: int = 1) -> None:
        """
        initialize a dataloader
        Args:
            batch_size: batch size
            sequence_length: length of each sequence
            sample_interval: sample interval ,its value is the interval of the resolution
            upper_sequence_length: upper limit of sequence length, to avoid a sequence length larger than dataset length
            or specified by the users
            num_batches_per_epoch: how
            n_prediction_steps: how many stpes to predict in advance
        """
        super().__init__(batch_size=batch_size)
        self.window_size: int = window_size
        self.upper_sequence_length = upper_sequence_length
        self.n_prediction_steps = n_prediction_steps
        self.sample_interval = 1
        # length of the tail, for instance if a sequence_length = 2, sample_interval =2, n_prediction = 2,
        # the time sequence should look like: [X, y, X, y, y] [test_data](values in tail is marked with X)
        #self.subseq_length = self.sample_interval * (self.window_size - 1) + 1
        self.subseq_length = self.window_size
        self.num_batches_per_epoch = num_batches_per_epoch if num_batches_per_epoch is not None else np.inf

    def fit(self, X: Dict[str, Any], y: Any = None) -> torch.utils.data.DataLoader:
        """
        Fits a component by using an input dictionary with pre-requisites

        Args:
            X (X: Dict[str, Any]): Dependencies needed by current component to perform fit
            y (Any): not used. To comply with sklearn API

        Returns:
            A instance of self
        """
        sample_interval = X.get('sample_interval', 1)
        self.sample_interval = sample_interval

        # self.subseq_length = self.sample_interval * (self.window_size - 1) + 1
        # we want models with different sample_interval to have similar length scale
        self.subseq_length = self.window_size

        # Make sure there is an optimizer
        self.check_requirements(X, y)

        # Incorporate the transform to the dataset
        datamanager = X['backend'].load_datamanager()  # type: TimeSeriesForcecastingDataset
        assert self.subseq_length < datamanager.seq_length_min, "dataloader's window size must be smaller than the" \
                                                                "minimal sequence length of the dataset!!"
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
        if X['dataset_properties']["is_small_preprocess"]:
            # This parameter indicates that the data has been pre-processed for speed
            # Overwrite the datamanager with the pre-processes data
            datamanager.replace_data(X['X_train'], X['X_test'] if 'X_test' in X else None)
            self.dataset_small_preprocess = True
            self.preprocess_transforms_test = X['preprocess_transforms']
        else:
            self.dataset_small_preprocess = False

        self.n_prediction_steps = datamanager.n_prediction_steps
        train_dataset, val_dataset = datamanager.get_dataset_for_training(split_id=X['split_id'])

        train_split, test_split = datamanager.splits[X['split_id']]

        num_instances_dataset = np.size(train_split)
        num_instances_train = self.num_batches_per_epoch * self.batch_size

        # get the length of each sequence of training data (after split)
        # as we already know that the elements in 'train_split' increases consecutively with a certain number of
        # discontinuity where a new sequence is sampled: [0, 1, 2 ,3, 7 ,8 ].
        #  A new sequence must start from the index 7. We could then split each unique values to represent the length
        # of each split
        _, seq_train_length = np.unique(train_split - np.arange(len(train_split)), return_counts=True)
        num_instances_per_seqs = np.ceil(num_instances_train / num_instances_dataset * seq_train_length)
        num_instances_per_seqs = num_instances_per_seqs.astype(seq_train_length.dtype)
        # at least one element of each sequence should be selected

        # TODO consider the case where num_instances_train is greater than num_instances_dataset,
        # In which case we simply iterate through all the datasets

        """
        # to allow a time sequence data with resolution self.sample_interval and windows size with self.window_size
        # we need to drop the first part of each sequence
        for seq_idx, seq_length in enumerate(datamanager.sequence_lengths_train):
            idx_end = idx_start + seq_length
            #full_sequence = np.random.choice(np.arange(idx_start, idx_end)[self.subseq_length:], 5)
            #full_sequence = np.arange(idx_start, idx_end)[self.subseq_length:]
            #full_sequence = np.random.choice(np.arange(idx_start, idx_end)[self.subseq_length:], 5)
            full_sequence = np.arange(idx_start, idx_end)
            valid_indices.append(full_sequence)
            idx_start = idx_end

        valid_indices = np.hstack([valid_idx for valid_idx in valid_indices])
        _, sampler_indices_train, _ = np.intersect1d(train_split, valid_indices, return_indices=True)
        """
        # test_indices not required as testsets usually lies on the trail of hte sequence
        #_, sampler_indices_test, _ = np.intersect1d(test_split, valid_indices)

        sampler_indices_train = np.arange(num_instances_dataset)

        self.sampler_train = TimeSeriesSampler(indices=sampler_indices_train, seq_lengths=seq_train_length,
                                               num_instances_per_seqs=num_instances_per_seqs)

        self.train_data_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=min(self.batch_size, len(sampler_indices_train)),
            shuffle=False,
            num_workers=X.get('num_workers', 0),
            pin_memory=X.get('pin_memory', True),
            drop_last=X.get('drop_last', True),
            collate_fn=custom_collate_fn,
            sampler=self.sampler_train,
        )

        self.val_data_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=min(self.batch_size, len(val_dataset)),
            shuffle=False,
            num_workers=X.get('num_workers', 0),
            pin_memory=X.get('pin_memory', True),
            drop_last=X.get('drop_last', False),
            collate_fn=custom_collate_fn,
        )
        return self

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

        #if 'test' in mode or not X['dataset_properties']['is_small_preprocess']:
        #    candidate_transformations.extend(X['preprocess_transforms'])

        candidate_transformations.append((SequenceBuilder(sample_interval=self.sample_interval,
                                                          window_size=self.window_size,
                                                          subseq_length=self.subseq_length)))
        candidate_transformations.append((ExpandTransformTimeSeries()))

        # Transform to tensor
        candidate_transformations.append(torch.from_numpy)

        return torchvision.transforms.Compose(candidate_transformations)

    def get_loader(self, X: Union[np.ndarray, TimeSeriesSequence], y: Optional[np.ndarray] = None, batch_size: int = np.inf,
                   ) -> torch.utils.data.DataLoader:
        """
        Creates a data loader object from the provided data,
        applying the transformations meant to validation objects
        This is a lazy loaded test set, each time only one piece of series
        """
        # TODO any better way to deal with prediction data loader for multiple sequences
        if isinstance(X, np.ndarray):
            X = X[-self.subseq_length - self.n_prediction_steps + 1:]

            if self.dataset_small_preprocess:
                for preprocess in self.preprocess_transforms_test:
                    if isinstance(preprocess, TimeSeriesTransformer):
                        if preprocess.is_training:
                            preprocess.eval()

                transform = torchvision.transforms.Compose(self.preprocess_transforms_test)
                X = transform(X)

            if y is not None:
                # we want to make sure that X, and y can be mapped one to one (as sampling y requires a shifted value)
                y = y[-self.subseq_length - self.n_prediction_steps + 1:]

            dataset = TimeSeriesSequence(
                X=X, Y=y,
                # This dataset is used for loading test data in a batched format
                train_transforms=self.test_transform,
                val_transforms=self.test_transform,
                n_prediction_steps=0,
            )

        elif isinstance(X, TimeSeriesSequence):
            dataset = X
            dataset.update_transform(self.test_transform, train=False)
        else:
            raise ValueError(f"Unsupported type of input X: {type(X)}")

        # we only consider the last sequence as validation set
        test_seq_indices = [len(dataset) - 1]

        dataset_test = TransformSubset(dataset, indices=test_seq_indices, train=False)

        return torch.utils.data.DataLoader(
            dataset_test,
            batch_size=min(batch_size, len(dataset)),
            shuffle=False,
            collate_fn=custom_collate_fn,
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
                                        batch_size: Tuple[Tuple, int] = ((32, 320), 64),
                                        window_size: Tuple[Tuple, int] = ((20, 50), 25)
                                        ) -> ConfigurationSpace:
        batch_size = UniformIntegerHyperparameter(
            "batch_size", batch_size[0][0], batch_size[0][1], default_value=batch_size[1])
        if "upper_window_size" not in dataset_properties:
            warnings.warn('max_sequence_length is not given in dataset property , might exists the risk of selecting '
                          'length that is greater than the maximal allowed length of the dataset')
            upper_window_size = min(np.iinfo(np.int32).max, window_size[0][1])
        else:
            upper_window_size = min(dataset_properties["upper_window_size"], window_size[0][1])
        if window_size[0][0] >= upper_window_size:
            if upper_window_size == 1:
                warnings.warn("window size is fixed as 1")
                window_size = Constant("window_size", value=1)
            else:
                warnings.warn("the lower bound of window size is greater than the upper bound")
                window_size = UniformIntegerHyperparameter("window_size",
                                                           lower=1,
                                                           upper=upper_window_size,
                                                           default_value=upper_window_size)
        elif window_size[0][0] <= upper_window_size < window_size[0][1]:
            window_size = UniformIntegerHyperparameter("window_size",
                                                       lower=window_size[0][0],
                                                       upper=upper_window_size,
                                                       default_value=upper_window_size)
        else:
            window_size = UniformIntegerHyperparameter("window_size",
                                                       lower=window_size[0][0],
                                                       upper=window_size[0][1],
                                                       default_value=window_size[1])
        cs = ConfigurationSpace()
        cs.add_hyperparameters([batch_size, window_size])
        return cs

    def __str__(self) -> str:
        """ Allow a nice understanding of what components where used """
        string = self.train_data_loader.__class__.__name__
        return string
