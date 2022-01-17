from typing import Any, Dict, Optional, Tuple, Union, Sequence, List
from functools import partial

from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformIntegerHyperparameter, CategoricalHyperparameter
from ConfigSpace.conditions import EqualsCondition

import numpy as np

import torch
import collections
from torch.utils.data.sampler import SubsetRandomSampler
from torch._six import string_classes
from torch.utils.data._utils.collate import np_str_obj_array_pattern, default_collate_err_msg_format, default_collate

import torchvision

from autoPyTorch.datasets.base_dataset import TransformSubset
from autoPyTorch.datasets.time_series_dataset import TimeSeriesForecastingDataset, TimeSeriesSequence
from autoPyTorch.utils.common import (
    HyperparameterSearchSpace,
    custom_collate_fn,
    add_hyperparameter,
    get_hyperparameter
)

from autoPyTorch.pipeline.components.training.data_loader.feature_data_loader import FeatureDataLoader


class TestSequenceDataset(TransformSubset):
    def __init__(self, dataset: List[TimeSeriesSequence], train: bool = False) -> None:
        self.dataset = dataset
        self.indices = torch.arange(len(dataset))
        self.train = train

    def __getitem__(self, idx: int) -> np.ndarray:
        # we only consider the entire sequence
        seq = self.dataset[idx]
        return seq.__getitem__(len(seq) - 1, self.train)


def pad_sequence_from_start(sequences: List[torch.Tensor],
                            seq_minimal_length: int,
                            seq_max_length: int = np.inf,
                            batch_first=True,
                            padding_value=0.0) -> torch.Tensor:
    r"""
    This function is quite similar to  torch.nn.utils.rnn.pad_sequence except that we pad new values from the start of
    the sequence. i.e., instead of extending [1,2,3] to [1,2,3,0,0], we extend it as [0,0,1,2,3]. Additionally, the
    generated sequnece needs to have a length of at least seq_minimal_length
    """

    # assuming trailing dimensions and type of all the Tensors
    # in sequences are same and fetching those from sequences[0]
    max_size = sequences[0].size()
    trailing_dims = max_size[1:]
    max_len = min(max(max([s.size(0) for s in sequences]), seq_minimal_length), seq_max_length)
    if seq_max_length > max_len:
        seq_max_length = max_len
    if batch_first:
        out_dims = (len(sequences), max_len) + trailing_dims
    else:
        out_dims = (max_len, len(sequences)) + trailing_dims

    out_tensor = sequences[0].new_full(out_dims, padding_value)
    for i, tensor in enumerate(sequences):
        length = min(tensor.size(0), seq_max_length)
        # use index notation to prevent duplicate references to the tensor
        if batch_first:
            out_tensor[i, -length:, ...] = tensor[-length:]
        else:
            out_tensor[-length:, i, ...] = tensor[-length:]

    return out_tensor


class PadSequenceCollector:
    """
    A collector that transform the sequences from dataset. Since the sequences might contain different
    length, they need to be padded with constant value. Given that target value might require special value to
    fit the requirement of distribution, past_target will be padded with special values

    """

    def __init__(self, window_size: int, target_padding_value: float = 0.0, seq_max_length: int = np.inf):
        self.window_size = window_size
        self.target_padding_value = target_padding_value
        self.seq_max_length = seq_max_length

    def __call__(self, batch, padding_value=0.0):
        elem = batch[0]
        elem_type = type(elem)
        if isinstance(elem, torch.Tensor):
            seq = pad_sequence_from_start(batch,
                                          seq_minimal_length=self.window_size,
                                          seq_max_length=self.seq_max_length,
                                          batch_first=True, padding_value=padding_value)  # type: torch.Tensor
            return seq

        elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
                and elem_type.__name__ != 'string_':
            if elem_type.__name__ == 'ndarray' or elem_type.__name__ == 'memmap':
                # array of string classes and object
                if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                    raise TypeError(default_collate_err_msg_format.format(elem.dtype))

                return default_collate([torch.as_tensor(b) for b in batch])
            elif elem.shape == ():  # scalars
                return torch.as_tensor(batch)
        elif isinstance(elem, float):
            return torch.tensor(batch, dtype=torch.float64)
        elif isinstance(elem, int):
            return torch.tensor(batch)
        elif isinstance(elem, string_classes):
            return batch
        elif isinstance(elem, collections.abc.Mapping):
            return {key: self([d[key] for d in batch]) if key != "past_target"
            else self([d[key] for d in batch], self.target_padding_value) for key in elem}
        raise TypeError(f"Unsupported data type {elem_type}")


class TimeSeriesSampler(SubsetRandomSampler):
    def __init__(self,
                 indices: Sequence[int],
                 seq_lengths: Sequence[int],
                 num_instances_per_seqs: Optional[List[float]] = None,
                 min_start: int = 0,
                 generator: Optional[torch.Generator] = None) -> None:
        """
        A sampler designed for time series sequence. For the sake of efficiency, it will not sample each possible
        sequences from indices. Instead, it samples 'num_instances_per_seqs' for each sequence. This sampler samples
        the instances in a Latin-Hypercube likewise way: we divide each sequence in to num_instances_per_seqs interval
        and  randomly sample one instance from each interval. If num_instances_per_seqs is not an integral, then the
        first interval is selected with a certain probability:
        for instance, if we want to sample 1.3 instance from a sequence [0,1,2,3,4,5], then we first divide the seuqence
        into two parts: [0, 3] and [3, 6], one sample is sampled from the second part, while an expected value of 0.3 is
        sampled from the first part (This part will be sampled in the very end with torch.multinomial)

        Parameters
        ----------
        indices: Sequence[int]
            The set of all the possible indices that can be sampled from
        seq_lengths: Sequence[int]
            lengths of each sequence, applied to unsqueeze indices
        num_instances_per_seqs: Optional[List[int]]=None
            expected number of instances to be sampled in each sequence, if it is None, all the sequences will be
            sampled
        min_start: int
            the how many first instances we want to skip (the first few sequences need to be padded with 0)
        generator: Optional[torch.Generator]
            pytorch generator to control the randomness
        """
        super(TimeSeriesSampler, self).__init__(indices, generator)
        if num_instances_per_seqs is None:
            self.iter_all_seqs = True
        else:
            self.iter_all_seqs = False
            if len(seq_lengths) != len(num_instances_per_seqs):
                raise ValueError(f'the lengths of seq_lengths must equal the lengths of num_instances_per_seqs.'
                                 f'However, they are {len(seq_lengths)} versus {len(num_instances_per_seqs)}')
            seq_intervals_int = []
            seq_intervals_decimal = []
            # seq_intervals_decimal_length = []
            num_expected_ins_decimal = []
            idx_tracker = 0
            for seq_idx, (num_instances, seq_length) in enumerate(zip(num_instances_per_seqs, seq_lengths)):
                idx_end = idx_tracker + seq_length
                idx_start = idx_tracker + min_start
                if idx_start > idx_end:
                    idx_start = idx_tracker

                num_interval = int(np.ceil(num_instances))
                if num_interval > idx_end - idx_start or num_interval == 0:
                    interval = np.linspace(idx_start, idx_end, 2, endpoint=True, dtype=np.int)
                    # we consider
                    num_expected_ins_decimal.append(num_instances)
                    seq_intervals_decimal.append(interval[:2])
                    seq_intervals_int.append(interval[1:])
                else:
                    interval = np.linspace(idx_start, idx_end, num_interval + 1, endpoint=True, dtype=np.int)

                    num_expected_ins_decimal.append(np.modf(num_instances)[0])
                    seq_intervals_decimal.append(interval[:2])

                    seq_intervals_int.append(interval[1:])
                idx_tracker += seq_length

            num_expected_ins_decimal = np.stack(num_expected_ins_decimal)
            # seq_intervals_decimal_length = np.stack(seq_intervals_decimal_length)
            self.seq_lengths = seq_lengths
            self.seq_lengths_sum = np.sum(seq_lengths)
            self.num_instances = int(np.round(np.sum(num_instances_per_seqs)))

            self.seq_intervals_decimal = torch.from_numpy(np.stack(seq_intervals_decimal))
            self.seq_intervals_int = seq_intervals_int

            self.num_expected_ins_decimal = torch.from_numpy(num_expected_ins_decimal) + 1e-8

    def __iter__(self):
        if self.iter_all_seqs:
            return super().__iter__()
        samples = torch.ones(self.num_instances, dtype=torch.int)
        idx_samples_start = 0
        idx_samples_end = 0
        for idx_seq, (interval, seq_length) in enumerate(zip(self.seq_intervals_int, self.seq_lengths)):
            if len(interval) == 1:
                continue
            num_samples = len(interval) - 1
            idx_samples_end = idx_samples_start + num_samples

            samples_shift = torch.rand(num_samples, generator=self.generator) * (interval[1:] - interval[:-1])
            samples_seq = torch.floor(samples_shift + interval[:-1]).int()
            samples[idx_samples_start: idx_samples_end] = samples_seq

            idx_samples_start = idx_samples_end
        num_samples_remain = self.num_instances - idx_samples_end
        if num_samples_remain > 0:
            if num_samples_remain > self.num_expected_ins_decimal[-1]:
                replacement = True
            else:
                replacement = False
            samples_idx = torch.multinomial(self.num_expected_ins_decimal, num_samples_remain, replacement)
            seq_interval = self.seq_intervals_decimal[samples_idx]

            samples_shift = torch.rand(num_samples_remain, generator=self.generator)
            samples_shift *= (seq_interval[:, 1] - seq_interval[:, 0])
            samples_seq_remain = torch.floor(samples_shift).int() + seq_interval[:, 0]
            samples[-num_samples_remain:] = samples_seq_remain

        # sometimes if self.seq_lengths_sum is too large, float might not be accurate enough
        samples = torch.where(samples == self.seq_lengths_sum, samples - 1, samples)

        yield from (samples[i] for i in torch.randperm(self.num_instances, generator=self.generator))

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

    def __init__(self, sample_interval: int = 1, ):
        """
        initialization
        Args:
            sample_interval: int: sample resolution
            window_size: int: the size of the sliding window
        """
        self.sample_interval = sample_interval
        # assuming that subseq_length is 10, e.g., we can only start from -10. sample_interval = -4
        # we will sample the following indices: [-9,-5,-1]
        # self.first_indices = -(self.sample_interval * ((subseq_length - 1) // self.sample_interval) + 1)

    def __call__(self, data: np.ndarray) -> np.ndarray:
        if self.sample_interval == 1:
            return data
        else:
            subseq_length = len(data)
            first_indices = -(self.sample_interval * ((subseq_length - 1) // self.sample_interval) + 1)
            sample_indices = np.arange(first_indices, 0, step=self.sample_interval)

            return data[sample_indices]


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

        # this value corresponds to budget type resolution
        sample_interval = X.get('sample_interval', 1)
        padding_value = X.get('required_padding_value', 0.0)

        if sample_interval > 1:
            # for lower resolution, window_size should be smaller
            self.window_size = (self.window_size - 1) // sample_interval + 1

        max_lagged_value = max(X['dataset_properties'].get('lagged_value', [np.inf]))
        max_lagged_value += self.window_size + self.n_prediction_steps

        self.padding_collector = PadSequenceCollector(self.window_size, padding_value, max_lagged_value)

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
        if X['dataset_properties']["is_small_preprocess"]:
            # This parameter indicates that the data has been pre-processed for speed
            # Overwrite the datamanager with the pre-processes data
            datamanager.replace_data(X['X_train'], X['X_test'] if 'X_test' in X else None)
            self.dataset_small_preprocess = True
        else:
            self.dataset_small_preprocess = False

        train_dataset, val_dataset = datamanager.get_dataset_for_training(split_id=X['split_id'])

        train_split, test_split = datamanager.splits[X['split_id']]

        num_instances_dataset = np.size(train_split)
        num_instances_train = self.num_batches_per_epoch * self.batch_size

        # get the length of each sequence of training data (after split)
        # as we already know that the elements in 'train_split' increases consecutively with a certain number of
        # discontinuity where a new sequence is sampled: [0, 1, 2 ,3, 7 ,8 ].
        #  A new sequence must start from the index 7. We could then split each unique values to represent the length
        # of each split

        # TODO consider min_starrt as a hp (multiple of self.n_prediction_steps?)
        min_start = self.n_prediction_steps

        dataset_seq_length_train_all = X['dataset_properties']['sequence_lengths_train']
        if np.sum(dataset_seq_length_train_all) == len(train_split):
            # this works if we want to fit the entire datasets
            seq_train_length = np.array(dataset_seq_length_train_all)
        else:
            _, seq_train_length = np.unique(train_split - np.arange(len(train_split)), return_counts=True)
        # create masks for masking
        seq_idx_inactivate = np.where(self.random_state.rand(seq_train_length.size) > fraction_seq)
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

        self.val_data_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=min(1000, len(val_dataset)),
            shuffle=False,
            num_workers=X.get('num_workers', 0),
            pin_memory=X.get('pin_memory', True),
            drop_last=X.get('drop_last', False),
            collate_fn=partial(custom_collate_fn, x_collector=self.padding_collector),
        )
        return self

    def transform(self, X: Dict) -> Dict:
        X.update({"window_size": self.window_size})
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
        if self.sample_interval > 1:
            candidate_transformations.append(SequenceBuilder(sample_interval=self.sample_interval))

        candidate_transformations.append(ExpandTransformTimeSeries())
        if "test" in mode or not X['dataset_properties']['is_small_preprocess']:
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
        # TODO more supported inputs
        if isinstance(X, (np.ndarray, torch.Tensor)):
            if isinstance(X, torch.Tensor):
                X = X.numpy()
            if X.ndim == 1:
                X = [X]
        if isinstance(X, TimeSeriesSequence):
            X.update_transform(self.test_transform, train=False)
            dataset = [X]
        elif isinstance(X, Sequence):
            dataset = []
            if isinstance(X[0], TimeSeriesSequence):
                for X_seq in X:
                    X_seq.update_transform(self.test_transform, train=False)
                    dataset.append(X_seq)
            else:
                if y is None:
                    for X_seq in X:
                        seq = TimeSeriesSequence(
                            X=X_seq, Y=y,
                            # This dataset is used for loading test data in a batched format
                            train_transforms=self.test_transform,
                            val_transforms=self.test_transform,
                            n_prediction_steps=0,
                        )
                        dataset.append(seq)
                else:
                    for X_seq, y_seq in zip(X, y):
                        seq = TimeSeriesSequence(
                            X=X_seq, Y=y_seq,
                            # This dataset is used for loading test data in a batched format
                            train_transforms=self.test_transform,
                            val_transforms=self.test_transform,
                            n_prediction_steps=0,
                        )
                        dataset.append(seq)
        else:
            raise NotImplementedError(f"Unsupported type of input X: {type(X)}")

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
                                                                  default_value=2)
                                        ) -> ConfigurationSpace:
        """
        hyperparameter search space for forecasting dataloader. Forecasting dataloader construct the window size in two
        ways: either window_size is directly assigned or it is computed by backcast_period * n_prediction_steps
        (introduced by nbeats:
        Oreshkin et al., N-BEATS: Neural basis expansion analysis for interpretable time series forecasting, ICLR 2020
        https://arxiv.org/abs/1905.10437)
        Currently back_cast_period is only activate when back_cast is activate
        TODO ablation study on whether this technique can be applied to other models
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

        Returns:
            cs: Configuration Space

        """
        cs = ConfigurationSpace()
        add_hyperparameter(cs, batch_size, UniformIntegerHyperparameter)
        add_hyperparameter(cs, num_batch_per_epoch, UniformIntegerHyperparameter)
        add_hyperparameter(cs, sample_strategy, CategoricalHyperparameter)

        window_size = get_hyperparameter(window_size, UniformIntegerHyperparameter)
        backcast = get_hyperparameter(backcast, CategoricalHyperparameter)
        backcast_period = get_hyperparameter(backcast_period, UniformIntegerHyperparameter)

        cs.add_hyperparameters([window_size, backcast, backcast_period])

        window_size_cond = EqualsCondition(window_size, backcast, False)
        backcast_period_cond = EqualsCondition(backcast_period, backcast, True)
        cs.add_conditions([window_size_cond, backcast_period_cond])

        return cs

    def __str__(self) -> str:
        """ Allow a nice understanding of what components where used """
        string = self.train_data_loader.__class__.__name__
        return string
