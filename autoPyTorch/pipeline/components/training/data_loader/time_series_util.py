import collections
from typing import Iterator, List, Mapping, Optional, Sequence, Sized, Union

import numpy as np

import torch
from torch._six import string_classes
from torch.utils.data._utils.collate import default_collate, default_collate_err_msg_format, np_str_obj_array_pattern
from torch.utils.data.sampler import SequentialSampler, SubsetRandomSampler

from autoPyTorch.datasets.base_dataset import TransformSubset
from autoPyTorch.datasets.time_series_dataset import TimeSeriesSequence


class TestSequenceDataset(TransformSubset):
    def __init__(self, dataset: List[TimeSeriesSequence], train: bool = False) -> None:
        self.dataset = dataset
        self.indices = torch.arange(len(dataset))
        self.train = train

    def __getitem__(self, idx: int) -> np.ndarray:
        # we only consider the entire sequence
        seq = self.dataset[idx]
        return seq.__getitem__(len(seq) - 1, self.train)


def pad_sequence_with_minimal_length(sequences: List[torch.Tensor],
                                     seq_minimal_length: int = 1,
                                     seq_max_length: int = np.iinfo(np.int32).max,
                                     batch_first: bool = True,
                                     padding_value: float = 0.0) -> torch.Tensor:
    r"""
    This function is quite similar to  torch.nn.utils.rnn.pad_sequence except that we constraint the sequence to be
    at least seq_minimal_length and at most seq_max_length
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
    if sequences[0].dtype == torch.bool:
        out_tensor = sequences[0].new_full(out_dims, False)
    else:
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

    def __init__(self, window_size: int, sample_interval: int = 1, target_padding_value: float = 0.0,
                 seq_max_length: int = np.iinfo(np.int32).max):
        self.window_size = window_size
        self.sample_interval = sample_interval
        self.target_padding_value = target_padding_value
        self.seq_max_length = seq_max_length

    def __call__(self, batch: Sequence[torch.Tensor], sample_interval: int = 1,
                 seq_minimal_length: int = 1, padding_value: float = 0.0) -> Union[torch.Tensor, Mapping]:
        elem = batch[0]
        elem_type = type(elem)
        if isinstance(elem, torch.Tensor):
            seq: torch.Tensor = pad_sequence_with_minimal_length(batch,  # type: ignore[arg-type]
                                                                 seq_minimal_length=seq_minimal_length,
                                                                 seq_max_length=self.seq_max_length,
                                                                 batch_first=True, padding_value=padding_value)

            if sample_interval > 1:
                subseq_length = seq.shape[1]
                first_indices = -(sample_interval * ((subseq_length - 1) // sample_interval) + 1)
                sample_indices = torch.arange(first_indices, 0, step=sample_interval)
                return seq[:, sample_indices]
            else:
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
            # only past targets and features needs to be transformed
            return {
                key: self([d[key] for d in batch]) if "past" not in key else self(
                    [d[key] for d in batch],
                    self.sample_interval,
                    self.window_size,
                    self.target_padding_value if "targets" in key else 0.0
                ) for key
                in elem}

        elif elem is None:
            return None
        raise TypeError(f"Unsupported data type {elem_type}")


class TimeSeriesSampler(SubsetRandomSampler):
    """
    A sampler designed for time series sequence. For the sake of efficiency, it will not sample each possible
    sequences from indices. Instead, it samples 'num_instances_per_seqs' for each sequence. This sampler samples
    the instances in a Latin-Hypercube likewise way: we divide each sequence in to num_instances_per_seqs interval
    and  randomly sample one instance from each interval. If num_instances_per_seqs is not an integral, then the
    first interval is selected with a certain probability:
    for instance, if we want to sample 1.3 instance from a sequence [0,1,2,3,4,5], then we first divide the seuqence
    into two parts: [0, 3] and [3, 6], one sample is sampled from the second part, while an expected value of 0.3 is
    sampled from the first part (This part will be sampled in the very end with torch.multinomial)

    Attributes:
        indices (Sequence[int]):
            The set of all the possible indices that can be sampled from
        seq_lengths (Union[Sequence[int], np.ndarray]):
            lengths of each sequence, applied to unsqueeze indices
        num_instances_per_seqs (Optional[List[int]]):
            expected number of instances to be sampled in each sequence, if it is None, all the sequences will be
            sampled
        min_start (int):
            how many first time steps we want to skip (the first few sequences need to be padded with 0)
        generator (Optional[torch.Generator]):
            pytorch generator to control the randomness
    """
    def __init__(self,
                 indices: Sequence[int],
                 seq_lengths: Union[Sequence[int], np.ndarray],
                 num_instances_per_seqs: Optional[Union[List[float], np.ndarray]] = None,
                 min_start: int = 0,
                 generator: Optional[torch.Generator] = None) -> None:
        super().__init__(indices, generator)
        if num_instances_per_seqs is None:
            self.iter_all_seqs = True
        else:
            self.iter_all_seqs = False
            if len(seq_lengths) != len(num_instances_per_seqs):
                raise ValueError(f'the lengths of seq_lengths must equal the lengths of num_instances_per_seqs.'
                                 f'However, they are {len(seq_lengths)} versus {len(num_instances_per_seqs)}')
            seq_intervals_int = []
            seq_intervals_decimal = []

            num_expected_ins_decimal = []
            idx_tracker = 0
            for seq_idx, (num_instances, seq_length) in enumerate(zip(num_instances_per_seqs, seq_lengths)):
                idx_end = idx_tracker + seq_length
                idx_start = idx_tracker + min_start
                if idx_start > idx_end:
                    idx_start = idx_tracker

                num_interval = int(np.ceil(num_instances))
                if num_interval > idx_end - idx_start or num_interval == 0:
                    interval = np.linspace(idx_start, idx_end, 2, endpoint=True, dtype=np.intp)
                    # In this case, seq_intervals_decimal contains the entire interval of the sequence.
                    num_expected_ins_decimal.append(num_instances)
                    seq_intervals_decimal.append(interval[:2])
                    seq_intervals_int.append(interval[1:])
                else:
                    interval = np.linspace(idx_start, idx_end, num_interval + 1, endpoint=True, dtype=np.intp)
                    # The first two item determines the first sequence interval where most of the samples need to be
                    # padded, we then make it the interval for the expected decimal
                    num_expected_ins_decimal.append(np.modf(num_instances)[0])
                    seq_intervals_decimal.append(interval[:2])

                    seq_intervals_int.append(interval[1:])
                idx_tracker += seq_length

            num_expected_ins_decimal_stacked = np.stack(num_expected_ins_decimal)

            self.seq_lengths = seq_lengths
            self.seq_lengths_sum = np.sum(seq_lengths)
            self.num_instances = int(np.round(np.sum(num_instances_per_seqs)))

            self.seq_intervals_decimal = torch.from_numpy(np.stack(seq_intervals_decimal))
            self.seq_intervals_int = seq_intervals_int

            self.num_expected_ins_decimal = torch.from_numpy(num_expected_ins_decimal_stacked) + 1e-8

    def __iter__(self) -> Iterator[int]:
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
            if num_samples_remain > self.num_expected_ins_decimal.shape[-1]:
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

    def __len__(self) -> int:
        return self.num_instances


class SequentialSubSetSampler(SequentialSampler):
    """
    Sampler for validation set that allows to sample only a fraction of the datasetset. For those datasets that
    have a big amount of datapoints. This function helps to reduce the inference time during validation after each
    epoch


    Attributes:
        data_source (Dataset):
            dataset to sample from, it is composed of several TimeSeriesSequence. for each TimeSeriesSequence only 1
            sample is allowed
        num_samples (int):
            number of samples to be sampled from the dataset source
        generator (Optional[torch.Generator]):
            torch random generator
    """
    data_source: Sized

    def __init__(self, data_source: Sized, num_samples: int, generator: Optional[torch.Generator] = None) -> None:
        super(SequentialSubSetSampler, self).__init__(data_source)
        if num_samples > len(data_source):
            self.eval_all_sequences = True
            self.num_samples = len(data_source)
        else:
            self.eval_all_sequences = False
            self.num_samples = num_samples
        self.generator = generator

    def __iter__(self) -> Iterator[int]:
        if self.eval_all_sequences:
            yield from super(SequentialSubSetSampler, self).__iter__()
        else:
            yield from torch.randperm(len(self.data_source), generator=self.generator)[:self.num_samples]

    def __len__(self) -> int:
        return self.num_samples


class ExpandTransformTimeSeries(object):
    """Expand Dimensionality so tabular transformations see
       a 2d Array, unlike the ExpandTransform defined under tabular dataset, the dimension is expanded
       along the last axis
    """

    def __call__(self, data: np.ndarray) -> np.ndarray:
        if len(data.shape) <= 1:
            data = np.expand_dims(data, axis=-1)
        return data
