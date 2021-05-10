from enum import IntEnum
from typing import List, NamedTuple, Optional, Tuple, Union

import numpy as np

from sklearn.model_selection import (
    KFold,
    ShuffleSplit,
    StratifiedKFold,
    StratifiedShuffleSplit,
    TimeSeriesSplit,
    train_test_split
)

from torch.utils.data import Dataset


class _ResamplingStrategyArgs(NamedTuple):
    val_share: float = 0.33
    num_splits: int = 5
    shuffle: bool = False
    stratify: bool = False


class HoldoutFuncs():
    @staticmethod
    def holdout_validation(
        indices: np.ndarray,
        random_state: Optional[np.random.RandomState] = None,
        val_share: Optional[float] = None,
        shuffle: bool = False,
        labels_to_stratify: Optional[Union[Tuple[np.ndarray, np.ndarray], Dataset]] = None
    ) -> List[Tuple[np.ndarray, np.ndarray]]:

        train, val = train_test_split(
            indices, test_size=val_share,
            shuffle=shuffle, random_state=random_state,
            stratify=labels_to_stratify
        )
        return [(train, val)]


class CrossValFuncs():
    # (shuffle, is_stratify) -> split_fn
    _args2split_fn = {
        (True, True): StratifiedShuffleSplit,
        (True, False): ShuffleSplit,
        (False, True): StratifiedKFold,
        (False, False): KFold,
    }

    @staticmethod
    def k_fold_cross_validation(
        indices: np.ndarray,
        random_state: Optional[np.random.RandomState] = None,
        num_splits: Optional[int] = None,
        shuffle: bool = False,
        labels_to_stratify: Optional[Union[Tuple[np.ndarray, np.ndarray], Dataset]] = None
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Returns:
            splits (List[Tuple[List, List]]): list of tuples of training and validation indices
        """

        split_fn = CrossValFuncs._args2split_fn[(shuffle, labels_to_stratify is not None)]
        cv = split_fn(n_splits=num_splits, random_state=random_state)
        splits = list(cv.split(indices))
        return splits

    @staticmethod
    def time_series(
        indices: np.ndarray,
        random_state: Optional[np.random.RandomState] = None,
        num_splits: Optional[int] = None,
        shuffle: bool = False,
        labels_to_stratify: Optional[Union[Tuple[np.ndarray, np.ndarray], Dataset]] = None
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Returns train and validation indices respecting the temporal ordering of the data.

        Examples:
            >>> indices = np.array([0, 1, 2, 3])
            >>> CrossValFuncs.time_series_cross_validation(3, indices)
                [([0], [1]),
                 ([0, 1], [2]),
                 ([0, 1, 2], [3])]

        """
        cv = TimeSeriesSplit(n_splits=num_splits)
        splits = list(cv.split(indices))
        return splits


class CrossValTypes(IntEnum):
    """The type of cross validation

    This class is used to specify the cross validation function
    and is not supposed to be instantiated.

    Examples: This class is supposed to be used as follows
    >>> cv_type = CrossValTypes.k_fold_cross_validation
    >>> print(cv_type.name)

    k_fold_cross_validation

    >>> for cross_val_type in CrossValTypes:
            print(cross_val_type.name, cross_val_type.value)

    k_fold_cross_validation 100
    time_series 101
    """
    k_fold_cross_validation = 100
    time_series = 101

    def __call__(
        self,
        indices: np.ndarray,
        random_state: Optional[np.random.RandomState] = None,
        num_splits: Optional[int] = None,
        shuffle: bool = False,
        labels_to_stratify: Optional[Union[Tuple[np.ndarray, np.ndarray], Dataset]] = None
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        This function allows to call and type-check the specified function.

        Args:
            random_state (np.random.RandomState): random number genetor for the reproducibility
            num_splits (int): The number of splits in cross validation
            indices (np.ndarray): The indices of data points in a dataset
            shuffle (bool): If shuffle the indices or not
            labels_to_stratify (Optional[Union[Tuple[np.ndarray, np.ndarray], Dataset]]):
                The labels of the corresponding data points. It is used for the stratification.

        Returns:
            splits (List[Tuple[np.ndarray, np.ndarray]]):
                splits[a split identifier][0: train, 1: val][a data point identifier]

        """

        default_num_splits = _ResamplingStrategyArgs().num_splits
        num_splits = num_splits if num_splits is not None else default_num_splits
        split_fn = getattr(CrossValFuncs, self.name)

        return split_fn(
            random_state=random_state if shuffle else None,
            num_splits=num_splits,
            indices=indices,
            shuffle=shuffle,
            labels_to_stratify=labels_to_stratify
        )


class HoldoutValTypes(IntEnum):
    """The type of holdout validation

    This class is used to specify the holdout validation function
    and is not supposed to be instantiated.

    Examples: This class is supposed to be used as follows
    >>> holdout_type = HoldoutValTypes.holdout_validation
    >>> print(holdout_type.name)

    holdout_validation

    >>> print(holdout_type.value)

    0

    >>> for holdout_type in HoldoutValTypes:
            print(holdout_type.name)

    holdout_validation

    Additionally, HoldoutValTypes.<function> can be called directly.
    """

    holdout_validation = 0

    def __call__(
        self,
        indices: np.ndarray,
        random_state: Optional[np.random.RandomState] = None,
        val_share: Optional[float] = None,
        shuffle: bool = False,
        labels_to_stratify: Optional[Union[Tuple[np.ndarray, np.ndarray], Dataset]] = None
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        This function allows to call and type-check the specified function.

        Args:
            random_state (np.random.RandomState): random number genetor for the reproducibility
            val_share (float): The ratio of validation dataset vs the given dataset
            indices (np.ndarray): The indices of data points in a dataset
            shuffle (bool): If shuffle the indices or not
            labels_to_stratify (Optional[Union[Tuple[np.ndarray, np.ndarray], Dataset]]):
                The labels of the corresponding data points. It is used for the stratification.

        Returns:
            splits (List[Tuple[np.ndarray, np.ndarray]]):
                splits[a split identifier][0: train, 1: val][a data point identifier]

        """

        default_val_share = _ResamplingStrategyArgs().val_share
        val_share = val_share if val_share is not None else default_val_share
        split_fn = getattr(HoldoutFuncs, self.name)

        return split_fn(
            random_state=random_state if shuffle else None,
            val_share=val_share,
            indices=indices,
            shuffle=shuffle,
            labels_to_stratify=labels_to_stratify
        )
