from enum import IntEnum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from sklearn.model_selection import (
    KFold,
    ShuffleSplit,
    StratifiedKFold,
    StratifiedShuffleSplit,
    TimeSeriesSplit,
    train_test_split
)

from typing_extensions import Protocol


# Use callback protocol as workaround, since callable with function fields count 'self' as argument
class CrossValFunc(Protocol):
    def __call__(self,
                 random_state: np.random.RandomState,
                 num_splits: int,
                 indices: np.ndarray,
                 stratify: Optional[Any]) -> List[Tuple[np.ndarray, np.ndarray]]:
        ...


class HoldOutFunc(Protocol):
    def __call__(self, random_state: np.random.RandomState, val_share: float,
                 indices: np.ndarray, stratify: Optional[Any]
                 ) -> Tuple[np.ndarray, np.ndarray]:
        ...


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

    stratified_k_fold_cross_validation 1
    k_fold_cross_validation 2
    stratified_shuffle_split_cross_validation 3
    shuffle_split_cross_validation 4
    time_series_cross_validation 5
    """
    stratified_k_fold_cross_validation = 1
    k_fold_cross_validation = 2
    stratified_shuffle_split_cross_validation = 3
    shuffle_split_cross_validation = 4
    time_series_cross_validation = 5

    def is_stratified(self) -> bool:
        stratified = [self.stratified_k_fold_cross_validation,
                      self.stratified_shuffle_split_cross_validation]
        return getattr(self, self.name) in stratified


class HoldoutValTypes(IntEnum):
    """TODO: change to enum using functools.partial"""
    """The type of hold out validation (refer to CrossValTypes' doc-string)"""
    holdout_validation = 6
    stratified_holdout_validation = 7

    def is_stratified(self) -> bool:
        stratified = [self.stratified_holdout_validation]
        return getattr(self, self.name) in stratified


# TODO: replace it with another way
RESAMPLING_STRATEGIES = [CrossValTypes, HoldoutValTypes]

DEFAULT_RESAMPLING_PARAMETERS: Dict[Union[HoldoutValTypes, CrossValTypes], Dict[str, Any]] = {
    HoldoutValTypes.holdout_validation: {
        'val_share': 0.33,
    },
    HoldoutValTypes.stratified_holdout_validation: {
        'val_share': 0.33,
    },
    CrossValTypes.k_fold_cross_validation: {
        'num_splits': 5,
    },
    CrossValTypes.stratified_k_fold_cross_validation: {
        'num_splits': 5,
    },
    CrossValTypes.shuffle_split_cross_validation: {
        'num_splits': 5,
    },
    CrossValTypes.time_series_cross_validation: {
        'num_splits': 5,
    },
}


class HoldOutFuncs():
    @staticmethod
    def holdout_validation(random_state: np.random.RandomState,
                           val_share: float,
                           indices: np.ndarray,
                           **kwargs: Any
                           ) -> Tuple[np.ndarray, np.ndarray]:
        shuffle = kwargs.get('shuffle', True)
        train, val = train_test_split(indices, test_size=val_share,
                                      shuffle=shuffle,
                                      random_state=random_state if shuffle else None,
                                      )
        return train, val

    @staticmethod
    def stratified_holdout_validation(random_state: np.random.RandomState,
                                      val_share: float,
                                      indices: np.ndarray,
                                      **kwargs: Any
                                      ) -> Tuple[np.ndarray, np.ndarray]:
        train, val = train_test_split(indices, test_size=val_share, shuffle=True, stratify=kwargs["stratify"],
                                      random_state=random_state)
        return train, val

    @classmethod
    def get_holdout_validators(cls, *holdout_val_types: HoldoutValTypes) -> Dict[str, HoldOutFunc]:

        holdout_validators = {
            holdout_val_type.name: getattr(cls, holdout_val_type.name)
            for holdout_val_type in holdout_val_types
        }
        return holdout_validators


class CrossValFuncs():
    @staticmethod
    def shuffle_split_cross_validation(random_state: np.random.RandomState,
                                       num_splits: int,
                                       indices: np.ndarray,
                                       **kwargs: Any
                                       ) -> List[Tuple[np.ndarray, np.ndarray]]:
        cv = ShuffleSplit(n_splits=num_splits, random_state=random_state)
        splits = list(cv.split(indices))
        return splits

    @staticmethod
    def stratified_shuffle_split_cross_validation(random_state: np.random.RandomState,
                                                  num_splits: int,
                                                  indices: np.ndarray,
                                                  **kwargs: Any
                                                  ) -> List[Tuple[np.ndarray, np.ndarray]]:
        cv = StratifiedShuffleSplit(n_splits=num_splits, random_state=random_state)
        splits = list(cv.split(indices, kwargs["stratify"]))
        return splits

    @staticmethod
    def stratified_k_fold_cross_validation(random_state: np.random.RandomState,
                                           num_splits: int,
                                           indices: np.ndarray,
                                           **kwargs: Any
                                           ) -> List[Tuple[np.ndarray, np.ndarray]]:

        shuffle = kwargs.get('shuffle', True)
        cv = StratifiedKFold(n_splits=num_splits, shuffle=shuffle,
                             random_state=random_state if not shuffle else None)
        splits = list(cv.split(indices, kwargs["stratify"]))
        return splits

    @staticmethod
    def k_fold_cross_validation(random_state: np.random.RandomState,
                                num_splits: int,
                                indices: np.ndarray,
                                **kwargs: Any
                                ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Standard k fold cross validation.

        Args:
            indices (np.ndarray): array of indices to be split
            num_splits (int): number of cross validation splits

        Returns:
            splits (List[Tuple[List, List]]): list of tuples of training and validation indices
        """
        shuffle = kwargs.get('shuffle', True)
        cv = KFold(n_splits=num_splits, random_state=random_state if shuffle else None, shuffle=shuffle)
        splits = list(cv.split(indices))
        return splits

    @staticmethod
    def time_series_cross_validation(random_state: np.random.RandomState,
                                     num_splits: int,
                                     indices: np.ndarray,
                                     **kwargs: Any
                                     ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Returns train and validation indices respecting the temporal ordering of the data.

        Args:
            indices (np.ndarray): array of indices to be split
            num_splits (int): number of cross validation splits

        Returns:
            splits (List[Tuple[List, List]]): list of tuples of training and validation indices

        Examples:
            >>> indices = np.array([0, 1, 2, 3])
            >>> CrossValFuncs.time_series_cross_validation(3, indices)
                [([0], [1]),
                 ([0, 1], [2]),
                 ([0, 1, 2], [3])]

        """
        cv = TimeSeriesSplit(n_splits=num_splits, random_state=random_state)
        splits = list(cv.split(indices))
        return splits

    @classmethod
    def get_cross_validators(cls, *cross_val_types: CrossValTypes) -> Dict[str, CrossValFunc]:
        cross_validators = {
            cross_val_type.name: getattr(cls, cross_val_type.name)
            for cross_val_type in cross_val_types
        }
        return cross_validators
