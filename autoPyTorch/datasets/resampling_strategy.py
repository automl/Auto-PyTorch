from enum import IntEnum
from typing import Any, Dict, List, Optional, Tuple, Union, Callable

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
from autoPyTorch.utils.common import ConstantKeys


NUM_SPLITS, VAL_SHARE = ConstantKeys.NUM_SPLITS, ConstantKeys.VAL_SHARE
STRATIFY, STRATIFIED = ConstantKeys.STRATIFY, ConstantKeys.STRATIFIED


class CrossValTypes(IntEnum):
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
    holdout_validation = 6
    stratified_holdout_validation = 7

    def is_stratified(self) -> bool:
        stratified = [self.stratified_holdout_validation]
        return getattr(self, self.name) in stratified


# Use callback protocol as workaround, since callable with function fields count 'self' as argument
class CrossValFuncs(Protocol):
    def __call__(self,
                 num_splits: int,
                 indices: np.ndarray,
                 stratify: Optional[Any]) -> List[Tuple[np.ndarray, np.ndarray]]:
        pass
    
    @staticmethod
    def shuffle_split_cross_validation(num_splits: int, indices: np.ndarray, **kwargs: Any) \
            -> List[Tuple[np.ndarray, np.ndarray]]:
        cv = ShuffleSplit(n_splits=num_splits)
        splits = list(cv.split(indices))
        return splits

    @staticmethod
    def stratified_shuffle_split_cross_validation(num_splits: int, indices: np.ndarray, **kwargs: Any) \
            -> List[Tuple[np.ndarray, np.ndarray]]:
        cv = StratifiedShuffleSplit(n_splits=num_splits)
        splits = list(cv.split(indices, kwargs[STRATIFY]))
        return splits

    @staticmethod
    def stratified_k_fold_cross_validation(num_splits: int, indices: np.ndarray, **kwargs: Any) \
            -> List[Tuple[np.ndarray, np.ndarray]]:
        cv = StratifiedKFold(n_splits=num_splits)
        splits = list(cv.split(indices, kwargs[STRATIFY]))
        return splits

    @staticmethod
    def k_fold_cross_validation(num_splits: int, indices: np.ndarray, **kwargs: Any) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Standard k fold cross validation.

        :param indices: array of indices to be split
        :param num_splits: number of cross validation splits
        :return: list of tuples of training and validation indices
        """
        cv = KFold(n_splits=num_splits)
        splits = list(cv.split(indices))
        return splits

    @staticmethod
    def time_series_cross_validation(num_splits: int, indices: np.ndarray, **kwargs: Any) \
            -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Returns train and validation indices respecting the temporal ordering of the data.
        Dummy example: [0, 1, 2, 3] with 3 folds yields
            [0] [1]
            [0, 1] [2]
            [0, 1, 2] [3]

        :param indices: array of indices to be split
        :param num_splits: number of cross validation splits
        :return: list of tuples of training and validation indices
        """
        cv = TimeSeriesSplit(n_splits=num_splits)
        splits = list(cv.split(indices))
        return splits

    @classmethod
    def get_cross_validators(cls, *cross_val_types: Tuple[CrossValTypes]) \
        -> Dict[str, Callable[[int, np.ndarray, Any], List[Tuple[np.ndarray, np.ndarray]]]]:
        cross_validators = {
            cross_val_type.name: getattr(cls, cross_val_type.name)
            for cross_val_type in cross_val_types
        }
        return cross_validators


class HoldOutFuncs(Protocol):
    def __call__(self,
                 val_share: float,
                 indices: np.ndarray,
                 stratify: Optional[Any]) -> Tuple[np.ndarray, np.ndarray]:
        pass

    @staticmethod
    def holdout_validation(val_share: float, indices: np.ndarray, **kwargs: Any) -> Tuple[np.ndarray, np.ndarray]:
        train, val = train_test_split(indices, test_size=val_share, shuffle=False)
        return train, val

    @staticmethod
    def stratified_holdout_validation(val_share: float, indices: np.ndarray, **kwargs: Any) \
            -> Tuple[np.ndarray, np.ndarray]:
        train, val = train_test_split(indices, test_size=val_share, shuffle=False, stratify=kwargs[STRATIFY])
        return train, val
    
    @classmethod
    def get_holdout_validators(cls, *holdout_val_types: Tuple[HoldoutValTypes]) \
        -> Dict[str, Callable[[float, np.ndarray, Any], List[Tuple[np.ndarray, np.ndarray]]]]:
        holdout_validators = {
            holdout_val_type.name: getattr(cls, holdout_val_type.name)
            for holdout_val_type in holdout_val_types
        }
        return holdout_validators


RESAMPLING_STRATEGIES = [CrossValTypes, HoldoutValTypes]

DEFAULT_RESAMPLING_PARAMETERS = {
    HoldoutValTypes.holdout_validation: {
        VAL_SHARE: 0.33,
    },
    HoldoutValTypes.stratified_holdout_validation: {
        VAL_SHARE: 0.33,
    },
    CrossValTypes.k_fold_cross_validation: {
        NUM_SPLITS: 3,
    },
    CrossValTypes.stratified_k_fold_cross_validation: {
        NUM_SPLITS: 3,
    },
    CrossValTypes.shuffle_split_cross_validation: {
        NUM_SPLITS: 3,
    },
    CrossValTypes.time_series_cross_validation: {
        NUM_SPLITS: 3,
    },
}  # type: Dict[Union[HoldoutValTypes, CrossValTypes], Dict[str, Any]]
