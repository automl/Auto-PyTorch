from enum import IntEnum
from typing import Any, Dict, List, Optional, Tuple, Union, Callable, NamedTuple

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
"""TODO: remove constantkeys"""
from autoPyTorch.utils.common import ConstantKeys, BaseNamedTuple


SplitFunc = Callable[[int, np.ndarray, Any], List[Tuple[np.ndarray, np.ndarray]]]


class CrossValParameters(BaseNamedTuple, NamedTuple):
    n_splits: int = 3
    indices: np.ndarray = None
    stratify: Optional[np.ndarray] = None
    random_state: Optional[int] = 42


class HoldOutParameters(BaseNamedTuple, NamedTuple):
    val_ratio: int = 0.33
    indices: np.ndarray = None
    stratify: Optional[np.ndarray] = None
    random_state: Optional[int] = 42


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


def not_implemented_stratify(stratify: np.ndarray)-> None:
    if stratify is None:
        raise ValueError("stratify (label data) required as input")


class CrossValFuncs():
    @staticmethod
    def shuffle_split_cross_validation(cv_params: Union[Dict[str, Any], CrossValParameters]) \
            -> List[Tuple[np.ndarray, np.ndarray]]:

        cv_params = CrossValParameters(**cv_params) if isinstance(cv_params, dict) else cv_params

        cv = ShuffleSplit(n_splits=cv_params.n_splits, random_state=cv_params.random_state)
        splits = list(cv.split(cv_params.indices))
        return splits

    @staticmethod
    def stratified_shuffle_split_cross_validation(cv_params: Union[Dict[str, Any], CrossValParameters]) \
            -> List[Tuple[np.ndarray, np.ndarray]]:
        
        cv_params = CrossValParameters(**cv_params) if isinstance(cv_params, dict) else cv_params
        not_implemented_stratify(cv_params.stratify)

        cv = StratifiedShuffleSplit(n_splits=cv_params.n_splits, random_state=cv_params.random_state)
        splits = list(cv.split(cv_params.indices, cv_params.stratify))
        return splits

    @staticmethod
    def stratified_k_fold_cross_validation(cv_params: Union[Dict[str, Any], CrossValParameters]) \
            -> List[Tuple[np.ndarray, np.ndarray]]:

        cv_params = CrossValParameters(**cv_params) if isinstance(cv_params, dict) else cv_params
        not_implemented_stratify(cv_params.stratify)

        cv = StratifiedKFold(n_splits=cv_params.n_splits, random_state=cv_params.random_state)
        splits = list(cv.split(cv_params.indices, cv_params.stratify))
        return splits

    @staticmethod
    def k_fold_cross_validation(cv_params: Union[Dict[str, Any], CrossValParameters]) \
            -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Standard k fold cross validation.

        :param indices: array of indices to be split
        :param n_splits: number of cross validation splits
        :return: list of tuples of training and validation indices
        """
        cv_params = CrossValParameters(**cv_params) if isinstance(cv_params, dict) else cv_params

        cv = KFold(n_splits=cv_params.n_splits, random_state=cv_params.random_state)
        splits = list(cv.split(cv_params.indices))
        return splits

    @staticmethod
    def time_series_cross_validation(cv_params: Union[Dict[str, Any], CrossValParameters]) \
            -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Returns train and validation indices respecting the temporal ordering of the data.
        Dummy example: [0, 1, 2, 3] with 3 folds yields
            [0] [1]
            [0, 1] [2]
            [0, 1, 2] [3]

        :param indices: array of indices to be split
        :param n_splits: number of cross validation splits
        :return: list of tuples of training and validation indices
        """
        cv_params = CrossValParameters(**cv_params) if isinstance(cv_params, dict) else cv_params
        
        cv = TimeSeriesSplit(n_splits=cv_params.n_splits)
        splits = list(cv.split(cv_params.indices))
        return splits

    @classmethod
    def get_cross_validators(cls, *cross_val_types: Tuple[CrossValTypes]) \
        -> Dict[str, SplitFunc]:

        cross_validators = {
            cross_val_type.name: getattr(cls, cross_val_type.name)
            for cross_val_type in cross_val_types
        }
        return cross_validators


class HoldOutFuncs():
    @staticmethod
    def holdout_validation(holdout_params: Union[Dict[str, Any], HoldOutParameters]) \
            -> List[Tuple[np.ndarray, np.ndarray]]:

        train, val = train_test_split(holdout_params.indices, test_size=holdout_params.val_ratio,
                                      shuffle=False, random_state=holdout_params.random_state)
        return train, val

    @staticmethod
    def stratified_holdout_validation(holdout_params: Union[Dict[str, Any], HoldOutParameters]) \
            -> List[Tuple[np.ndarray, np.ndarray]]:
        
        not_implemented_stratify(stratify)

        train, val = train_test_split(holdout_params.indices, test_size=holdout_params.val_ratio, shuffle=True,
                                      stratify=holdout_params.stratify, random_state=holdout_params.random_state)
        return train, val

    @classmethod
    def get_holdout_validators(cls, *holdout_val_types: Tuple[HoldoutValTypes])-> Dict[str, SplitFunc]:

        holdout_validators = {
            holdout_val_type.name: getattr(cls, holdout_val_type.name)
            for holdout_val_type in holdout_val_types
        }
        return holdout_validators

"""
TODO: remove both from all the files. 
Currently, used in the followings:
autoPyTorch/datasets/base_dataset.py
autoPyTorch/datasets/image_dataset.py
autoPyTorch/datasets/resampling_strategy.py
autoPyTorch/datasets/tabular_dataset.py
autoPyTorch/optimizer/smbo.py


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
"""