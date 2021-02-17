"""The title of the module description
* Describe at the beginning of the source code.
* Describe before the package imports

TODO:
    * add doc-string for each class
"""

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

from autoPyTorch.utils.common import BaseNamedTuple


SplitFunc = Callable[[int, np.ndarray, Any], List[Tuple[np.ndarray, np.ndarray]]]


"""TODO: Change to BaseDict"""
class CrossValParameters(BaseNamedTuple, NamedTuple):
    """The parameters of cross validators

    Attributes:
        n_splits (int): The number of splits for cross validation
        random_state (int or None): The random seed
    """
    n_splits: int = 3
    random_state: Optional[int] = 42


class HoldOutParameters(BaseNamedTuple, NamedTuple):
    """The parameters of hold out validators

    Attributes:
        val_ratio (float): The ratio of validation size against training size
        random_state (int or None): The random seed
    """
    val_ratio: int = 0.33
    random_state: Optional[int] = 42


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


class HoldOutTypes(IntEnum):
    """The type of hold out validation (refer to CrossValTypes' doc-string)"""
    holdout_validation = 6
    stratified_holdout_validation = 7

    def is_stratified(self) -> bool:
        stratified = [self.stratified_holdout_validation]
        return getattr(self, self.name) in stratified


def not_implemented_stratify(stratify: np.ndarray) -> None:
    if stratify is None:
        raise ValueError("stratify (label data) required as input")


class CrossValFuncs():
    @staticmethod
    def input_warning(cv_params: CrossValParameters):
        if type(cv_params.n_splits) is not int:
            raise TypeError("n_splits for cross validation must be integer.")

    @staticmethod
    def shuffle_split_cross_validation(indices: np.ndarray, stratify: Optional[np.ndarray],
                                       cv_params: Union[Dict[str, Any], CrossValParameters]) \
            -> List[Tuple[np.ndarray, np.ndarray]]:

        cv_params = CrossValParameters(**cv_params) if isinstance(cv_params, dict) else cv_params
        CrossValFuncs.input_warning(cv_params)

        cv = ShuffleSplit(n_splits=cv_params.n_splits, random_state=cv_params.random_state)
        splits = list(cv.split(indices))
        return splits

    @staticmethod
    def stratified_shuffle_split_cross_validation(indices: np.ndarray, stratify: Optional[np.ndarray],
                                                  cv_params: Union[Dict[str, Any], CrossValParameters]) \
            -> List[Tuple[np.ndarray, np.ndarray]]:

        cv_params = CrossValParameters(**cv_params) if isinstance(cv_params, dict) else cv_params
        CrossValFuncs.input_warning(cv_params)
        not_implemented_stratify(stratify)

        cv = StratifiedShuffleSplit(n_splits=cv_params.n_splits, random_state=cv_params.random_state)
        splits = list(cv.split(indices, stratify))
        return splits

    @staticmethod
    def stratified_k_fold_cross_validation(indices: np.ndarray, stratify: Optional[np.ndarray],
                                           cv_params: Union[Dict[str, Any], CrossValParameters]) \
            -> List[Tuple[np.ndarray, np.ndarray]]:

        cv_params = CrossValParameters(**cv_params) if isinstance(cv_params, dict) else cv_params
        CrossValFuncs.input_warning(cv_params)
        not_implemented_stratify(stratify)

        cv = StratifiedKFold(n_splits=cv_params.n_splits, random_state=cv_params.random_state)
        splits = list(cv.split(indices, stratify))
        return splits

    @staticmethod
    def k_fold_cross_validation(indices: np.ndarray, stratify: Optional[np.ndarray],
                                cv_params: Union[Dict[str, Any], CrossValParameters]) \
            -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Standard k fold cross validation.

        :param indices: array of indices to be split
        :param n_splits: number of cross validation splits
        :return: list of tuples of training and validation indices
        """
        cv_params = CrossValParameters(**cv_params) if isinstance(cv_params, dict) else cv_params
        CrossValFuncs.input_warning(cv_params)

        cv = KFold(n_splits=cv_params.n_splits, random_state=cv_params.random_state)
        splits = list(cv.split(indices))
        return splits

    @staticmethod
    def time_series_cross_validation(indices: np.ndarray, stratify: Optional[np.ndarray],
                                     cv_params: Union[Dict[str, Any], CrossValParameters]) \
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
        CrossValFuncs.input_warning(cv_params)

        cv = TimeSeriesSplit(n_splits=cv_params.n_splits)
        splits = list(cv.split(indices))
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
    def input_warning(holdout_params: HoldOutParameters):
        if not 0 < holdout_params.val_ratio < 1:
            raise ValueError(f"val_ratio must be in (0, 1), but got {holdout_params.val_ratio}.")

    @staticmethod
    def holdout_validation(indices: np.ndarray, stratify: Optional[np.ndarray],
                           holdout_params: Union[Dict[str, Any], HoldOutParameters]) \
            -> List[Tuple[np.ndarray, np.ndarray]]:

        HoldOutFuncs.input_warning(holdout_params)
        train, val = train_test_split(indices, test_size=holdout_params.val_ratio,
                                      shuffle=False, random_state=holdout_params.random_state)
        return [(train, val)]

    @staticmethod
    def stratified_holdout_validation(indices: np.ndarray, stratify: Optional[np.ndarray],
                                      holdout_params: Union[Dict[str, Any], HoldOutParameters]) \
            -> List[Tuple[np.ndarray, np.ndarray]]:

        HoldOutFuncs.input_warning(holdout_params)
        not_implemented_stratify(stratify)

        train, val = train_test_split(indices, test_size=holdout_params.val_ratio, shuffle=True,
                                      stratify=stratify, random_state=holdout_params.random_state)
        return [(train, val)]

    @classmethod
    def get_holdout_validators(cls, *holdout_val_types: Tuple[HoldOutTypes]) -> Dict[str, SplitFunc]:

        holdout_validators = {
            holdout_val_type.name: getattr(cls, holdout_val_type.name)
            for holdout_val_type in holdout_val_types
        }
        return holdout_validators
