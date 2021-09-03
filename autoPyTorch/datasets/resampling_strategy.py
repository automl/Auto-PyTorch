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
class CROSS_VAL_FN(Protocol):
    def __call__(self,
                 num_splits: int,
                 indices: np.ndarray,
                 stratify: Optional[Any]) -> List[Tuple[np.ndarray, np.ndarray]]:
        ...


class HOLDOUT_FN(Protocol):
    def __call__(self, val_share: float, indices: np.ndarray, stratify: Optional[Any]
                 ) -> Tuple[np.ndarray, np.ndarray]:
        ...


class CrossValTypes(IntEnum):
    stratified_k_fold_cross_validation = 1
    k_fold_cross_validation = 2
    stratified_shuffle_split_cross_validation = 3
    shuffle_split_cross_validation = 4
    time_series_cross_validation = 5


class HoldoutValTypes(IntEnum):
    holdout_validation = 6
    stratified_holdout_validation = 7
    time_series_hold_out_validation = 8


RESAMPLING_STRATEGIES = [CrossValTypes, HoldoutValTypes]

DEFAULT_RESAMPLING_PARAMETERS = {
    HoldoutValTypes.holdout_validation: {
        'val_share': 0.33,
    },
    HoldoutValTypes.stratified_holdout_validation: {
        'val_share': 0.33,
    },
    HoldoutValTypes.time_series_hold_out_validation: {
    'val_share': 0.2
    },
    CrossValTypes.k_fold_cross_validation: {
        'num_splits': 3,
    },
    CrossValTypes.stratified_k_fold_cross_validation: {
        'num_splits': 3,
    },
    CrossValTypes.shuffle_split_cross_validation: {
        'num_splits': 3,
    },
    CrossValTypes.time_series_cross_validation: {
        'num_splits': 3,
    },
}  # type: Dict[Union[HoldoutValTypes, CrossValTypes], Dict[str, Any]]


def get_cross_validators(*cross_val_types: CrossValTypes) -> Dict[str, CROSS_VAL_FN]:
    cross_validators = {}  # type: Dict[str, CROSS_VAL_FN]
    for cross_val_type in cross_val_types:
        cross_val_fn = globals()[cross_val_type.name]
        cross_validators[cross_val_type.name] = cross_val_fn
    return cross_validators


def get_holdout_validators(*holdout_val_types: HoldoutValTypes) -> Dict[str, HOLDOUT_FN]:
    holdout_validators = {}  # type: Dict[str, HOLDOUT_FN]
    for holdout_val_type in holdout_val_types:
        holdout_val_fn = globals()[holdout_val_type.name]
        holdout_validators[holdout_val_type.name] = holdout_val_fn
    return holdout_validators


def is_stratified(val_type: Union[str, CrossValTypes, HoldoutValTypes]) -> bool:
    if isinstance(val_type, str):
        return val_type.lower().startswith("stratified")
    else:
        return val_type.name.lower().startswith("stratified")


def holdout_validation(val_share: float, indices: np.ndarray, **kwargs: Any) -> Tuple[np.ndarray, np.ndarray]:
    train, val = train_test_split(indices, test_size=val_share, shuffle=False)
    return train, val


def stratified_holdout_validation(val_share: float, indices: np.ndarray, **kwargs: Any) \
        -> Tuple[np.ndarray, np.ndarray]:
    train, val = train_test_split(indices, test_size=val_share, shuffle=False, stratify=kwargs["stratify"])
    return train, val


def shuffle_split_cross_validation(num_splits: int, indices: np.ndarray, **kwargs: Any) \
        -> List[Tuple[np.ndarray, np.ndarray]]:
    cv = ShuffleSplit(n_splits=num_splits)
    splits = list(cv.split(indices))
    return splits


def stratified_shuffle_split_cross_validation(num_splits: int, indices: np.ndarray, **kwargs: Any) \
        -> List[Tuple[np.ndarray, np.ndarray]]:
    cv = StratifiedShuffleSplit(n_splits=num_splits)
    splits = list(cv.split(indices, kwargs["stratify"]))
    return splits


def stratified_k_fold_cross_validation(num_splits: int, indices: np.ndarray, **kwargs: Any) \
        -> List[Tuple[np.ndarray, np.ndarray]]:
    cv = StratifiedKFold(n_splits=num_splits)
    splits = list(cv.split(indices, kwargs["stratify"]))
    return splits


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


# TODO DO we move these under autoPyTorch/datasets/time_series_dataset.py?
# TODO rewrite this part, as we only need holdout sets
def time_series_hold_out_validation(val_share: float, indices: np.ndarray, **kwargs: Any) \
        -> Tuple[np.ndarray, np.ndarray]:
    """
    Return holdout indices respecting hte temporal ordering of the data
    Args:
        val_share:
        indices: List of all possible indices
        **kwargs:

    Returns:
    """
    # TODO consider how we handle test size properly
    # Time Series prediction only requires on set of prediction for each
    # This implement needs to be combined with time series forecasting dataloader, where each time an entire time series
    # is used for prediction
    test_size = kwargs['n_prediction_steps']
    cv = TimeSeriesSplit(n_splits=2, test_size=1, gap=kwargs['n_prediction_steps'] - 1)
    train, val = list(cv.split(indices))[-1]
    return train, val


def time_series_cross_validation(num_splits: int, indices: np.ndarray, **kwargs: Any) \
        -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Returns train and validation indices respecting the temporal ordering of the data.
    Dummy example: [0, 1, 2, 3] with 3 folds yields
        [0] [1]
        [0, 1] [2]
        [0, 1, 2] [3]

    :param indices: array of indices to be split, seq_length
    :param num_splits: number of cross validation splits
    :return: list of tuples of training and validation indices
    """
    # TODO: we use gap=n_prediction_step here, we need to consider if we want to implement n_prediction_step here or
    # under DATALOADER!!!
    # TODO do we need cross valriadtion for time series datasets?
    test_size = kwargs['n_prediction_steps']
    cv = TimeSeriesSplit(n_splits=num_splits, test_size=1, gap=kwargs['n_prediction_steps'] - 1)
    splits = list(cv.split(indices))
    return splits
