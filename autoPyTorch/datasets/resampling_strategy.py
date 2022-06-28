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
class NoResamplingFunc(Protocol):
    def __call__(self,
                 random_state: np.random.RandomState,
                 indices: np.ndarray) -> np.ndarray:
        ...


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


def holdout_split_forecasting(holdout: TimeSeriesSplit, indices: np.ndarray, n_prediction_steps: int,
                              n_repeats: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """
    A function that do holdout split without raising an error: When the target sequence is too short to be split into
    training and validation set, the training set will simply ignore that and we only consider the validation set.
    """
    try:
        train, val = list(holdout.split(indices))[-1]
        val = [val[-1 - i * n_prediction_steps] for i in reversed(range(n_repeats))]
    except (ValueError, IndexError):
        train = np.array([], dtype=indices.dtype)
        val = [-1]
    return indices[train], indices[val]


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
    time_series_ts_cross_validation = 6

    def is_stratified(self) -> bool:
        stratified = [self.stratified_k_fold_cross_validation,
                      self.stratified_shuffle_split_cross_validation]
        return getattr(self, self.name) in stratified


class HoldoutValTypes(IntEnum):
    """TODO: change to enum using functools.partial"""
    """The type of hold out validation (refer to CrossValTypes' doc-string)"""
    holdout_validation = 11
    stratified_holdout_validation = 12
    time_series_hold_out_validation = 13

    def is_stratified(self) -> bool:
        stratified = [self.stratified_holdout_validation]
        return getattr(self, self.name) in stratified


class NoResamplingStrategyTypes(IntEnum):
    no_resampling = 8

    def is_stratified(self) -> bool:
        return False


# TODO: replace it with another way
ResamplingStrategies = Union[CrossValTypes, HoldoutValTypes, NoResamplingStrategyTypes]

DEFAULT_RESAMPLING_PARAMETERS: Dict[
    ResamplingStrategies,
    Dict[str, Any]
] = {
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
        'num_splits': 5,
    },
    CrossValTypes.stratified_k_fold_cross_validation: {
        'num_splits': 5,
    },
    CrossValTypes.shuffle_split_cross_validation: {
        'num_splits': 5,
    },
    CrossValTypes.time_series_cross_validation: {
        'num_splits': 3,
    },
    CrossValTypes.time_series_ts_cross_validation: {
        'num_splits': 2
    },
    NoResamplingStrategyTypes.no_resampling: {}
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

    @staticmethod
    def time_series_hold_out_validation(random_state: np.random.RandomState,
                                        val_share: float, indices: np.ndarray, **kwargs: Any) \
            -> Tuple[np.ndarray, np.ndarray]:
        """
        Return holdout indices respecting hte temporal ordering of the data
        Args:
            val_share:
            indices: List of all possible indices
            **kwargs:

        Returns:
        """
        n_prediction_steps = kwargs['n_prediction_steps']
        n_repeats = kwargs['n_repeats']
        # Time Series prediction only requires on set of prediction for each
        # This implement needs to be combined with time series forecasting dataloader, where each time an entire
        # time series is used for prediction
        cv = TimeSeriesSplit(n_splits=2, test_size=1 + n_prediction_steps * (n_repeats - 1), gap=n_prediction_steps - 1)

        train, val = holdout_split_forecasting(holdout=cv,
                                               indices=indices,
                                               n_prediction_steps=n_prediction_steps,
                                               n_repeats=n_repeats)
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
            n_prediction_steps(int): forecsting horizon, to ensure that there is no overlapping between splits
            n_repeats (int): number of sequences inside each split, e.g., inside each split, we could ask the model to
                predict n_reapet times

        Returns:
            splits (List[Tuple[List, List]]): list of tuples of training and validation indices

        Examples:
            >>> indices = np.array([0, 1, 2, 3])
            >>> CrossValFuncs.time_series_cross_validation(3, indices)
                [([0], [1]),
                 ([0, 1], [2]),
                 ([0, 1, 2], [3])]

        """
        test_size = kwargs['n_prediction_steps']
        n_repeats = kwargs['n_repeats']
        cv = TimeSeriesSplit(n_splits=num_splits, test_size=test_size * n_repeats, gap=0)
        splits = [(
            indices[split[0]],
            indices[split[1][[-1 - n * test_size for n in reversed(range(n_repeats))]]]) for split in cv.split(indices)]
        return splits

    @staticmethod
    def time_series_ts_cross_validation(random_state: np.random.RandomState,
                                        num_splits: int,
                                        indices: np.ndarray,
                                        **kwargs: Any
                                        ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        A special sort of Time series cross validator: it could be considered as a mixture of two sorts of holdout set:
        The first holdout setting: trend setting, simply consider the tail of the sequence as validation sets and the
        part before as training set
        The second holdout setting: seasonality setting, ensures that the distance between validation sets and test sets
        is a multiple of seasonality period. We could thus ensure that validation and test sets are at the same
        position of the period

        Args:
            indices (np.ndarray): array of indices to be split
            num_splits (int): number of cross validation splits
            seasonality_h_value (int): distance between the start of the validation set and the test set, this value
                need to be (roughly) a multiple of freq_value

        Returns:
            splits (List[Tuple[List, List]]): list of tuples of training and validation indices
        """
        n_prediction_steps = kwargs['n_prediction_steps']
        seasonality_h_value = kwargs['seasonality_h_value']
        n_repeats = kwargs["n_repeats"]

        assert seasonality_h_value >= n_prediction_steps
        cv = TimeSeriesSplit(n_splits=2, test_size=n_prediction_steps * n_repeats, gap=0)

        train_t, val_t = holdout_split_forecasting(holdout=cv,
                                                   indices=indices,
                                                   n_prediction_steps=n_prediction_steps,
                                                   n_repeats=n_repeats)

        splits = [(train_t, val_t)]
        if num_splits > 1:
            cv = TimeSeriesSplit(n_splits=2, test_size=seasonality_h_value * n_repeats, gap=0)
            n_tail = - n_prediction_steps
            for i_split in range(1, num_splits):
                n_tail += seasonality_h_value
                if n_tail > len(indices):
                    # normally this should not happen as seasonality_h_value is carefully computed by ForecastingDataset
                    indices_split = indices
                else:
                    indices_split = indices[:-n_tail]
                train_s, val_s = holdout_split_forecasting(cv, indices_split,
                                                           n_prediction_steps=seasonality_h_value,
                                                           n_repeats=n_repeats)
                if len(train_s) > 0:
                    train_s = np.concatenate(
                        [train_s, np.arange(seasonality_h_value - n_prediction_steps) + train_s[-1] + 1]
                    )
                splits.append((train_s, val_s))
        return splits

    @classmethod
    def get_cross_validators(cls, *cross_val_types: CrossValTypes) -> Dict[str, CrossValFunc]:
        cross_validators = {
            cross_val_type.name: getattr(cls, cross_val_type.name)
            for cross_val_type in cross_val_types
        }
        return cross_validators


class NoResamplingFuncs():
    @classmethod
    def get_no_resampling_validators(cls, *no_resampling_types: NoResamplingStrategyTypes
                                     ) -> Dict[str, NoResamplingFunc]:
        no_resampling_strategies: Dict[str, NoResamplingFunc] = {
            no_resampling_type.name: getattr(cls, no_resampling_type.name)
            for no_resampling_type in no_resampling_types
        }
        return no_resampling_strategies

    @staticmethod
    def no_resampling(random_state: np.random.RandomState,
                      indices: np.ndarray) -> np.ndarray:
        """
        Returns the indices without performing
        any operation on them. To be used for
        fitting on the whole dataset.
        This strategy is not compatible with
        HPO search.
        Args:
            indices:  array of indices
        Returns:
            np.ndarray: array of indices
        """
        return indices
