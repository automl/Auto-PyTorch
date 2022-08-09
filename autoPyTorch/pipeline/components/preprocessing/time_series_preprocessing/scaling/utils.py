from typing import Any, Tuple, Union

import numpy as np

import pandas as pd

from sklearn.base import BaseEstimator

from autoPyTorch.constants import VERY_SMALL_VALUE


# Similar to / inspired by
# https://github.com/tslearn-team/tslearn/blob/a3cf3bf/tslearn/preprocessing/preprocessing.py
class TimeSeriesScaler(BaseEstimator):
    def __init__(self, mode: str,
                 dataset_is_small_preprocess: bool = True,
                 static_features: Union[Tuple[Union[str, int], ...], Tuple[()]] = ()):
        self.mode = mode
        self.dataset_is_small_preprocess = dataset_is_small_preprocess
        self.static_features = static_features

    def fit(self, X: Union[pd.DataFrame, np.ndarray], y: Any = None) -> "TimeSeriesScaler":
        """
        The transformer is transformed on the fly (for each batch)
        """
        if self.dataset_is_small_preprocess:
            if not isinstance(X, pd.DataFrame):
                raise ValueError(f'Scaling that works on small_preprocess dataset must work with pd.DataFrame.'
                                 f'However, it gets {type(X)}')

            static_features = [static_fea for static_fea in self.static_features if static_fea in X.columns]
            self.static_features = static_features  # type: ignore[assignment]

            if self.mode == "standard":
                X_grouped = X.groupby(X.index)

                self.loc = X_grouped.agg("mean")
                self.scale = X_grouped.agg("std").fillna(0.0)

                # for static features, if we do normalization w.r.t. each group, then they will become the same values,
                # thus we treat them differently: normalize with the entire dataset
                self.scale[self.static_features] = X[self.static_features].std().fillna(0.0)
                self.loc[self.static_features] = X[self.static_features].mean()

                # ensure that if all the values are the same in a group, we could still normalize them correctly
                self.scale[self.scale < VERY_SMALL_VALUE] = 1.

            elif self.mode == "min_max":
                X_grouped = X.groupby(X.index)
                min_ = X_grouped.agg("min")
                max_ = X_grouped.agg("max")

                min_[self.static_features] = min_[self.static_features].min()
                max_[self.static_features] = max_[self.static_features].max()

                diff_ = max_ - min_
                self.loc = min_
                self.scale = diff_
                self.scale.mask(self.scale == 0.0, self.loc)
                self.scale[self.scale < VERY_SMALL_VALUE] = 1.0

            elif self.mode == "max_abs":
                X_abs = X.transform("abs")
                max_abs_ = X_abs.groupby(X_abs.index).agg("max")
                max_abs_[self.static_features] = max_abs_[self.static_features].max()

                max_abs_[max_abs_ < VERY_SMALL_VALUE] = 1.0
                self.loc = None
                self.scale = max_abs_

            elif self.mode == 'mean_abs':
                X_abs = X.transform("abs")
                X_abs = X_abs.groupby(X_abs.index)
                mean_abs_ = X_abs.agg("mean")
                mean_abs_[self.static_features] = mean_abs_[self.static_features].mean()
                self.scale = mean_abs_.mask(mean_abs_ == 0.0, X_abs.agg("max"))

                self.scale[self.scale < VERY_SMALL_VALUE] = 1
                self.loc = None

            elif self.mode == "none":
                self.loc = None
                self.scale = None

            else:
                raise ValueError(f"Unknown mode {self.mode} for time series scaler")
        else:
            static_features = [static_fea for static_fea in self.static_features if static_fea < X.shape[1]]
            self.static_features = static_features  # type: ignore[assignment]

        return self

    def transform(self, X: Union[pd.DataFrame, np.ndarray]) -> Union[pd.DataFrame, np.ndarray]:
        """
        X = sklearn.utils.check_array(
            X,
            force_all_finite=True,
            ensure_2d=False,
            allow_nd=True,
            accept_sparse=False,
            accept_large_sparse=False
        ) # type: np.ndarray
        """
        if not self.dataset_is_small_preprocess:
            if not isinstance(X, np.ndarray):
                raise ValueError(f'Scaling that works on none-small_preprocess dataset must work with np.ndarray.'
                                 f'However, it gets {type(X)}')
            if self.mode == 'standard':
                # in this case X is a np array
                loc = X.mean(axis=0, keepdims=True)
                scale = np.nan_to_num(X.std(axis=0, ddof=1, keepdims=True))
                scale = np.where(scale == 0, loc, scale)
                scale[scale < VERY_SMALL_VALUE] = 1.
                return (X - loc) / scale

            elif self.mode == 'min_max':
                min_ = X.min(axis=0, keepdims=True)
                max_ = X.max(axis=0, keepdims=True)

                diff_ = max_ - min_
                loc = min_
                scale = diff_
                scale = np.where(scale == 0., loc, scale)
                scale[scale < VERY_SMALL_VALUE] = 1.0
                return (X - loc) / scale

            elif self.mode == "max_abs":
                X_abs = np.abs(X)
                max_abs_ = X_abs.max(0, keepdims=True)
                max_abs_[max_abs_ < VERY_SMALL_VALUE] = 1.0
                scale = max_abs_
                return X / scale

            elif self.mode == 'mean_abs':
                X_abs = np.abs(X)
                mean_abs_ = X_abs.mean(0, keepdims=True)
                scale = np.where(mean_abs_ == 0.0, np.max(X_abs), mean_abs_)
                scale[scale < VERY_SMALL_VALUE] = 1
                return X / scale

            elif self.mode == "none":
                return X
            else:
                raise ValueError(f"Unknown mode {self.mode} for time series scaler")

        if self.mode == "standard":
            return (X - self.loc) / self.scale

        elif self.mode == "min_max":
            return (X - self.loc) / self.scale

        elif self.mode == "max_abs":
            return X / self.scale

        elif self.mode == 'mean_abs':
            return X / self.scale

        elif self.mode == "none":
            return X
        else:
            raise ValueError(f"Unknown mode {self.mode} for time series scaler")
