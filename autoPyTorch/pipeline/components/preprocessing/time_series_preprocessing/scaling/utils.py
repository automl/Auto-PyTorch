from typing import Any, Union, Tuple

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator


# Similar to / inspired by
# https://github.com/tslearn-team/tslearn/blob/a3cf3bf/tslearn/preprocessing/preprocessing.py
class TimeSeriesScaler(BaseEstimator):
    def __init__(self, mode: str,
                 dataset_is_small_preprocess: bool = False,
                 static_features: Tuple[Union[str, int]] = ()):
        self.mode = mode
        self.dataset_is_small_preprocess = dataset_is_small_preprocess
        self.static_features = static_features

    def fit(self, X: pd.DataFrame, y: Any = None) -> "TimeSeriesScaler":
        """
        The transformer is transformed on the fly (for each batch)
        """
        static_features = [static_fea for static_fea in self.static_features if static_fea in X.columns]
        self.static_features = static_features
        return self

    def transform(self, X: pd.DataFrame) -> Tuple[np.ndarray, ...]:
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
        if self.mode == "standard":
            if self.dataset_is_small_preprocess:
                X_grouped = X.groupby(X.index)

                self.loc = X_grouped.agg("mean")
                self.scale = X_grouped.agg("std")

                # for static features, if we do normalization w.r.t. each group, then they will become the same values,
                # thus we treat them differently: normalize with the entire dataset
                self.scale[self.static_features] = self.loc[self.static_features].std()
                self.loc[self.static_features] = self.loc[self.static_features].mean()
            else:
                self.loc = X.mean()
                self.scale = X.std()

            # ensure that if all the values are the same in a group, we could still normalize them correctly
            self.scale.mask(self.scale == 0.0, self.loc)
            self.scale[self.scale == 0] = 1.

            return (X - self.loc) / self.scale

        elif self.mode == "min_max":
            if self.dataset_is_small_preprocess:
                X_grouped = X.groupby(X.index)
                min_ = X_grouped.agg("min")
                max_ = X_grouped.agg("max")

                min_[self.static_features] = min_[self.static_features].min()
                max_[self.static_features] = max_[self.static_features].max()

            else:
                min_ = X.min()
                max_ = X.max()

            diff_ = max_ - min_
            self.loc = min_
            self.scale = diff_
            self.scale.mask(self.scale == 0.0, self.loc)
            self.scale[self.scale == 0.0] = 1.0
            return (X - self.loc) / self.scale

        elif self.mode == "max_abs":
            X_abs = X.transform("abs")
            if self.dataset_is_small_preprocess:
                max_abs_ = X_abs.groupby(X_abs.index).transform("max")
                max_abs_[self.static_features] = max_abs_[self.static_features].max()
            else:
                max_abs_ = X_abs.max()

            max_abs_[max_abs_ == 0.0] = 1.0
            self.loc = None
            self.scale = max_abs_

            return X / self.scale

        elif self.mode == 'mean_abs':
            X_abs = X.transform("abs")
            if self.dataset_is_small_preprocess:
                X_abs = X_abs.groupby(X_abs.index)
                mean_abs_ = X_abs.agg("mean")
                mean_abs_[self.static_features] = mean_abs_[self.static_features].mean()
            else:
                mean_abs_ = X_abs.mean()
            self.loc = None
            self.scale = mean_abs_.mask(mean_abs_ == 0.0, X_abs.agg("max"))

            return X / self.scale

        elif self.mode == "none":
            self.loc = None
            self.scale = None

            return X
        else:
            raise ValueError(f"Unknown mode {self.mode} for time series scaler")
