from typing import Any, List, Callable, Optional, Union, Tuple

import numpy as np
import pandas as pd
from pandas.core.groupby.generic import DataFrameGroupBy

import sklearn
from sklearn.base import BaseEstimator


# Similar to / inspired by
# https://github.com/tslearn-team/tslearn/blob/a3cf3bf/tslearn/preprocessing/preprocessing.py
class TimeSeriesScaler(BaseEstimator):
    def __init__(self, mode: str):
        self.mode = mode
        #self.loc = 0.  # type: Union[np.ndarray, float]
        #self.scale = 1.  # type: Union[np.ndarray, float]

    def fit(self, X: pd.DataFrame, y: Any = None) -> "TimeSeriesScaler":
        """
        The transformer is transformed on the fly (for each batch)
        """
        if self.mode == "standard":
            X_grouped = X.groupby(X.index)

            self.loc = X_grouped.agg("mean")
            self.scale = X_grouped.agg("std")
            # ensure that if all the values are the same in a group, we could still normalize them correctly
            self.scale.mask(self.scale == 0.0, self.loc)
            self.scale[self.scale == 0] = 1.

        elif self.mode == "min_max":
            X_grouped = X.groupby(X.index)

            min_ = X_grouped.agg("min")
            max_ = X_grouped.agg("max")

            diff_ = max_ - min_
            self.loc = min_
            self.scale = diff_
            self.scale.mask(self.scale == 0.0, self.loc)
            self.scale[self.scale == 0.0] = 1.0

        elif self.mode == "max_abs":
            X_abs = X.transform("abs")
            max_abs_ = X_abs.groupby(X_abs.index).transform("max")
            max_abs_[max_abs_ == 0.0] = 1.0
            self.loc = None
            self.scale = max_abs_

        elif self.mode == 'mean_abs':
            X_abs = X.transform("abs")
            X_abs = X_abs.groupby(X_abs.index)
            mean_abs_ = X_abs.agg("mean")
            self.loc = None
            self.scale = mean_abs_.mask(mean_abs_ == 0.0, X_abs.agg("max"))

        elif self.mode == "none":
            self.loc = None
            self.scale = None
        else:
            raise ValueError(f"Unknown mode {self.mode} for time series scaler")
        return self

    def transform(self, X: np.ndarray) -> Tuple[np.ndarray, ...]:
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

        if self.mode in {"standard", "min_max"}:
            return (X - self.loc) / self.scale
        elif self.mode in {"max_abs", "mean_abs"}:
            return X / self.scale
        else:
            return X


