from typing import Any, List, Callable, Optional, Union, Tuple

import numpy as np

import sklearn
from sklearn.base import BaseEstimator


# Similar to / inspired by
# https://github.com/tslearn-team/tslearn/blob/a3cf3bf/tslearn/preprocessing/preprocessing.py
class TimeSeriesScaler(BaseEstimator):
    def __init__(self, mode: str):
        self.mode = mode
        #self.loc = 0.  # type: Union[np.ndarray, float]
        #self.scale = 1.  # type: Union[np.ndarray, float]

    def fit(self, X: np.ndarray, y: Any = None) -> "TimeSeriesScaler":
        """
        The transformer is transformed on the fly (for each batch)
        """
        # we assuem that the last two
        if self.mode == "standard":
            self.loc = np.mean(X, axis=-2, keepdims=True)
            self.scale = np.std(X, axis=-2, keepdims=True)

        elif self.mode == "min_max":
            min_ = np.min(X, axis=-2, keepdims=True)
            max_ = np.max(X, axis=-2, keepdims=True)

            diff_ = max_ - min_
            self.loc = min_
            self.scale = diff_

        elif self.mode == "max_abs":
            max_abs_ = np.max(np.abs(X), axis=-2, keepdims=True)
            max_abs_[max_abs_ == 0.0] = 1.0
            self.loc = np.zeros_like(max_abs_)
            self.scale = max_abs_

        elif self.mode == "none":
            self.loc = np.zeros([*X.shape[:-2], 1, X.shape[-1]])
            self.scale = np.ones([*X.shape[:-2], 1, X.shape[-1]])
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

        return (X - self.loc) / self.scale


