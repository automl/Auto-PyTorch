from typing import Any

import numpy as np

import sklearn
from sklearn.base import BaseEstimator

import gluonts

# Similar to / inspired by
# https://github.com/tslearn-team/tslearn/blob/a3cf3bf/tslearn/preprocessing/preprocessing.py
class TimeSeriesScaler(BaseEstimator):
    def __init__(self, mode: str):
        self.mode = mode

    def fit(self, X: np.ndarray, y: Any = None) -> "TimeSeriesScaler":
        """
        For time series we do not need to fit anything since each time series is scaled individually
        """
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = sklearn.utils.check_array(
            X,
            force_all_finite=True,
            ensure_2d=False,
            allow_nd=True,
            accept_sparse=False,
            accept_large_sparse=False
        )


        if self.mode == "standard":
            #mean_ = np.mean(X, axis=1, keepdims=True)
            #std_ = np.std(X, axis=1, keepdims=True)
            #std_[std_ == 0.0] = 1.0

            mean_ = np.mean(X)
            std_ = np.std(X)
            if std_ == 0.0:
                std_ = 1.0

            return (X - mean_) / std_

        elif self.mode == "min_max":
            #min_ = np.min(X, axis=1, keepdims=True)
            #max_ = np.max(X, axis=1, keepdims=True)
            min_ = np.min(X)
            max_ = np.max(X)

            diff_ = max_ - min_
            diff_[diff_ == 0.0] = 1.0

            return (X - min_) / diff_

        elif self.mode == "max_abs":
            #max_abs_ = np.max(np.abs(X), axis=1, keepdims=True)
            #max_abs_[max_abs_ == 0.0] = 1.0
            max_abs_ = np.max(np.abs(X))
            if max_abs_ == 0.0:
                max_abs_ = 1.0

            return X / max_abs_

        elif self.mode == "none":
            return X

        else:
            raise ValueError(f"Unknown mode {self.mode} for time series scaler")
