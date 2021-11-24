from typing import Any, List, Callable, Optional

import numpy as np

import sklearn
from sklearn.base import BaseEstimator


# Similar to / inspired by
# https://github.com/tslearn-team/tslearn/blob/a3cf3bf/tslearn/preprocessing/preprocessing.py
class TimeSeriesScaler(BaseEstimator):
    def __init__(self, mode: str, sequence_lengths_train: List[int], is_training=True):
        self.mode = mode
        self.sequence_lengths_train = sequence_lengths_train
        self.is_training = is_training

    def fit(self, X: np.ndarray, y: Any = None) -> "TimeSeriesScaler":
        """
        For time series we do not need to fit anything since each time series is scaled individually
        """
        return self

    def eval(self):
        self.is_training = False

    def scale_individual_seq(self, X, scaling: Callable):
        idx_start = 0
        for seq_length_train in self.sequence_lengths_train:
            idx_end = seq_length_train + idx_start
            X[idx_start: idx_end] = scaling(X[idx_start: idx_end])
            idx_start = idx_end
        return X

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        X = sklearn.utils.check_array(
            X,
            force_all_finite=True,
            ensure_2d=False,
            allow_nd=True,
            accept_sparse=False,
            accept_large_sparse=False
        )
        """
        if self.mode == "standard":
            #mean_ = np.mean(X, axis=1, keepdims=True)
            #std_ = np.std(X, axis=1, keepdims=True)
            #std_[std_ == 0.0] = 1.0

            def standard_scaling(x_seq):
                mean_ = np.mean(x_seq)
                std_ = np.std(x_seq)
                if std_ == 0.0:
                    std_ = 1.0
                return (x_seq - mean_) / std_

            if self.is_training:
                return self.scale_individual_seq(X, standard_scaling)
            else:
                return standard_scaling(X)

        elif self.mode == "min_max":
            #min_ = np.min(X, axis=1, keepdims=True)
            #max_ = np.max(X, axis=1, keepdims=True)
            def min_max_scaling(x_seq):
                min_ = np.min(x_seq)
                max_ = np.max(x_seq)

                diff_ = max_ - min_
                if diff_ == 0.0:
                    diff_ = 1.0

                return (x_seq - min_) / diff_
            if self.is_training:
                return self.scale_individual_seq(X, min_max_scaling)
            else:
                return min_max_scaling(X)


        elif self.mode == "max_abs":
            #max_abs_ = np.max(np.abs(X), axis=1, keepdims=True)
            #max_abs_[max_abs_ == 0.0] = 1.0
            def max_abs_scaling(x_seq):
                max_abs_ = np.max(np.abs(x_seq))
                if max_abs_ == 0.0:
                    max_abs_ = 1.0

                return x_seq / max_abs_
            if self.is_training:
                return self.scale_individual_seq(X, max_abs_scaling)
            else:
                return max_abs_scaling(X)

        elif self.mode == "none":
            return X

        else:
            raise ValueError(f"Unknown mode {self.mode} for time series scaler")
