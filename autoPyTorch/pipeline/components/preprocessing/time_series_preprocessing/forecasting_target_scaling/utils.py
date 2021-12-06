from typing import Any, Dict, Callable, Optional, Union, Tuple

import torch
import sklearn
from sklearn.base import BaseEstimator


# Similar to / inspired by
# https://github.com/tslearn-team/tslearn/blob/a3cf3bf/tslearn/preprocessing/preprocessing.py
class TargetScaler(BaseEstimator):
    """
    To accelerate training, this scaler is only applied under trainer (after the data is loaded by dataloader)
    """
    def __init__(self, mode: str):
        self.mode = mode

    def fit(self, X: Dict, y: Any = None) -> "TimeSeriesScalerBatch":
        return self

    def transform(self, X: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        if self.mode == "standard":
            loc = torch.mean(X, dim=-2, keepdim=True)
            scale = torch.std(X, dim=-2, keepdim=True)
            scale[scale == 0.0] = 1.0
            return (X - loc) / scale, loc, scale

        elif self.mode == "min_max":
            min_ = torch.min(X, dim=-2, keepdim=True)[0]
            max_ = torch.max(X, dim=-2, keepdim=True)[0]

            diff_ = max_ - min_
            loc = min_
            scale = diff_
            scale[scale == 0.0] = 1.0
            return (X - loc) / scale, loc, scale

        elif self.mode == "max_abs":
            max_abs_ = torch.max(torch.abs(X), dim=-2, keepdim=True)[0]
            max_abs_[max_abs_ == 0.0] = 1.0
            scale = max_abs_
            return X / scale, None, scale

        elif self.mode == "none":
            return X, None, None

        else:
            raise ValueError(f"Unknown mode {self.mode} for Forecasting scaler")