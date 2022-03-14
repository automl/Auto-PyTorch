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

    def transform(self, past_targets: torch.Tensor, future_targets: Optional[torch.Tensor]=None) -> \
            Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        if self.mode == "standard":
            loc = torch.mean(past_targets, dim=-2, keepdim=True)
            scale = torch.std(past_targets, dim=-2, keepdim=True)
            scale[scale == 0.0] = 1.0
            if future_targets is not None:
                future_targets = (future_targets - loc) / scale
            return (past_targets - loc) / scale, future_targets, loc, scale

        elif self.mode == "min_max":
            min_ = torch.min(past_targets, dim=-2, keepdim=True)[0]
            max_ = torch.max(past_targets, dim=-2, keepdim=True)[0]

            diff_ = max_ - min_
            loc = min_ - 1e-10
            scale = diff_
            scale[scale == 0.0] = 1.0
            if future_targets is not None:
                future_targets = (future_targets - loc) / scale
            return (past_targets - loc) / scale, future_targets, loc, scale

        elif self.mode == "max_abs":
            max_abs_ = torch.max(torch.abs(past_targets), dim=-2, keepdim=True)[0]
            max_abs_[max_abs_ == 0.0] = 1.0
            scale = max_abs_
            if future_targets is not None:
                future_targets = future_targets / scale
            return past_targets / scale, future_targets, None, scale

        elif self.mode == 'mean_abs':
            mean_abs = torch.mean(torch.abs(past_targets), dim=1,  keepdim=True)
            mean_abs[mean_abs == 0.0] = 1.0
            scale = mean_abs
            if future_targets is not None:
                future_targets = future_targets / scale
            return past_targets / scale, future_targets, None, scale


        elif self.mode == "none":
            return past_targets, future_targets, None, None

        else:
            raise ValueError(f"Unknown mode {self.mode} for Forecasting scaler")