from typing import Any, Dict, Optional, Tuple

from sklearn.base import BaseEstimator

import torch

from autoPyTorch.constants import VERY_SMALL_VALUE


# Similar to / inspired by
# https://github.com/tslearn-team/tslearn/blob/a3cf3bf/tslearn/preprocessing/preprocessing.py
class TargetScaler(BaseEstimator):
    """
    To accelerate training, this scaler is only applied under trainer (after the data is loaded by dataloader)
    """

    def __init__(self, mode: str):
        self.mode = mode

    def fit(self, X: Dict, y: Any = None) -> "TargetScaler":
        return self

    def transform(self,
                  past_targets: torch.Tensor,
                  past_observed_values: torch.BoolTensor,
                  future_targets: Optional[torch.Tensor] = None) -> \
            Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        if past_observed_values is None or torch.all(past_observed_values):
            if self.mode == "standard":
                loc = torch.mean(past_targets, dim=1, keepdim=True)
                scale = torch.std(past_targets, dim=1, keepdim=True)

                offset_targets = past_targets - loc
                scale = torch.where(torch.logical_or(scale == 0.0, scale == torch.nan), offset_targets[:, [-1]], scale)
                scale[scale < VERY_SMALL_VALUE] = 1.0
                if future_targets is not None:
                    future_targets = (future_targets - loc) / scale
                return (past_targets - loc) / scale, future_targets, loc, scale

            elif self.mode == "min_max":
                min_ = torch.min(past_targets, dim=1, keepdim=True)[0]
                max_ = torch.max(past_targets, dim=1, keepdim=True)[0]

                diff_ = max_ - min_
                loc = min_
                scale = torch.where(diff_ == 0, past_targets[:, [-1]], diff_)
                scale[scale < VERY_SMALL_VALUE] = 1.0
                if future_targets is not None:
                    future_targets = (future_targets - loc) / scale
                return (past_targets - loc) / scale, future_targets, loc, scale

            elif self.mode == "max_abs":
                max_abs_ = torch.max(torch.abs(past_targets), dim=1, keepdim=True)[0]
                max_abs_[max_abs_ < VERY_SMALL_VALUE] = 1.0
                scale = max_abs_
                if future_targets is not None:
                    future_targets = future_targets / scale
                return past_targets / scale, future_targets, None, scale

            elif self.mode == 'mean_abs':
                mean_abs = torch.mean(torch.abs(past_targets), dim=1, keepdim=True)
                scale = torch.where(mean_abs == 0.0, past_targets[:, [-1]], mean_abs)
                scale[scale < VERY_SMALL_VALUE] = 1.0
                if future_targets is not None:
                    future_targets = future_targets / scale
                return past_targets / scale, future_targets, None, scale

            elif self.mode == "none":
                return past_targets, future_targets, None, None

            else:
                raise ValueError(f"Unknown mode {self.mode} for Forecasting scaler")
        else:
            valid_past_targets = past_observed_values * past_targets
            valid_past_obs = torch.sum(past_observed_values, dim=1, keepdim=True)
            if self.mode == "standard":
                dfredom = 1
                loc = torch.sum(valid_past_targets, dim=1, keepdim=True) / valid_past_obs
                scale = torch.sum(torch.square((valid_past_targets - loc * past_observed_values)), dim=1, keepdim=True)

                scale /= valid_past_obs - dfredom
                scale = torch.sqrt(scale)

                offset_targets = past_targets - loc
                # ensure that all the targets are scaled properly
                scale = torch.where(torch.logical_or(scale == 0.0, scale == torch.nan), offset_targets[:, [-1]], scale)
                scale[scale < VERY_SMALL_VALUE] = 1.0

                if future_targets is not None:
                    future_targets = (future_targets - loc) / scale

                scaled_past_targets = torch.where(past_observed_values, offset_targets / scale, past_targets)
                return scaled_past_targets, future_targets, loc, scale

            elif self.mode == "min_max":
                obs_mask = ~past_observed_values
                min_masked_past_targets = past_targets.masked_fill(obs_mask, value=torch.inf)
                max_masked_past_targets = past_targets.masked_fill(obs_mask, value=-torch.inf)
                min_ = torch.min(min_masked_past_targets, dim=1, keepdim=True)[0]
                max_ = torch.max(max_masked_past_targets, dim=1, keepdim=True)[0]

                diff_ = max_ - min_
                loc = min_
                scale = torch.where(diff_ == 0, past_targets[:, [-1]], diff_)
                scale[scale < VERY_SMALL_VALUE] = 1.0

                if future_targets is not None:
                    future_targets = (future_targets - loc) / scale
                scaled_past_targets = torch.where(past_observed_values, (past_targets - loc) / scale, past_targets)

                return scaled_past_targets, future_targets, loc, scale

            elif self.mode == "max_abs":
                max_abs_ = torch.max(torch.abs(valid_past_targets), dim=1, keepdim=True)[0]
                max_abs_[max_abs_ < VERY_SMALL_VALUE] = 1.0
                scale = max_abs_
                if future_targets is not None:
                    future_targets = future_targets / scale

                scaled_past_targets = torch.where(past_observed_values, past_targets / scale, past_targets)

                return scaled_past_targets, future_targets, None, scale

            elif self.mode == 'mean_abs':
                mean_abs = torch.sum(torch.abs(valid_past_targets), dim=1, keepdim=True) / valid_past_obs
                scale = torch.where(mean_abs == 0.0, valid_past_targets[:, [-1]], mean_abs)
                # in case that all values in the tensor is too small
                scale[scale < VERY_SMALL_VALUE] = 1.0
                if future_targets is not None:
                    future_targets = future_targets / scale

                scaled_past_targets = torch.where(past_observed_values, past_targets / scale, past_targets)
                return scaled_past_targets, future_targets, None, scale

            elif self.mode == "none":
                return past_targets, future_targets, None, None

            else:
                raise ValueError(f"Unknown mode {self.mode} for Forecasting scaler")
