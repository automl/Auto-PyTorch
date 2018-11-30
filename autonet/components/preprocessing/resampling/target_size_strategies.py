import numpy as np
from autonet.components.preprocessing.resampling_base import TargetSizeStrategyBase

class TargetSizeStrategyUpsample(TargetSizeStrategyBase):
    def get_target_size(self, targets, counts):
        return int(np.max(counts))

class TargetSizeStrategyDownsample(TargetSizeStrategyBase):
    def get_target_size(self, targets, counts):
        return int(np.min(counts))

class TargetSizeStrategyAverageSample(TargetSizeStrategyBase):
    def get_target_size(self, targets, counts):
        return int(np.average(counts))

class TargetSizeStrategyMedianSample(TargetSizeStrategyBase):
    def get_target_size(self, targets, counts):
        return int(np.median(counts))
