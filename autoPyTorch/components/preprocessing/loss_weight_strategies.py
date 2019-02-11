__author__ = "Max Dippel, Michael Burkart and Matthias Urban"
__version__ = "0.0.1"
__license__ = "BSD"


import numpy as np


class LossWeightStrategyWeighted():
    def __call__(self, pipeline_config, X, Y):
        
        counts = np.sum(Y, axis=0)
        total_weight = Y.shape[0]
        weight_per_class = total_weight / Y.shape[1]

        weights = (np.ones(Y.shape[1]) * weight_per_class) / np.maximum(counts, 1)
        return weights

class LossWeightStrategyWeightedBinary():
    def __call__(self, pipeline_config, X, Y):
        
        counts_one = np.sum(Y, axis=0)
        counts_zero = counts_one + (-Y.shape[0])
        weights = counts_zero / np.maximum(counts_one, 1)

        return weights
