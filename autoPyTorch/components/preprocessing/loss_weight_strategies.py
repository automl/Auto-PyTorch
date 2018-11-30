__author__ = "Max Dippel, Michael Burkart and Matthias Urban"
__version__ = "0.0.1"
__license__ = "BSD"


import numpy as np


class LossWeightStrategyWeighted():
    def __call__(self, pipeline_config, X, Y):
        
        classes, counts = np.unique(Y, axis=0, return_counts=True)
        classes, counts = classes[::-1], counts[::-1]

        total_weight = Y.shape[0]
        weight_per_class = total_weight / classes.shape[0]
        weights = (np.ones(classes.shape[0]) * weight_per_class) / counts
        return weights

class LossWeightStrategyWeightedBinary():
    def __call__(self, pipeline_config, X, Y):
        
        classes, counts = np.unique(Y, axis=0, return_counts=True)
        classes, counts = classes[::-1], counts[::-1]

        weights = []
        for i in range(Y.shape[1]):
            _, counts = np.unique(Y[:, i], return_counts=True)
            weights.append(counts[0] / counts[1])
        weights = np.array(weights)

        return weights
