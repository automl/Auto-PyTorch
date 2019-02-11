import numpy as np
import torch

import ConfigSpace
import ConfigSpace.hyperparameters as CSH

from autoPyTorch.components.preprocessing.preprocessor_base import PreprocessorBase



class TruncatedSVD(PreprocessorBase):
    def __init__(self, hyperparameter_config):
        self.target_dim = hyperparameter_config['target_dim']
        self.preprocessor = None

    def fit(self, X, Y):
        import sklearn.decomposition

        self.target_dim = int(self.target_dim)
        target_dim = min(self.target_dim, X.shape[1] - 1)
        self.preprocessor = sklearn.decomposition.TruncatedSVD(target_dim, algorithm='randomized')
        self.preprocessor.fit(X, Y)

    def transform(self, X):
        if self.preprocessor is None:
            raise NotImplementedError()
        return self.preprocessor.transform(X)

    @staticmethod
    def get_hyperparameter_search_space(dataset_info=None):
        cs = ConfigSpace.ConfigurationSpace()
        cs.add_hyperparameter(CSH.UniformIntegerHyperparameter("target_dim", lower=10, upper=256, default_value=128))
        return cs