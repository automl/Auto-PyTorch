import torch

import ConfigSpace
import ConfigSpace.hyperparameters as CSH

from autoPyTorch.utils.config_space_hyperparameter import get_hyperparameter, add_hyperparameter
from autoPyTorch.components.preprocessing.preprocessor_base import PreprocessorBase


class PolynomialFeatures(PreprocessorBase):
    def __init__(self, hyperparameter_config):
        self.degree = hyperparameter_config['degree']
        self.interaction_only = hyperparameter_config['interaction_only']
        self.include_bias = hyperparameter_config['include_bias']
        self.preprocessor = None

    def fit(self, X, Y):
        import sklearn.preprocessing

        self.preprocessor = sklearn.preprocessing.PolynomialFeatures(
            degree=self.degree, interaction_only=self.interaction_only,
            include_bias=self.include_bias)

        self.preprocessor.fit(X, Y)

    def transform(self, X):
        if self.preprocessor is None:
            raise NotImplementedError()
        return self.preprocessor.transform(X)

    @staticmethod
    def get_hyperparameter_search_space(
        dataset_info=None,
        degree=(2, 3),
        interaction_only=(True, False),
        include_bias=(True, False)
    ):
        cs = ConfigSpace.ConfigurationSpace()
        add_hyperparameter(cs, CSH.UniformIntegerHyperparameter, "degree", degree)
        add_hyperparameter(cs, CSH.CategoricalHyperparameter, "interaction_only", [False, True])
        add_hyperparameter(cs, CSH.CategoricalHyperparameter, "include_bias", [True, False])

        return cs