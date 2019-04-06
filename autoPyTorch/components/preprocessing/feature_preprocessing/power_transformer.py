import numpy as np
import torch

from autoPyTorch.utils.config_space_hyperparameter import add_hyperparameter, get_hyperparameter

import ConfigSpace
import ConfigSpace.hyperparameters as CSH

from autoPyTorch.components.preprocessing.preprocessor_base import PreprocessorBase



class PowerTransformer(PreprocessorBase):
    def __init__(self, hyperparameter_config):
        self.preprocessor = None
        self.method = hyperparameter_config["method"] if "method" in hyperparameter_config else "yeo-johnson"
        self.standardize = hyperparameter_config["standardize"]

    def fit(self, X, Y):
        import sklearn.preprocessing

        try:
            self.preprocessor = sklearn.preprocessing.PowerTransformer(method=self.method, standardize=self.standardize, copy=False)
            self.preprocessor.fit(X, Y)
        except ValueError as exception:
            print(exception)
            print("Using yeo-johnson instead")
            self.preprocessor = sklearn.preprocessing.PowerTransformer(standardize=self.standardize, copy=False)
            self.preprocessor.fit(X, Y)

    def transform(self, X):
        if self.preprocessor is None:
            raise NotImplementedError()
        return self.preprocessor.transform(X)

    @staticmethod
    def get_hyperparameter_search_space(
        dataset_info=None,
        standardize=(True, False),
        method=("yeo-johnson", "box-cox"),
    ):
        cs = ConfigSpace.ConfigurationSpace()
        add_hyperparameter(cs, CSH.CategoricalHyperparameter, "standardize", standardize)
        if dataset_info is None or (
                (dataset_info.x_min_value is None or dataset_info.x_min_value > 0) and not any(dataset_info.categorical_features)):
            add_hyperparameter(cs, CSH.CategoricalHyperparameter, "method", method)
        return cs