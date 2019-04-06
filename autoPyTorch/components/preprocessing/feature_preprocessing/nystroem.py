import numpy as np

import ConfigSpace
import ConfigSpace.hyperparameters as CSH
import ConfigSpace.conditions as CSC

from autoPyTorch.utils.config_space_hyperparameter import get_hyperparameter, add_hyperparameter
from autoPyTorch.components.preprocessing.preprocessor_base import PreprocessorBase


class Nystroem(PreprocessorBase):
    def __init__(self, hyperparameter_config):
        self.kernel = hyperparameter_config['kernel']
        self.n_components = int(hyperparameter_config['n_components'])
        self.gamma = float(hyperparameter_config['gamma']) if self.kernel in ["poly", "rbf", "sigmoid"] else 1.0
        self.degree =  int(hyperparameter_config['degree']) if self.kernel == "poly" else 3
        self.coef0 = float(hyperparameter_config['coef0']) if self.kernel in ["poly", "sigmoid"] else 1

    def fit(self, X, Y=None):
        import sklearn.kernel_approximation

        self.preprocessor = sklearn.kernel_approximation.Nystroem(
            kernel=self.kernel, n_components=self.n_components,
            gamma=self.gamma, degree=self.degree, coef0=self.coef0)

        self.preprocessor.fit(X.astype(np.float64))
        return self

    def transform(self, X):
        if self.preprocessor is None:
            raise NotImplementedError()
        return self.preprocessor.transform(X)

    @staticmethod
    def get_hyperparameter_search_space(
        dataset_info=None,
        kernel=('poly', 'rbf', 'sigmoid', 'cosine'),
        n_components=((50, 10000), True),
        gamma=((3.0517578125e-05, 8), True),
        degree=(2, 5),
        coef0=(-1, 1)
    ):
        cs = ConfigSpace.ConfigurationSpace()
        kernel_hp = add_hyperparameter(cs, CSH.CategoricalHyperparameter, 'kernel', kernel)
        add_hyperparameter(cs, CSH.UniformIntegerHyperparameter, "n_components", n_components)

        if "poly" in kernel:
            degree_hp = add_hyperparameter(cs, CSH.UniformIntegerHyperparameter, 'degree', degree)
            cs.add_condition(CSC.EqualsCondition(degree_hp, kernel_hp, "poly"))
        if set(["poly", "sigmoid"]) & set(kernel):
            coef0_hp = add_hyperparameter(cs, CSH.UniformFloatHyperparameter, "coef0", coef0)
            cs.add_condition(CSC.InCondition(coef0_hp, kernel_hp, list(set(["poly", "sigmoid"]) & set(kernel))))
        if set(["poly", "rbf", "sigmoid"]) & set(kernel):
            gamma_hp = add_hyperparameter(cs, CSH.UniformFloatHyperparameter, "gamma", gamma)
            cs.add_condition(CSC.InCondition(gamma_hp, kernel_hp, list(set(["poly", "rbf", "sigmoid"]) & set(kernel))))

        return cs