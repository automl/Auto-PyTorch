import numpy as np

import ConfigSpace
import ConfigSpace.hyperparameters as CSH
import ConfigSpace.conditions as CSC

from autonet.components.preprocessing.preprocessor_base import PreprocessorBase


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
    def get_hyperparameter_search_space():

        possible_kernels = ['poly', 'rbf', 'sigmoid', 'cosine']
        kernel = CSH.CategoricalHyperparameter('kernel', possible_kernels, 'rbf')
        n_components = CSH.UniformIntegerHyperparameter("n_components", 50, 10000, default_value=100, log=True)
        gamma = CSH.UniformFloatHyperparameter("gamma", 3.0517578125e-05, 8, log=True, default_value=0.1)
        degree = CSH.UniformIntegerHyperparameter('degree', 2, 5, 3)
        coef0 = CSH.UniformFloatHyperparameter("coef0", -1, 1, default_value=0)

        cs = ConfigSpace.ConfigurationSpace()
        cs.add_hyperparameters([kernel, degree, gamma, coef0, n_components])

        degree_depends_on_poly = CSC.EqualsCondition(degree, kernel, "poly")
        coef0_condition = CSC.InCondition(coef0, kernel, ["poly", "sigmoid"])

        gamma_kernels = ["poly", "rbf", "sigmoid"]
        gamma_condition = CSC.InCondition(gamma, kernel, gamma_kernels)
        cs.add_conditions([degree_depends_on_poly, coef0_condition, gamma_condition])
        return cs