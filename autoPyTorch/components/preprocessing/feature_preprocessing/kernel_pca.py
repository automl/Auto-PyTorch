import warnings

import numpy as np

import ConfigSpace
import ConfigSpace.hyperparameters as CSH
import ConfigSpace.conditions as CSC

from autoPyTorch.utils.config_space_hyperparameter import add_hyperparameter
from autoPyTorch.components.preprocessing.preprocessor_base import PreprocessorBase


class KernelPCA(PreprocessorBase):
    def __init__(self, hyperparameter_config):
        self.n_components = int(hyperparameter_config['n_components'])
        self.kernel = hyperparameter_config['kernel']
        
        self.degree = int(hyperparameter_config['degree']) if self.kernel == 'poly' else 3
        self.gamma = float(hyperparameter_config['gamma']) if self.kernel in ['poly', 'rbf'] else 0.25
        self.coef0 = float(hyperparameter_config['coef0']) if self.kernel in ['poly', 'sigmoid'] else 0.0

    def fit(self, X, Y=None):
        import scipy.sparse
        import sklearn.decomposition

        self.preprocessor = sklearn.decomposition.KernelPCA(
            n_components=self.n_components, kernel=self.kernel,
            degree=self.degree, gamma=self.gamma, coef0=self.coef0,
            remove_zero_eig=True)

        if scipy.sparse.issparse(X):
            X = X.astype(np.float64)
        with warnings.catch_warnings():
            warnings.filterwarnings("error")
            self.preprocessor.fit(X)

        # Raise an informative error message, equation is based ~line 249 in
        # kernel_pca.py in scikit-learn
        if len(self.preprocessor.alphas_ / self.preprocessor.lambdas_) == 0:
            raise ValueError('KernelPCA removed all features!')
        return self

    def transform(self, X):
        if self.preprocessor is None:
            raise NotImplementedError()
        with warnings.catch_warnings():
            warnings.filterwarnings("error")
            X_new = self.preprocessor.transform(X)

            # TODO write a unittest for this case
            if X_new.shape[1] == 0:
                raise ValueError("KernelPCA removed all features!")

            return X_new

    @staticmethod
    def get_hyperparameter_search_space(
        dataset_info=None,
        kernel=('poly', 'rbf', 'sigmoid', 'cosine'),
        n_components=(10, 2000),
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

