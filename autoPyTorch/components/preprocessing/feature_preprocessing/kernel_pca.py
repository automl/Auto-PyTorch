import warnings

import numpy as np

import ConfigSpace
import ConfigSpace.hyperparameters as CSH
import ConfigSpace.conditions as CSC

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
    def get_hyperparameter_search_space(dataset_info=None):
        n_components = CSH.UniformIntegerHyperparameter("n_components", 10, 2000, default_value=100)
        kernel = CSH.CategoricalHyperparameter('kernel', ['poly', 'rbf', 'sigmoid', 'cosine'], 'rbf')
        gamma = CSH.UniformFloatHyperparameter("gamma", 3.0517578125e-05, 8, log=True, default_value=1.0)
        degree = CSH.UniformIntegerHyperparameter('degree', 2, 5, 3)
        coef0 = CSH.UniformFloatHyperparameter("coef0", -1, 1, default_value=0)
        cs = ConfigSpace.ConfigurationSpace()
        cs.add_hyperparameters([n_components, kernel, degree, gamma, coef0])

        degree_depends_on_poly = CSC.EqualsCondition(degree, kernel, "poly")
        coef0_condition = CSC.InCondition(coef0, kernel, ["poly", "sigmoid"])
        gamma_condition = CSC.InCondition(gamma, kernel, ["poly", "rbf"])
        cs.add_conditions([degree_depends_on_poly, coef0_condition, gamma_condition])
        return cs

