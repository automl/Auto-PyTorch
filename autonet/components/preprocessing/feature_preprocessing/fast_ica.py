import torch
import warnings

import ConfigSpace
import ConfigSpace.hyperparameters as CSH
import ConfigSpace.conditions as CSC

from autonet.components.preprocessing.preprocessor_base import PreprocessorBase


class FastICA(PreprocessorBase):
    def __init__(self, hyperparameter_config):
        self.algorithm = hyperparameter_config['algorithm']
        self.whiten = hyperparameter_config['whiten']
        self.fun = hyperparameter_config['fun']
        self.n_components = None
        if (self.whiten):
            self.n_components = hyperparameter_config['n_components']

    def fit(self, X, Y):
        import sklearn.decomposition

        self.preprocessor = sklearn.decomposition.FastICA(
            n_components=self.n_components, algorithm=self.algorithm,
            fun=self.fun, whiten=self.whiten
        )

        # Make the RuntimeWarning an Exception!
        with warnings.catch_warnings():
            warnings.filterwarnings("error", message='array must not contain infs or NaNs')
            try:
                return self.preprocessor.fit(X)
            except ValueError as e:
                if 'array must not contain infs or NaNs' in e.args[0]:
                    raise ValueError("Bug in scikit-learn: https://github.com/scikit-learn/scikit-learn/pull/2738")


    def transform(self, X):
        if self.preprocessor is None:
            raise NotImplementedError()
        return self.preprocessor.transform(X)

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None):
        cs = ConfigSpace.ConfigurationSpace()

        n_components = CSH.UniformIntegerHyperparameter("n_components", lower=10, upper=2000)
        algorithm = CSH.CategoricalHyperparameter('algorithm', ['parallel', 'deflation'])
        whiten = CSH.CategoricalHyperparameter('whiten', [True, False])
        fun = CSH.CategoricalHyperparameter('fun', ['logcosh', 'exp', 'cube'])
        cs.add_hyperparameters([n_components, algorithm, whiten, fun])

        cs.add_condition(CSC.EqualsCondition(n_components, whiten, True))

        return cs
