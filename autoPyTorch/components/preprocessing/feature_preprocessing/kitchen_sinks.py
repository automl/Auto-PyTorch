import ConfigSpace
import ConfigSpace.hyperparameters as CSH

from autoPyTorch.components.preprocessing.preprocessor_base import PreprocessorBase

class RandomKitchenSinks(PreprocessorBase):

    def __init__(self, hyperparameter_config):
        self.gamma = float(hyperparameter_config['gamma'])
        self.n_components = int(hyperparameter_config['n_components'])

    def fit(self, X, Y):
        import sklearn.kernel_approximation

        self.preprocessor = sklearn.kernel_approximation.RBFSampler(self.gamma, self.n_components)
        self.preprocessor.fit(X)
    
    def transform(self, X):
        if self.preprocessor is None:
            raise NotImplementedError()
        return self.preprocessor.transform(X)

    @staticmethod
    def get_hyperparameter_search_space(dataset_info=None):
        gamma = CSH.UniformFloatHyperparameter("gamma", 3.0517578125e-05, 8, default_value=1.0, log=True)
        n_components = CSH.UniformIntegerHyperparameter("n_components", 50, 10000, default_value=100, log=True)
        cs = ConfigSpace.ConfigurationSpace()
        cs.add_hyperparameters([gamma, n_components])
        return cs