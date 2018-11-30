from autoPyTorch.components.preprocessing.resampling_base import ResamplingMethodBase
import ConfigSpace
import ConfigSpace.hyperparameters as CSH

class SMOTE(ResamplingMethodBase):
    def __init__(self, hyperparameter_config):
        self.k_neighbors = hyperparameter_config["k_neighbors"]

    def resample(self, X, y, target_size_strategy):
        from imblearn.over_sampling import SMOTE as imblearn_SMOTE
        k_neighbors = self.k_neighbors
        resampler = imblearn_SMOTE(sampling_strategy=target_size_strategy, k_neighbors=k_neighbors)
        return resampler.fit_resample(X, y)

    @staticmethod
    def get_hyperparameter_search_space():
        k_neighbors = CSH.UniformIntegerHyperparameter("k_neighbors", lower=3, upper=7)
        cs = ConfigSpace.ConfigurationSpace()
        cs.add_hyperparameter(k_neighbors)
        return cs