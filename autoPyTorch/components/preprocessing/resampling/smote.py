from autoPyTorch.components.preprocessing.resampling_base import ResamplingMethodBase
from autoPyTorch.utils.config_space_hyperparameter import add_hyperparameter, get_hyperparameter
import ConfigSpace
import ConfigSpace.hyperparameters as CSH

class SMOTE(ResamplingMethodBase):
    def __init__(self, hyperparameter_config):
        self.k_neighbors = hyperparameter_config["k_neighbors"]

    def resample(self, X, y, target_size_strategy, seed):
        from imblearn.over_sampling import SMOTE as imblearn_SMOTE
        k_neighbors = self.k_neighbors
        resampler = imblearn_SMOTE(sampling_strategy=target_size_strategy, k_neighbors=k_neighbors, random_state=seed)
        return resampler.fit_resample(X, y)

    @staticmethod
    def get_hyperparameter_search_space(
        k_neighbors=(3, 7)
    ):
        k_neighbors = get_hyperparameter(CSH.UniformIntegerHyperparameter, "k_neighbors", k_neighbors)
        cs = ConfigSpace.ConfigurationSpace()
        cs.add_hyperparameter(k_neighbors)
        return cs