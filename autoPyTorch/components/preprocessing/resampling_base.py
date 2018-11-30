import numpy as np
import ConfigSpace

class TargetSizeStrategyBase():
    def over_sample_strategy(self, y):
        result = dict()
        targets, counts = np.unique(y, return_counts=True)
        target_size = self.get_target_size(targets, counts)
        for target, count in zip(targets, counts):
            if target_size > count:
                result[target] = target_size
        return result

    def under_sample_strategy(self, y):
        result = dict()
        targets, counts = np.unique(y, return_counts=True)
        target_size = self.get_target_size(targets, counts)
        for target, count in zip(targets, counts):
            if target_size < count:
                result[target] = target_size
        return result
    
    def get_target_size(self, targets, counts):
        raise NotImplementedError()


class ResamplingMethodBase():
    def __init__(self, hyperparameter_config):
        pass

    def resample(self, X, y, target_size_strategy):
        """Fit preprocessor with X and y.
        
        Arguments:
            X {tensor} -- feature matrix
            y {tensor} -- labels
            target_size_strategy {dict} -- determine target size for each label
        """
        raise NotImplementedError()

    @staticmethod
    def get_hyperparameter_search_space():
        cs = ConfigSpace.ConfigurationSpace()
        return cs


class ResamplingMethodNone(ResamplingMethodBase):
    def resample(self, X, y, target_size_strategy):
        return X, y