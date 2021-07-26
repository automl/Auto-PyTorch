from autoPyTorch.training.base_training import BaseBatchLossComputationTechnique
import numpy as np
from torch.autograd import Variable
import ConfigSpace
import torch

class Mixup(BaseBatchLossComputationTechnique):
    def set_up(self, pipeline_config, hyperparameter_config, logger):
        super(Mixup, self).set_up(pipeline_config, hyperparameter_config, logger)
        self.alpha = hyperparameter_config["alpha"]

    def prepare_data(self, x, y):

        lam = np.random.beta(self.alpha, self.alpha) if self.alpha > 0. else 1.
        batch_size = x.size()[0]
        index = torch.randperm(batch_size).cuda() if x.is_cuda else torch.randperm(batch_size)

        mixed_x = lam * x + (1 - lam) * x[index, :]
        y_a, y_b = y, y[index]
        return mixed_x, { 'y_a': y_a, 'y_b': y_b, 'lam' : lam }

    def criterion(self, y_a, y_b, lam):
        return lambda criterion, pred: lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
        
    def evaluate(self, metric, y_pred, y_a, y_b, lam):
        return lam * metric(y_pred, y_a) + (1 - lam) * metric(y_pred, y_b)

    @staticmethod
    def get_hyperparameter_search_space(**pipeline_config):
        cs = ConfigSpace.ConfigurationSpace()
        cs.add_hyperparameter(ConfigSpace.hyperparameters.UniformFloatHyperparameter("alpha", lower=0, upper=1, default_value=1))
        return cs