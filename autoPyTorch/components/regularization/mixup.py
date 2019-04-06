from autoPyTorch.components.training.base_training import BaseBatchLossComputationTechnique
from autoPyTorch.utils.config_space_hyperparameter import add_hyperparameter
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

    @staticmethod
    def get_hyperparameter_search_space(
        alpha=(0, 1)
    ):
        cs = ConfigSpace.ConfigurationSpace()
        add_hyperparameter(cs, ConfigSpace.hyperparameters.UniformFloatHyperparameter, "alpha", alpha)
        return cs