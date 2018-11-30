from autoPyTorch.training.base_training import BaseBatchLossComputationTechnique
import numpy as np
from torch.autograd import Variable
import ConfigSpace
import torch

class Mixup(BaseBatchLossComputationTechnique):
    def set_up(self, pipeline_config, hyperparameter_config, logger):
        super(Mixup, self).set_up(pipeline_config, hyperparameter_config, logger)
        self.alpha = hyperparameter_config["alpha"]

    def prepare_batch_data(self, X_batch, y_batch):
        '''Returns mixed inputs, pairs of targets, and lambda'''
        if self.alpha > 0:
            self.lam = np.random.beta(self.alpha, self.alpha)
        else:
            self.lam = 1

        batch_size = X_batch.size()[0]
        if X_batch.is_cuda:
            index = torch.randperm(batch_size).cuda()
        else:
            index = torch.randperm(batch_size)

        self.mixed_x = self.lam * X_batch + (1 - self.lam) * X_batch[index, :]
        self.y_a, self.y_b = y_batch, y_batch[index]
    
    def compute_batch_loss(self, loss_function, y_batch_pred):
        # self.logger.debug("Computing batch loss with mixup")

        result = self.lam * loss_function(y_batch_pred, Variable(self.y_a)) + \
                 (1 - self.lam) * loss_function(y_batch_pred, Variable(self.y_b))
        self.lam = None
        self.mixed_x = None
        self.y_a = None
        self.y_b = None
        return result

    @staticmethod
    def get_hyperparameter_search_space(**pipeline_config):
        cs = ConfigSpace.ConfigurationSpace()
        cs.add_hyperparameter(ConfigSpace.hyperparameters.UniformFloatHyperparameter("alpha", lower=0, upper=1, default_value=1))
        return cs