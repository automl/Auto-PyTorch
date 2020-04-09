from autoPyTorch.components.training.base_training import BaseBatchLossComputationTechnique
from autoPyTorch.utils.config_space_hyperparameter import add_hyperparameter
import numpy as np
from torch.autograd import Variable
import ConfigSpace
import torch

class CutMix(BaseBatchLossComputationTechnique):
    def set_up(self, pipeline_config, hyperparameter_config, logger):
        super(CutMix, self).set_up(pipeline_config, hyperparameter_config, logger)
        self.beta = hyperparameter_config["beta"]
        self.cutmix_prob = hyperparameter_config["cutmix_prob"]

    def prepare_data(self, x, y):
        lam = np.random.beta(self.beta, self.beta)
        batch_size = x.size()[0]
        index = torch.randperm(batch_size).cuda() if x.is_cuda else torch.randperm(batch_size)

        r = np.random.rand(1)
        if self.beta <= 0 or r > self.cutmix_prob:
            return x, { 'y_a': y, 'y_b': y[index], 'lam' : 1 }

        # Draw parameters of a random bounding box
        bbx1, bbx2 = self.rand_bbox(x.size(), lam)

        x[:, bbx1:bbx2] = x[index, bbx1:bbx2]

        #Adjust lam
        lam = 1 - ((bbx2 - bbx1) / (x.size()[1]))

        y_a, y_b = y, y[index]

        return x, { 'y_a': y_a, 'y_b': y_b, 'lam' : lam }

    def criterion(self, y_a, y_b, lam):
        return lambda criterion, pred: lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

    @staticmethod
    def get_hyperparameter_search_space(
        beta=(1.0, 1.0),
        cutmix_prob=(0.0,1.0)
    ):
        cs = ConfigSpace.ConfigurationSpace()
        add_hyperparameter(cs, ConfigSpace.hyperparameters.UniformFloatHyperparameter, "beta", beta)
        add_hyperparameter(cs, ConfigSpace.hyperparameters.UniformFloatHyperparameter, "cutmix_prob", cutmix_prob)
        return cs

    def rand_bbox(self, size, lam):
        L = size[1]
        cut_rat = np.sqrt(1. - lam)
        cut_l = np.int(L * cut_rat)
        # uniform
        cx = np.random.randint(L)

        bbx1 = np.clip(cx - cut_l // 2, 0, L)
        bbx2 = np.clip(cx + cut_l // 2, 0, L)

        return bbx1, bbx2
        