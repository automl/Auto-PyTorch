from autoPyTorch.components.training.base_training import BaseBatchLossComputationTechnique
from autoPyTorch.utils.config_space_hyperparameter import add_hyperparameter
import numpy as np
from torch.autograd import Variable
import ConfigSpace
import torch
import random

class CutOut(BaseBatchLossComputationTechnique):
    def set_up(self, pipeline_config, hyperparameter_config, logger):
        super(CutOut, self).set_up(pipeline_config, hyperparameter_config, logger)
        self.patch_ratio = hyperparameter_config["patch_ratio"]
        self.cutout_prob = hyperparameter_config["cutout_prob"]

    def prepare_data(self, x, y):
        r = np.random.rand(1)
        if r > self.cutout_prob:
            y_a = y
            y_b = y
            lam = 1
            return x, { 'y_a': y_a, 'y_b': y_b, 'lam' : lam }

        # Draw a uniform sample of indices which denotes what to turn off 
        indices = self.rand_indices(x.size(), self.patch_ratio)

        x[:, indices.long()] = 0.0
        lam = 1
        y_a = y
        y_b = y

        return x, { 'y_a': y_a, 'y_b': y_b, 'lam' : lam }
 
    def criterion(self, y_a, y_b, lam):
        return lambda criterion, pred: lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

    @staticmethod
    def get_hyperparameter_search_space(
        patch_ratio=(0.0, 1.0),
        cutout_prob=(0.0,1.0)
    ):
        cs = ConfigSpace.ConfigurationSpace()
        add_hyperparameter(cs, ConfigSpace.hyperparameters.UniformFloatHyperparameter, "patch_ratio", patch_ratio)
        add_hyperparameter(cs, ConfigSpace.hyperparameters.UniformFloatHyperparameter, "cutout_prob", cutout_prob)
        return cs

    def rand_indices(self, size, patch_ratio):
        L = size[1]
        k_choose = np.int(L * patch_ratio)
        # sample indices
        sample_indices = random.sample(range(L), k_choose)

        return torch.tensor(sample_indices)
        