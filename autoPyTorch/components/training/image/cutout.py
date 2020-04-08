from autoPyTorch.components.training.base_training import BaseBatchLossComputationTechnique
from autoPyTorch.utils.config_space_hyperparameter import add_hyperparameter
import numpy as np
from torch.autograd import Variable
import ConfigSpace
import torch

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

        # Draw parameters of a random bounding box
        bbx1, bby1, bbx2, bby2 = self.rand_bbox(x.size(), self.patch_ratio)

        x[:, :, bbx1:bbx2, bby1:bby2] = 0.0
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

    def rand_bbox(self, size, patch_ratio):
        W = size[2]
        H = size[3]
        cut_rat = patch_ratio
        cut_w = np.int(W * cut_rat)
        cut_h = np.int(H * cut_rat)

        # uniform
        cx = np.random.randint(W)
        cy = np.random.randint(H)

        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)

        return bbx1, bby1, bbx2, bby2
        