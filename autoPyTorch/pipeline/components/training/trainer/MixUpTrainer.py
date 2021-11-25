from typing import Callable, Dict, Optional, Tuple, Union

from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import (
    CategoricalHyperparameter,
    UniformFloatHyperparameter,
)

import numpy as np

import torch

from autoPyTorch.constants import CLASSIFICATION_TASKS, STRING_TO_TASK_TYPES
from autoPyTorch.datasets.base_dataset import BaseDatasetPropertiesType
from autoPyTorch.pipeline.components.training.trainer.base_trainer import BaseTrainerComponent
from autoPyTorch.utils.common import HyperparameterSearchSpace, add_hyperparameter


class MixUpTrainer(BaseTrainerComponent):
    """
    References:
        Title: mixup: Beyond Empirical Risk Minimization
        Authors: Hougyi Zhang et. al.
        URL: https://arxiv.org/pdf/1710.09412.pdf%C2%A0
        Github URL: https://github.com/facebookresearch/mixup-cifar10/blob/master/train.py#L119-L138
    """
    def __init__(self, alpha: float, weighted_loss: bool = False,
                 random_state: Optional[np.random.RandomState] = None):
        """
        This class handles the training of a network for a single given epoch.

        Args:
            alpha (float): the mixup ratio

        """
        super().__init__(random_state=random_state)
        self.weighted_loss = weighted_loss
        self.alpha = alpha

    def data_preparation(self, X: torch.Tensor, y: torch.Tensor,
                         ) -> Tuple[torch.Tensor, Dict[str, np.ndarray]]:
        """
        Depending on the trainer choice, data fed to the network might be pre-processed
        on a different way. That is, in standard training we provide the data to the
        network as we receive it to the loader. Some regularization techniques, like mixup
        alter the data.

        Args:
            X (torch.Tensor): The batch training features
            y (torch.Tensor): The batch training labels

        Returns:
            torch.Tensor: that processes data
            Dict[str, np.ndarray]: arguments to the criterion function
                                          TODO: Fix this typing. It is not np.ndarray.
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        lam = self.random_state.beta(self.alpha, self.alpha) if self.alpha > 0. else 1.
        batch_size = X.shape[0]
        index = torch.randperm(batch_size).to(device)

        mixed_x = lam * X + (1 - lam) * X[index, :]
        y_a, y_b = y, y[index]
        return mixed_x, {'y_a': y_a, 'y_b': y_b, 'lam': lam}

    def criterion_preparation(self, y_a: torch.Tensor, y_b: torch.Tensor = None, lam: float = 1.0
                              ) -> Callable:
        return lambda criterion, pred: lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

    @staticmethod
    def get_properties(dataset_properties: Optional[Dict[str, BaseDatasetPropertiesType]] = None
                       ) -> Dict[str, Union[str, bool]]:
        return {
            'shortname': 'MixUpTrainer',
            'name': 'MixUp Regularized Trainer',
        }

    @staticmethod
    def get_hyperparameter_search_space(
        dataset_properties: Optional[Dict[str, BaseDatasetPropertiesType]] = None,
        alpha: HyperparameterSearchSpace = HyperparameterSearchSpace(hyperparameter="alpha",
                                                                     value_range=(0, 1),
                                                                     default_value=0.2),
        weighted_loss: HyperparameterSearchSpace = HyperparameterSearchSpace(hyperparameter="weighted_loss",
                                                                             value_range=(True, False),
                                                                             default_value=True),
    ) -> ConfigurationSpace:

        cs = ConfigurationSpace()
        add_hyperparameter(cs, alpha, UniformFloatHyperparameter)
        if dataset_properties is not None:
            if STRING_TO_TASK_TYPES[str(dataset_properties['task_type'])] in CLASSIFICATION_TASKS:
                add_hyperparameter(cs, weighted_loss, CategoricalHyperparameter)
        return cs
