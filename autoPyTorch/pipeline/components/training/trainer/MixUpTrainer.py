from typing import Any, Callable, Dict, Optional, Tuple

from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import (
    CategoricalHyperparameter,
    UniformFloatHyperparameter,
)

import numpy as np

import torch

from autoPyTorch.constants import CLASSIFICATION_TASKS, STRING_TO_TASK_TYPES
from autoPyTorch.pipeline.components.training.trainer.base_trainer import (
    BaseTrainerComponent,
    _NewLossParameters
)
from autoPyTorch.utils.common import HyperparameterSearchSpace, add_hyperparameter


class MixUpTrainer(BaseTrainerComponent):
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

    def _data_preprocessing(self, X: torch.Tensor, y: torch.Tensor,
                            ) -> Tuple[torch.Tensor, _NewLossParameters]:
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
            _NewLossParameters: arguments to the new loss function
        """
        lam = self.random_state.beta(self.alpha, self.alpha) if self.alpha > 0. else 1.
        batch_size = X.shape[0]
        index = torch.randperm(batch_size).to(self.device)

        mixed_x = lam * X + (1 - lam) * X[index, :]
        y_a, y_b = y, y[index]
        return mixed_x, _NewLossParameters(y_a=y_a, y_b=y_b, lam=lam)

    def _get_new_loss_fn(
        self,
        new_loss_params: _NewLossParameters
    ) -> Callable:

        y_a = new_loss_params.y_a
        y_b = new_loss_params.y_b
        lam = new_loss_params.lam

        if lam > 1.0 or lam < 0:
            raise ValueError("The mixup coefficient `lam` must be [0, 1], but got {:.2f}.".format(lam))

        return lambda criterion, pred: lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

    @staticmethod
    def get_properties(dataset_properties: Optional[Dict[str, Any]] = None
                       ) -> Dict[str, str]:
        return {
            'shortname': 'MixUpTrainer',
            'name': 'MixUp Regularized Trainer',
        }

    @staticmethod
    def get_hyperparameter_search_space(
        dataset_properties: Optional[Dict] = None,
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
            if STRING_TO_TASK_TYPES[dataset_properties['task_type']] in CLASSIFICATION_TASKS:
                add_hyperparameter(cs, weighted_loss, CategoricalHyperparameter)
        return cs
