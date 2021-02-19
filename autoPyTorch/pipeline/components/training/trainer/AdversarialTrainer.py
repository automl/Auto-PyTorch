from copy import deepcopy
import typing
from typing import Dict, Optional, Tuple

from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import (
    CategoricalHyperparameter,
    UniformFloatHyperparameter,
)

import numpy as np

import torch
from torch.autograd import Variable
from torch.utils.tensorboard.writer import SummaryWriter

from autoPyTorch.constants import CLASSIFICATION_TASKS, STRING_TO_TASK_TYPES
from autoPyTorch.pipeline.components.training.trainer.base_trainer import BaseTrainerComponent
from autoPyTorch.utils.logging_ import PicklableClientLogger


class AdversarialTrainer(BaseTrainerComponent):
    def __init__(
            self,
            epsilon: float,
            weighted_loss: bool = False,
            random_state: typing.Optional[np.random.RandomState] = None,
    ):
        """
        This class handles the training of a network for a single given epoch.

        Args:
            epsilon (float): The perturbation magnitude.

        """
        super().__init__(random_state=random_state)
        self.epsilon = epsilon
        self.weighted_loss = weighted_loss

    def data_preparation(self, X: np.ndarray, y: np.ndarray,
                         ) -> typing.Tuple[typing.Tuple[np.ndarray, np.ndarray], typing.Dict[str, np.ndarray]]:
        """Generate adversarial examples from the original inputs.

        Args:
            X (np.ndarray): The batch training features
            y (np.ndarray): The batch training labels

        Returns:
            typing.Tuple[np.ndarray, np.ndarray]: original examples, adversarial examples.
            typing.Dict[str, np.ndarray]: arguments to the criterion function.
        """
        X_adversarial = self.fgsm_attack(X, y)
        return (X, X_adversarial), {'y_a': y}

    def criterion_preparation(self, y_a: np.ndarray, y_b: np.ndarray = None, lam: float = 1.0
                              ) -> typing.Callable:
        # Initial implementation, consider the adversarial loss and the normal network loss
        # equally.
        return lambda criterion, pred, adversarial_pred: 0.5 * criterion(pred, y_a) + \
                                                         0.5 * criterion(adversarial_pred, y_a)

    def train_step(self, data: np.ndarray, targets: np.ndarray) -> Tuple[float, torch.Tensor]:
        """
        Allows to train 1 step of gradient descent, given a batch of train/labels

        Args:
            data (np.ndarray): input features to the network
            targets (np.ndarray): ground truth to calculate loss

        Returns:
            torch.Tensor: The predictions of the network
            float: the loss incurred in the prediction
        """
        # prepare
        data = data.float().to(self.device)
        targets = targets.long().to(self.device)

        data, criterion_kwargs = self.data_preparation(data, targets)
        original_data = data[0]
        adversarial_data = data[1]

        original_data = Variable(original_data)
        adversarial_data = Variable(adversarial_data)

        # training
        self.optimizer.zero_grad()
        original_outputs = self.model(original_data)
        adversarial_output = self.model(adversarial_data)

        loss_func = self.criterion_preparation(**criterion_kwargs)
        loss = loss_func(self.criterion, original_outputs, adversarial_output)
        loss.backward()
        self.optimizer.step()
        if self.scheduler:
            if 'ReduceLROnPlateau' in self.scheduler.__class__.__name__:
                self.scheduler.step(loss)
            else:
                self.scheduler.step()
        # only passing the original outputs since we do not care about
        # the adversarial performance.
        return loss.item(), original_outputs

    def fgsm_attack(
            self,
            data,
            targets,
    ) -> np.ndarray:
        """
        Generates the adversarial examples.

        Args:
            data (np.ndarray): input features to the network
            targets (np.ndarray): ground truth to calculate loss.
            eps: (float): magnitude of perturbation.

        Returns:
            adv_data (np.ndarray): the adversarial examples.
        """
        data_copy = deepcopy(data)
        data_copy = data_copy.float().to(self.device)
        targets = targets.long().to(self.device)
        data_copy = Variable(data_copy)
        data_copy.requires_grad = True

        outputs = self.model(data_copy)
        cost = self.criterion(outputs, targets)

        grad = torch.autograd.grad(cost, data_copy, retain_graph=False, create_graph=False)[0]

        adv_data = data_copy + self.epsilon * grad.sign()
        adv_data = torch.clamp(adv_data, min=0, max=1).detach()

        return adv_data

    @staticmethod
    def get_properties(dataset_properties: typing.Optional[typing.Dict[str, typing.Any]] = None
                       ) -> typing.Dict[str, str]:
        return {
            'shortname': 'AdversarialTrainer',
            'name': 'FGSM Adversarial Regularized Trainer',
        }

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties: typing.Optional[typing.Dict] = None,
                                        epsilon: typing.Tuple[typing.Tuple[float, float], float] = ((0.05, 0.2), 0.2),
                                        weighted_loss: typing.Tuple[typing.Tuple, bool] = ((True, False), True)
                                        ) -> ConfigurationSpace:
        epsilon = UniformFloatHyperparameter(
            "epsilon", epsilon[0][0], epsilon[0][1], default_value=epsilon[1])
        weighted_loss = CategoricalHyperparameter("weighted_loss", choices=weighted_loss[0],
                                                  default_value=weighted_loss[1])
        cs = ConfigurationSpace()
        cs.add_hyperparameters([epsilon])
        if dataset_properties is not None:
            if STRING_TO_TASK_TYPES[dataset_properties['task_type']] not in CLASSIFICATION_TASKS:
                cs.add_hyperparameters([weighted_loss])
        return cs
