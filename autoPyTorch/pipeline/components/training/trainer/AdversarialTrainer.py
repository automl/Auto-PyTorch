from copy import deepcopy
from typing import Any, Callable, Dict, Optional, Tuple, Union

from ConfigSpace.conditions import EqualsCondition
from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import (
    CategoricalHyperparameter,
    Constant,
    UniformFloatHyperparameter,
)

import numpy as np

import torch


from autoPyTorch.constants import CLASSIFICATION_TASKS, STRING_TO_TASK_TYPES
from autoPyTorch.pipeline.components.training.trainer.base_trainer import BaseTrainerComponent
from autoPyTorch.pipeline.components.training.trainer.utils import Lookahead
from autoPyTorch.utils.common import HyperparameterSearchSpace, add_hyperparameter, get_hyperparameter


class AdversarialTrainer(BaseTrainerComponent):
    def __init__(
            self,
            epsilon: float,
            weighted_loss: bool = False,
            random_state: Optional[np.random.RandomState] = None,
            use_stochastic_weight_averaging: bool = False,
            use_snapshot_ensemble: bool = False,
            se_lastk: int = 3,
            use_lookahead_optimizer: bool = True,
            **lookahead_config: Any
    ):
        """
        This class handles the training of a network for a single given epoch.

        Args:
            epsilon (float): The perturbation magnitude.
        
        References:
            Explaining and Harnessing Adversarial Examples
            Ian J. Goodfellow et. al.
            https://arxiv.org/pdf/1412.6572.pdf
        """
        super().__init__(random_state=random_state,
                         weighted_loss=weighted_loss,
                         use_stochastic_weight_averaging=use_stochastic_weight_averaging,
                         use_snapshot_ensemble=use_snapshot_ensemble,
                         se_lastk=se_lastk,
                         use_lookahead_optimizer=use_lookahead_optimizer,
                         **lookahead_config)
        self.epsilon = epsilon

    def data_preparation(self, X: np.ndarray, y: np.ndarray,
                         ) -> Tuple[Tuple[np.ndarray, np.ndarray], Dict[str, np.ndarray]]:
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
                              ) -> Callable:
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

        original_data = torch.autograd.Variable(original_data)
        adversarial_data = torch.autograd.Variable(adversarial_data)

        # training
        self.optimizer.zero_grad()
        original_outputs = self.model(original_data)
        adversarial_outputs = self.model(adversarial_data)

        loss_func = self.criterion_preparation(**criterion_kwargs)
        loss = loss_func(self.criterion, original_outputs, adversarial_outputs)
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
            data: np.ndarray,
            targets: np.ndarray,
    ) -> np.ndarray:
        """
        Generates the adversarial examples.

        Args:
            data (np.ndarray): input features to the network
            targets (np.ndarray): ground truth to calculate loss

        Returns:
            adv_data (np.ndarray): the adversarial examples.
        
        References:
            https://pytorch.org/tutorials/beginner/fgsm_tutorial.html#fgsm-attack
        """
        data_copy = deepcopy(data)
        data_copy = data_copy.float().to(self.device)
        targets = targets.long().to(self.device)
        data_copy = torch.autograd.Variable(data_copy)
        data_copy.requires_grad = True

        outputs = self.model(data_copy)
        cost = self.criterion(outputs, targets)

        grad = torch.autograd.grad(cost, data_copy, retain_graph=False, create_graph=False)[0]

        adv_data = data_copy + self.epsilon * grad.sign()
        adv_data = torch.clamp(adv_data, min=0, max=1).detach()

        return adv_data

    @staticmethod
    def get_properties(dataset_properties: Optional[Dict[str, Any]] = None
                       ) -> Dict[str, Union[str, bool]]:

        return {
            'shortname': 'AdversarialTrainer',
            'name': 'AdversarialTrainer',
            'handles_tabular': True,
            'handles_image': False,
            'handles_time_series': False,
        }

    @staticmethod
    def get_hyperparameter_search_space(
        dataset_properties: Optional[Dict] = None,
        weighted_loss: HyperparameterSearchSpace = HyperparameterSearchSpace(
            hyperparameter="weighted_loss",
            value_range=(True, False),
            default_value=True),
        la_steps: HyperparameterSearchSpace = HyperparameterSearchSpace(
            hyperparameter="la_steps",
            value_range=(5, 10),
            default_value=6,
            log=False),
        la_alpha: HyperparameterSearchSpace = HyperparameterSearchSpace(
            hyperparameter="la_alpha",
            value_range=(0.5, 0.8),
            default_value=0.6,
            log=False),
        use_lookahead_optimizer: HyperparameterSearchSpace = HyperparameterSearchSpace(
            hyperparameter="use_lookahead_optimizer",
            value_range=(True, False),
            default_value=True),
        use_stochastic_weight_averaging: HyperparameterSearchSpace = HyperparameterSearchSpace(
            hyperparameter="use_stochastic_weight_averaging",
            value_range=(True, False),
            default_value=True),
        use_snapshot_ensemble: HyperparameterSearchSpace = HyperparameterSearchSpace(
            hyperparameter="use_snapshot_ensemble",
            value_range=(True, False),
            default_value=True),
        se_lastk: HyperparameterSearchSpace = HyperparameterSearchSpace(
            hyperparameter="se_lastk",
            value_range=(3,),
            default_value=3),
        epsilon: HyperparameterSearchSpace = HyperparameterSearchSpace(
            hyperparameter="epsilon",
            value_range=(0.05, 0.2),
            default_value=0.2),
    ) -> ConfigurationSpace:
        cs = ConfigurationSpace()

        add_hyperparameter(cs, epsilon, UniformFloatHyperparameter)
        add_hyperparameter(cs, use_stochastic_weight_averaging, CategoricalHyperparameter)
        snapshot_ensemble_flag = any(use_snapshot_ensemble.value_range)

        use_snapshot_ensemble = get_hyperparameter(use_snapshot_ensemble, CategoricalHyperparameter)
        cs.add_hyperparameter(use_snapshot_ensemble)

        if snapshot_ensemble_flag:
            se_lastk = get_hyperparameter(se_lastk, Constant)
            cs.add_hyperparameter(se_lastk)
            cond = EqualsCondition(se_lastk, use_snapshot_ensemble, True)
            cs.add_condition(cond)

        lookahead_flag = any(use_lookahead_optimizer.value_range)

        use_lookahead_optimizer = get_hyperparameter(use_lookahead_optimizer, CategoricalHyperparameter)
        cs.add_hyperparameter(use_lookahead_optimizer)

        if lookahead_flag:
            la_config_space = Lookahead.get_hyperparameter_search_space(la_steps=la_steps,
                                                                        la_alpha=la_alpha)
            parent_hyperparameter = {'parent': use_lookahead_optimizer, 'value': True}
            cs.add_configuration_space(
                Lookahead.__name__,
                la_config_space,
                parent_hyperparameter=parent_hyperparameter
            )

        # TODO, decouple the weighted loss from the trainer
        if dataset_properties is not None:
            if STRING_TO_TASK_TYPES[dataset_properties['task_type']] in CLASSIFICATION_TASKS:
                add_hyperparameter(cs, weighted_loss, CategoricalHyperparameter)

        return cs
