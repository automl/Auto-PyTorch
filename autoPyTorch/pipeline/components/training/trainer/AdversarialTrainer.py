import typing
from copy import deepcopy

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


class AdversarialTrainer(BaseTrainerComponent):
    def __init__(
            self,
            epsilon: float,
            weighted_loss: bool = False,
            random_state: typing.Optional[np.random.RandomState] = None,
            use_stochastic_weight_averaging: bool = False,
            use_snapshot_ensemble: bool = False,
            se_lastk: int = 3,
            use_lookahead_optimizer: bool = True,
            **lookahead_config: typing.Any
    ):
        """
        This class handles the training of a network for a single given epoch.

        Args:
            epsilon (float): The perturbation magnitude.

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

        original_data = torch.autograd.Variable(original_data)
        adversarial_data = torch.autograd.Variable(adversarial_data)

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
    def get_properties(dataset_properties: typing.Optional[typing.Dict[str, typing.Any]] = None
                       ) -> typing.Dict[str, typing.Union[str, bool]]:

        return {
            'shortname': 'AdversarialTrainer',
            'name': 'AdversarialTrainer',
            'handles_tabular': True,
            'handles_image': False,
            'handles_time_series': False,
        }

    @staticmethod
    def get_hyperparameter_search_space(
        dataset_properties: typing.Optional[typing.Dict] = None,
        weighted_loss: typing.Tuple[typing.Tuple, bool] = ((True, False), True),
        use_stochastic_weight_averaging: typing.Tuple[typing.Tuple, bool] = ((True, False), True),
        use_snapshot_ensemble: typing.Tuple[typing.Tuple, bool] = ((True, False), True),
        se_lastk: typing.Tuple[typing.Tuple, int] = ((3,), 3),
        use_lookahead_optimizer: typing.Tuple[typing.Tuple, bool] = ((True, False), True),
        la_steps: typing.Tuple[typing.Tuple, int, bool] = ((5, 10), 6, False),
        la_alpha: typing.Tuple[typing.Tuple, float, bool] = ((0.5, 0.8), 0.6, False),
        epsilon: typing.Tuple[typing.Tuple[float, float], float] = ((0.05, 0.2), 0.2),
    ) -> ConfigurationSpace:
        epsilon = UniformFloatHyperparameter(
            "epsilon", epsilon[0][0], epsilon[0][1], default_value=epsilon[1])
        weighted_loss = CategoricalHyperparameter("weighted_loss", choices=weighted_loss[0],
                                                  default_value=weighted_loss[1])

        use_swa = CategoricalHyperparameter("use_stochastic_weight_averaging",
                                            choices=use_stochastic_weight_averaging[0],
                                            default_value=use_stochastic_weight_averaging[1])
        use_se = CategoricalHyperparameter("use_snapshot_ensemble",
                                           choices=use_snapshot_ensemble[0],
                                           default_value=use_snapshot_ensemble[1])

        # Note, this is not easy to be considered as a hyperparameter.
        # When used with cyclic learning rates, it depends on the number
        # of restarts.
        se_lastk = Constant('se_lastk', se_lastk[1])

        use_lookahead_optimizer = CategoricalHyperparameter("use_lookahead_optimizer",
                                                            choices=use_lookahead_optimizer[0],
                                                            default_value=use_lookahead_optimizer[1])

        config_space = Lookahead.get_hyperparameter_search_space(la_steps=la_steps,
                                                                 la_alpha=la_alpha)
        parent_hyperparameter = {'parent': use_lookahead_optimizer, 'value': True}

        cs = ConfigurationSpace()
        cs.add_hyperparameters([use_swa, use_se, se_lastk, use_lookahead_optimizer])
        cs.add_configuration_space(
            Lookahead.__name__,
            config_space,
            parent_hyperparameter=parent_hyperparameter
        )
        cond = EqualsCondition(se_lastk, use_se, True)
        cs.add_condition(cond)

        cs.add_hyperparameters([epsilon])
        if dataset_properties is not None:
            if STRING_TO_TASK_TYPES[dataset_properties['task_type']] not in CLASSIFICATION_TASKS:
                cs.add_hyperparameters([weighted_loss])
        return cs
