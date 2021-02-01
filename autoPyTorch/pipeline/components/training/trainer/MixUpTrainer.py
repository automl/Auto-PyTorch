import typing

from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import (
    UniformFloatHyperparameter,
)

import numpy as np

import torch

from autoPyTorch.pipeline.components.training.trainer.base_trainer import BaseTrainerComponent


class MixUpTrainer(BaseTrainerComponent):
    def __init__(self, alpha: float, random_state: typing.Optional[np.random.RandomState] = None):
        """
        This class handles the training of a network for a single given epoch.

        Args:
            alpha (float): the mixup ratio

        """
        super().__init__(random_state=random_state)
        self.alpha = alpha

    def data_preparation(self, X: np.ndarray, y: np.ndarray,
                         ) -> typing.Tuple[np.ndarray, typing.Dict[str, np.ndarray]]:
        """
        Depending on the trainer choice, data fed to the network might be pre-processed
        on a different way. That is, in standard training we provide the data to the
        network as we receive it to the loader. Some regularization techniques, like mixup
        alter the data.

        Args:
            X (np.ndarray): The batch training features
            y (np.ndarray): The batch training labels

        Returns:
            np.ndarray: that processes data
            typing.Dict[str, np.ndarray]: arguments to the criterion function
        """
        lam = np.random.beta(self.alpha, self.alpha) if self.alpha > 0. else 1.
        batch_size = X.size()[0]
        index = torch.randperm(batch_size).cuda() if X.is_cuda else torch.randperm(batch_size)

        mixed_x = lam * X + (1 - lam) * X[index, :]
        y_a, y_b = y, y[index]
        return mixed_x, {'y_a': y_a, 'y_b': y_b, 'lam': lam}

    def criterion_preparation(self, y_a: np.ndarray, y_b: np.ndarray = None, lam: float = 1.0
                              ) -> typing.Callable:
        return lambda criterion, pred: lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

    @staticmethod
    def get_properties(dataset_properties: typing.Optional[typing.Dict[str, typing.Any]] = None
                       ) -> typing.Dict[str, str]:
        return {
            'shortname': 'MixUpTrainer',
            'name': 'MixUp Regularized Trainer',
        }

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties: typing.Optional[typing.Dict] = None,
                                        alpha: typing.Tuple[typing.Tuple[float, float], float] = ((0, 1), 0.2)
                                        ) -> ConfigurationSpace:
        alpha = UniformFloatHyperparameter(
            "alpha", alpha[0][0], alpha[0][1], default_value=alpha[1])
        cs = ConfigurationSpace()
        cs.add_hyperparameters([alpha])
        return cs
