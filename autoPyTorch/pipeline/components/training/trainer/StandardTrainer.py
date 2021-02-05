import typing

from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import CategoricalHyperparameter

import numpy as np

from autoPyTorch.constants import CLASSIFICATION_TASKS, STRING_TO_TASK_TYPES
from autoPyTorch.pipeline.components.training.trainer.base_trainer import BaseTrainerComponent


class StandardTrainer(BaseTrainerComponent):
    def __init__(self, weighted_loss: bool = False,
                 random_state: typing.Optional[np.random.RandomState] = None):
        """
        This class handles the training of a network for a single given epoch.

        Args:
            weighted_loss (bool): whether to use weighted loss

        """
        super().__init__(random_state=random_state)
        self.weighted_loss = weighted_loss

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
        return X, {'y_a': y}

    def criterion_preparation(self, y_a: np.ndarray, y_b: np.ndarray = None, lam: float = 1.0
                              ) -> typing.Callable:
        return lambda criterion, pred: criterion(pred, y_a)

    @staticmethod
    def get_properties(dataset_properties: typing.Optional[typing.Dict[str, typing.Any]] = None
                       ) -> typing.Dict[str, str]:
        return {
            'shortname': 'StandardTrainer',
            'name': 'StandardTrainer',
        }

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties: typing.Optional[typing.Dict] = None,
                                        weighted_loss: typing.Tuple[typing.Tuple, bool] = ((True, False), True)
                                        ) -> ConfigurationSpace:
        weighted_loss = CategoricalHyperparameter("weighted_loss", choices=weighted_loss[0],
                                                  default_value=weighted_loss[1])
        cs = ConfigurationSpace()
        if dataset_properties is not None:
            if STRING_TO_TASK_TYPES[dataset_properties['task_type']] not in CLASSIFICATION_TASKS:
                cs.add_hyperparameters([weighted_loss])

        return cs
