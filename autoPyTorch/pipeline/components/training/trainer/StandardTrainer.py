from typing import Any, Callable, Dict, Optional, Tuple

from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import CategoricalHyperparameter

import numpy as np

import torch

from autoPyTorch.constants import CLASSIFICATION_TASKS, STRING_TO_TASK_TYPES
from autoPyTorch.pipeline.components.training.trainer.base_trainer import (
    BaseTrainerComponent,
    _NewLossParameters
)
from autoPyTorch.utils.common import HyperparameterSearchSpace, add_hyperparameter


class StandardTrainer(BaseTrainerComponent):
    def __init__(self, weighted_loss: bool = False,
                 random_state: Optional[np.random.RandomState] = None):
        """
        This class handles the training of a network for a single given epoch.

        Args:
            weighted_loss (bool): whether to use weighted loss

        """
        super().__init__(random_state=random_state)
        self.weighted_loss = weighted_loss

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
        return X, _NewLossParameters(y_a=y)

    def _get_new_loss_fn(
        self,
        new_loss_params: _NewLossParameters
    ) -> Callable:
        y_a = new_loss_params.y_a
        return lambda criterion, pred: criterion(pred, y_a)

    @staticmethod
    def get_properties(dataset_properties: Optional[Dict[str, Any]] = None
                       ) -> Dict[str, str]:
        return {
            'shortname': 'StandardTrainer',
            'name': 'StandardTrainer',
        }

    @staticmethod
    def get_hyperparameter_search_space(
        dataset_properties: Optional[Dict] = None,
        weighted_loss: HyperparameterSearchSpace = HyperparameterSearchSpace(hyperparameter="weighted_loss",
                                                                             value_range=(True, False),
                                                                             default_value=True),
    ) -> ConfigurationSpace:
        cs = ConfigurationSpace()
        if dataset_properties is not None:
            if STRING_TO_TASK_TYPES[dataset_properties['task_type']] in CLASSIFICATION_TASKS:
                add_hyperparameter(cs, weighted_loss, CategoricalHyperparameter)

        return cs
