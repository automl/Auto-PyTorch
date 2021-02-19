import typing

from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import (
    CategoricalHyperparameter,
    UniformFloatHyperparameter,
)

import numpy as np

from autoPyTorch.constants import CLASSIFICATION_TASKS, STRING_TO_TASK_TYPES
from autoPyTorch.utils.common import HyperparameterSearchSpace, add_hyperparameter


class CutOut:
    def __init__(self, patch_ratio: float,
                 cutout_prob: float,
                 weighted_loss: bool = False,
                 random_state: typing.Optional[np.random.RandomState] = None):
        """
        This class handles the training of a network for a single given epoch.

        Args:
            patch_ratio (float): Defines the size of the cut off
            cutout_prob (float): The probability of occurrence of this regulatization

        """
        super().__init__(random_state=random_state)
        self.weighted_loss = weighted_loss
        self.patch_ratio = patch_ratio
        self.cutout_prob = cutout_prob

    def criterion_preparation(self, y_a: np.ndarray, y_b: np.ndarray = None, lam: float = 1.0
                              ) -> typing.Callable:
        return lambda criterion, pred: lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties: typing.Optional[typing.Dict] = None,
                                        weighted_loss: HyperparameterSearchSpace = HyperparameterSearchSpace(
                                            hyperparameter="weighted_loss",
                                            value_range=(True, False),
                                            default_value=True),
                                        patch_ratio: HyperparameterSearchSpace = HyperparameterSearchSpace(
                                            hyperparameter="alpha",
                                            value_range=(0, 1),
                                            default_value=0.2),
                                        cutout_prob: HyperparameterSearchSpace = HyperparameterSearchSpace(
                                            hyperparameter="alpha",
                                            value_range=(0, 1),
                                            default_value=0.2),
                                        ) -> ConfigurationSpace:


        cs = ConfigurationSpace()

        add_hyperparameter(cs, patch_ratio, UniformFloatHyperparameter)
        add_hyperparameter(cs, cutout_prob, UniformFloatHyperparameter)
        if dataset_properties is not None:
            if STRING_TO_TASK_TYPES[dataset_properties['task_type']] in CLASSIFICATION_TASKS:
                add_hyperparameter(cs, weighted_loss, CategoricalHyperparameter)

        return cs
