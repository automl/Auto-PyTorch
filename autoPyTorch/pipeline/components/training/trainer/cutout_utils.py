import typing

from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import (
    CategoricalHyperparameter,
    UniformFloatHyperparameter,
)

import numpy as np

from autoPyTorch.constants import CLASSIFICATION_TASKS, STRING_TO_TASK_TYPES


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
                                        patch_ratio: typing.Tuple[typing.Tuple[float, float], float] = ((0.0, 1.0), 0.2),
                                        cutout_prob: typing.Tuple[typing.Tuple[float, float], float] = ((0.0, 1.0), 0.2),
                                        weighted_loss: typing.Tuple[typing.Tuple, bool] = ((True, False), True)
                                        ) -> ConfigurationSpace:
        patch_ratio = UniformFloatHyperparameter(
            "patch_ratio", patch_ratio[0][0], patch_ratio[0][1], default_value=patch_ratio[1])
        cutout_prob = UniformFloatHyperparameter(
            "cutout_prob", cutout_prob[0][0], cutout_prob[0][1], default_value=cutout_prob[1])
        weighted_loss = CategoricalHyperparameter("weighted_loss", choices=weighted_loss[0],
                                                  default_value=weighted_loss[1])
        cs = ConfigurationSpace()
        cs.add_hyperparameters([patch_ratio, cutout_prob])
        if dataset_properties is not None:
            if STRING_TO_TASK_TYPES[dataset_properties['task_type']] not in CLASSIFICATION_TASKS:
                cs.add_hyperparameters([weighted_loss])
        return cs
