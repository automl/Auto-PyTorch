from typing import Any, Dict, Optional, Tuple, Union

from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import (
    UniformFloatHyperparameter,
    UniformIntegerHyperparameter,
)

import numpy as np

import torch.optim.lr_scheduler
from torch.optim.lr_scheduler import _LRScheduler

from autoPyTorch.pipeline.components.setup.lr_scheduler.base_scheduler import BaseLRComponent


class StepLR(BaseLRComponent):
    """
    Decays the learning rate of each parameter group by gamma every step_size epochs.
    Notice that such decay can happen simultaneously with other changes to the learning
    rate from outside this scheduler. When last_epoch=-1, sets initial lr as lr.

    Args:
        step_size (int) – Period of learning rate decay.
        gamma (float) – Multiplicative factor of learning rate decay. Default: 0.1.

    """
    def __init__(
        self,
        step_size: int,
        gamma: float,
        random_state: Optional[np.random.RandomState] = None
    ):

        super().__init__()
        self.gamma = gamma
        self.step_size = step_size
        self.random_state = random_state
        self.scheduler = None  # type: Optional[_LRScheduler]

    def fit(self, X: Dict[str, Any], y: Any = None) -> BaseLRComponent:
        """
        Fits a component by using an input dictionary with pre-requisites

        Args:
            X (X: Dict[str, Any]): Dependencies needed by current component to perform fit
            y (Any): not used. To comply with sklearn API

        Returns:
            A instance of self
        """

        # Make sure there is an optimizer
        self.check_requirements(X, y)

        self.scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer=X['optimizer'],
            step_size=int(self.step_size),
            gamma=float(self.gamma),
        )
        return self

    @staticmethod
    def get_properties(dataset_properties: Optional[Dict[str, Any]] = None) -> Dict[str, Union[str, bool]]:
        return {
            'shortname': 'StepLR',
            'name': 'StepLR',
            'cyclic': False
        }

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties: Optional[Dict] = None,
                                        gamma: Tuple[Tuple[float, float], float] = ((0.001, 0.9), 0.1),
                                        step_size: Tuple[Tuple[int, int], int] = ((1, 10), 5)
                                        ) -> ConfigurationSpace:
        gamma = UniformFloatHyperparameter(
            "gamma", gamma[0][0], gamma[0][1], default_value=gamma[1])
        step_size = UniformIntegerHyperparameter(
            "step_size", step_size[0][0], step_size[0][1], default_value=step_size[1])
        cs = ConfigurationSpace()
        cs.add_hyperparameters([gamma, step_size])
        return cs
