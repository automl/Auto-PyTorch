from typing import Any, Dict, Optional, Tuple, Union

from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import (
    CategoricalHyperparameter,
    UniformFloatHyperparameter,
    UniformIntegerHyperparameter,
)

import numpy as np

import torch.optim.lr_scheduler
from torch.optim.lr_scheduler import _LRScheduler

from autoPyTorch.pipeline.components.setup.lr_scheduler.base_scheduler import BaseLRComponent


class CyclicLR(BaseLRComponent):
    """
    Sets the learning rate of each parameter group according to cyclical learning rate
    policy (CLR). The policy cycles the learning rate between two boundaries with a
    constant frequency.

    Args:
        base_lr (float): Initial learning rate which is the lower boundary in the
            cycle for each parameter group.
        mode (str): policy for the cycle
        step_size_up (int): Number of training iterations in the increasing half of a cycle.
        max_lr (float): Upper learning rate boundaries in the cycle for each parameter group.
            In this implementation, to make sure max_lr>base_lr, max_lr is the increment from
            base_lr. This simplifies the learning space

    """
    def __init__(
        self,
        base_lr: float,
        mode: str,
        step_size_up: int,
        max_lr: float = 0.1,
        random_state: Optional[np.random.RandomState] = None
    ):

        super().__init__()
        self.base_lr = base_lr
        self.mode = mode
        self.max_lr = max_lr
        self.step_size_up = step_size_up
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

        # No momentum to cycle in adam
        cycle_momentum = True
        if 'Adam' in X['optimizer'].__class__.__name__:
            cycle_momentum = False

        self.scheduler = torch.optim.lr_scheduler.CyclicLR(
            optimizer=X['optimizer'],
            base_lr=float(self.base_lr),
            max_lr=float(self.max_lr),
            step_size_up=int(self.step_size_up),
            cycle_momentum=cycle_momentum,
            mode=self.mode,
        )
        return self

    @staticmethod
    def get_properties(dataset_properties: Optional[Dict[str, Any]] = None) -> Dict[str, Union[str, bool]]:
        return {
            'shortname': 'CyclicLR',
            'name': 'CyclicLR',
            'cyclic': True
        }

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties: Optional[Dict] = None,
                                        base_lr: Tuple[Tuple[float, float], float] = ((1e-6, 1e-1), 0.01),
                                        mode: Tuple[Tuple, str] = (('triangular', 'triangular2', 'exp_range'),
                                                                   'triangular'),
                                        step_size_up: Tuple[Tuple[int, int], int] = ((1000, 4000), 2000),
                                        max_lr: Tuple[Tuple[float, float], float] = ((1e-3, 1e-1), 0.1)
                                        ) -> ConfigurationSpace:
        base_lr = UniformFloatHyperparameter(
            "base_lr", base_lr[0][0], base_lr[0][1], default_value=base_lr[1])
        mode = CategoricalHyperparameter('mode', choices=mode[0], default_value=mode[1])
        step_size_up = UniformIntegerHyperparameter(
            "step_size_up", step_size_up[0][0], step_size_up[0][1], default_value=step_size_up[1])
        max_lr = UniformFloatHyperparameter(
            "max_lr", max_lr[0][0], max_lr[0][1], default_value=max_lr[1])
        cs = ConfigurationSpace()
        cs.add_hyperparameters([base_lr, mode, step_size_up, max_lr])
        return cs
