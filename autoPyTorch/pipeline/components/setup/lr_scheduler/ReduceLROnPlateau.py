from typing import Any, Dict, Optional, Tuple, Union

from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import (
    CategoricalHyperparameter,
    UniformFloatHyperparameter,
    UniformIntegerHyperparameter
)

import numpy as np

import torch.optim.lr_scheduler
from torch.optim.lr_scheduler import _LRScheduler

from autoPyTorch.pipeline.components.setup.lr_scheduler.base_scheduler import BaseLRComponent


class ReduceLROnPlateau(BaseLRComponent):
    """
    Reduce learning rate when a metric has stopped improving. Models often benefit from
    reducing the learning rate by a factor of 2-10 once learning stagnates. This scheduler
    reads a metrics quantity and if no improvement is seen for a ‘patience’ number of epochs,
    the learning rate is reduced.

    Args:
        mode (str): One of min, max. In min mode, lr will be reduced when the quantity
            monitored has stopped decreasing; in max mode it will be reduced when
            the quantity monitored has stopped increasing
        factor (float): Factor by which the learning rate will be reduced. new_lr = lr * factor.
        patience (int): Number of epochs with no improvement after which learning
            rate will be reduced.
        random_state (Optional[np.random.RandomState]): random state
    """
    def __init__(
        self,
        mode: str,
        factor: float,
        patience: int,
        random_state: Optional[np.random.RandomState] = None
    ):

        super().__init__()
        self.mode = mode
        self.factor = factor
        self.patience = patience
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

        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=X['optimizer'],
            mode=self.mode,
            factor=float(self.factor),
            patience=int(self.patience),
        )
        return self

    @staticmethod
    def get_properties(dataset_properties: Optional[Dict[str, Any]] = None) -> Dict[str, Union[str, bool]]:
        return {
            'shortname': 'ReduceLROnPlateau',
            'name': 'ReduceLROnPlateau',
            'cyclic': False
        }

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties: Optional[Dict] = None,
                                        mode: Tuple[Tuple, str] = (('min', 'max'), 'min'),
                                        patience: Tuple[Tuple, int] = ((5, 20), 10),
                                        factor: Tuple[Tuple[float, float], float] = ((0.01, 0.9), 0.1)
                                        ) -> ConfigurationSpace:
        mode = CategoricalHyperparameter('mode', choices=mode[0], default_value=mode[1])
        patience = UniformIntegerHyperparameter(
            "patience", patience[0][0], patience[0][1], default_value=patience[1])
        factor = UniformFloatHyperparameter(
            "factor", factor[0][0], factor[0][1], default_value=factor[1])
        cs = ConfigurationSpace()
        cs.add_hyperparameters([mode, patience, factor])
        return cs
