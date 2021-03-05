from typing import Any, Dict, Optional, Tuple, Union

from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformIntegerHyperparameter

import numpy as np

import torch.optim.lr_scheduler
from torch.optim.lr_scheduler import _LRScheduler

from autoPyTorch.pipeline.components.setup.lr_scheduler.base_scheduler import BaseLRComponent
from autoPyTorch.utils.common import HyperparameterSearchSpace, add_hyperparameter


class CosineAnnealingWarmRestarts(BaseLRComponent):
    """
    Set the learning rate of each parameter group using a cosine annealing schedule,
    where \eta_{max}ηmax is set to the initial lr, T_{cur} is the number of epochs
    since the last restart and T_{i} is the number of epochs between two warm
    restarts in SGDR

    Args:
        n_restarts (int): Number of restarts. In autopytorch, based
            on the total budget(epochs) there are 'n_restarts'
            restarts made periodically.
        random_state (Optional[np.random.RandomState]): random state
    """

    def __init__(
        self,
        n_restarts: int,
        random_state: Optional[np.random.RandomState] = None
    ):
        super().__init__()
        self.n_restarts = n_restarts
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

        # initialise required attributes for the scheduler
        T_mult: int = 1
        T_0: int = max(X['epochs'] // self.n_restarts, 1)

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer=X['optimizer'],
            T_0=T_0,
            T_mult=T_mult,
        )
        return self

    @staticmethod
    def get_properties(dataset_properties: Optional[Dict[str, Any]] = None) -> Dict[str, Union[str, bool]]:
        return {
            'shortname': 'CosineAnnealingWarmRestarts',
            'name': 'Cosine Annealing WarmRestarts',
            'cyclic': True
        }

    @staticmethod
    def get_hyperparameter_search_space(
        dataset_properties: Optional[Dict] = None,
        n_restarts: HyperparameterSearchSpace = HyperparameterSearchSpace(hyperparameter='n_restarts',
                                                                          value_range=(1, 6),
                                                                          default_value=3,
                                                                          ),
    ) -> ConfigurationSpace:
        cs = ConfigurationSpace()
        add_hyperparameter(cs, n_restarts, UniformIntegerHyperparameter)

        return cs
