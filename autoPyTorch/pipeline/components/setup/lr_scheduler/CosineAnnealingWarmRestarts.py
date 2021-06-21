from typing import Any, Dict, Optional, Union

from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import (
    UniformFloatHyperparameter,
    UniformIntegerHyperparameter
)

import numpy as np

import torch.optim.lr_scheduler

from autoPyTorch.datasets.base_dataset import BaseDatasetPropertiesType
from autoPyTorch.pipeline.components.setup.lr_scheduler.base_scheduler import BaseLRComponent
from autoPyTorch.pipeline.components.setup.lr_scheduler.constants import StepIntervalUnit
from autoPyTorch.utils.common import HyperparameterSearchSpace, add_hyperparameter


class CosineAnnealingWarmRestarts(BaseLRComponent):
    r"""
    Set the learning rate of each parameter group using a cosine annealing schedule,
    where \eta_{max}ηmax is set to the initial lr, T_{cur} is the number of epochs
    since the last restart and T_{i} is the number of epochs between two warm
    restarts in SGDR

    Args:
        T_0 (int): Number of iterations for the first restart
        T_mult (int):  A factor increases T_{i} after a restart
        random_state (Optional[np.random.RandomState]): random state
    """

    def __init__(
        self,
        T_0: int,
        T_mult: int,
        step_interval: Union[str, StepIntervalUnit] = StepIntervalUnit.epoch,
        random_state: Optional[np.random.RandomState] = None,
    ):
        super().__init__(step_interval)
        self.T_0 = T_0
        self.T_mult = T_mult
        self.random_state = random_state

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

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer=X['optimizer'],
            T_0=int(self.T_0),
            T_mult=int(self.T_mult),
        )
        return self

    @staticmethod
    def get_properties(dataset_properties: Optional[Dict[str, BaseDatasetPropertiesType]] = None
                       ) -> Dict[str, Union[str, bool]]:
        return {
            'shortname': 'CosineAnnealingWarmRestarts',
            'name': 'Cosine Annealing WarmRestarts',
        }

    @staticmethod
    def get_hyperparameter_search_space(
        dataset_properties: Optional[Dict[str, BaseDatasetPropertiesType]] = None,
        T_0: HyperparameterSearchSpace = HyperparameterSearchSpace(hyperparameter='T_0',
                                                                   value_range=(1, 20),
                                                                   default_value=1,
                                                                   ),
        T_mult: HyperparameterSearchSpace = HyperparameterSearchSpace(hyperparameter='T_mult',
                                                                      value_range=(1.0, 2.0),
                                                                      default_value=1.0,
                                                                      )
    ) -> ConfigurationSpace:

        cs = ConfigurationSpace()
        add_hyperparameter(cs, T_0, UniformIntegerHyperparameter)
        add_hyperparameter(cs, T_mult, UniformFloatHyperparameter)

        return cs
