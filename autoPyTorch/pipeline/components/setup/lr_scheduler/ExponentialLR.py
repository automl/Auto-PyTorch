from typing import Any, Dict, Optional, Union

from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import (
    UniformFloatHyperparameter,
)

import numpy as np

import torch.optim.lr_scheduler

from autoPyTorch.datasets.base_dataset import BaseDatasetPropertiesType
from autoPyTorch.pipeline.components.setup.lr_scheduler.base_scheduler import BaseLRComponent
from autoPyTorch.pipeline.components.setup.lr_scheduler.constants import StepIntervalUnit
from autoPyTorch.utils.common import HyperparameterSearchSpace, add_hyperparameter


class ExponentialLR(BaseLRComponent):
    """
    Decays the learning rate of each parameter group by gamma every epoch.
    When last_epoch=-1, sets initial lr as lr.

    Args:
        gamma (float): Multiplicative factor of learning rate decay.

    """
    def __init__(
        self,
        gamma: float,
        step_interval: Union[str, StepIntervalUnit] = StepIntervalUnit.epoch,
        random_state: Optional[np.random.RandomState] = None
    ):

        super().__init__(step_interval)
        self.gamma = gamma
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

        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=X['optimizer'],
            gamma=float(self.gamma)
        )
        return self

    @staticmethod
    def get_properties(dataset_properties: Optional[Dict[str, BaseDatasetPropertiesType]] = None
                       ) -> Dict[str, Union[str, bool]]:
        return {
            'shortname': 'ExponentialLR',
            'name': 'Exponential Learning Rate Scheduler',
        }

    @staticmethod
    def get_hyperparameter_search_space(
        dataset_properties: Optional[Dict[str, BaseDatasetPropertiesType]] = None,
        gamma: HyperparameterSearchSpace = HyperparameterSearchSpace(hyperparameter='gamma',
                                                                     value_range=(0.7, 0.9999),
                                                                     default_value=0.9,
                                                                     )
    ) -> ConfigurationSpace:

        cs = ConfigurationSpace()
        add_hyperparameter(cs, gamma, UniformFloatHyperparameter)

        return cs
