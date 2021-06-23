from typing import Any, Dict, Optional, Union

from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import (
    UniformIntegerHyperparameter,
)

import numpy as np

import torch.optim.lr_scheduler

from autoPyTorch.datasets.base_dataset import BaseDatasetPropertiesType
from autoPyTorch.pipeline.components.setup.lr_scheduler.base_scheduler import BaseLRComponent
from autoPyTorch.pipeline.components.setup.lr_scheduler.constants import StepIntervalUnit
from autoPyTorch.utils.common import HyperparameterSearchSpace, add_hyperparameter


class CosineAnnealingLR(BaseLRComponent):
    """
    Set the learning rate of each parameter group using a cosine annealing schedule

    Args:
        T_max (int): Maximum number of iterations.

    """
    def __init__(
        self,
        T_max: int,
        step_interval: Union[str, StepIntervalUnit] = StepIntervalUnit.epoch,
        random_state: Optional[np.random.RandomState] = None
    ):

        super().__init__(step_interval)
        self.T_max = T_max
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

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer=X['optimizer'],
            T_max=int(self.T_max)
        )
        return self

    @staticmethod
    def get_properties(dataset_properties: Optional[Dict[str, BaseDatasetPropertiesType]] = None
                       ) -> Dict[str, Union[str, bool]]:
        return {
            'shortname': 'CosineAnnealing',
            'name': 'Cosine Annealing',
        }

    @staticmethod
    def get_hyperparameter_search_space(
        dataset_properties: Optional[Dict[str, BaseDatasetPropertiesType]] = None,
        T_max: HyperparameterSearchSpace = HyperparameterSearchSpace(hyperparameter='T_max',
                                                                     value_range=(10, 500),
                                                                     default_value=200,
                                                                     )
    ) -> ConfigurationSpace:

        cs = ConfigurationSpace()
        add_hyperparameter(cs, T_max, UniformIntegerHyperparameter)

        return cs
