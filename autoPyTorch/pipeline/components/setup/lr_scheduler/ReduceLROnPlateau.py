from typing import Any, Dict, Optional, Union

from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import (
    CategoricalHyperparameter,
    UniformFloatHyperparameter,
    UniformIntegerHyperparameter
)

import numpy as np

import torch.optim.lr_scheduler

from autoPyTorch.datasets.base_dataset import BaseDatasetPropertiesType
from autoPyTorch.pipeline.components.setup.lr_scheduler.base_scheduler import BaseLRComponent
from autoPyTorch.pipeline.components.setup.lr_scheduler.constants import StepIntervalUnit
from autoPyTorch.utils.common import HyperparameterSearchSpace, add_hyperparameter


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
        step_interval (str): step should be called after validation in the case of ReduceLROnPlateau
        random_state (Optional[np.random.RandomState]): random state

    Reference:
        https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.ReduceLROnPlateau.html#torch.optim.lr_scheduler.ReduceLROnPlateau
    """

    def __init__(
        self,
        mode: str,
        factor: float,
        patience: int,
        step_interval: Union[str, StepIntervalUnit] = StepIntervalUnit.valid,
        random_state: Optional[np.random.RandomState] = None,
    ):
        super().__init__(step_interval)
        self.mode = mode
        self.factor = factor
        self.patience = patience
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

        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=X['optimizer'],
            mode=self.mode,
            factor=float(self.factor),
            patience=int(self.patience),
        )
        return self

    @staticmethod
    def get_properties(dataset_properties: Optional[Dict[str, BaseDatasetPropertiesType]] = None
                       ) -> Dict[str, Union[str, bool]]:
        return {
            'shortname': 'ReduceLROnPlateau',
            'name': 'ReduceLROnPlateau',
        }

    @staticmethod
    def get_hyperparameter_search_space(
        dataset_properties: Optional[Dict[str, BaseDatasetPropertiesType]] = None,
        mode: HyperparameterSearchSpace = HyperparameterSearchSpace(hyperparameter='mode',
                                                                    value_range=('min', 'max'),
                                                                    default_value='min',
                                                                    ),
        patience: HyperparameterSearchSpace = HyperparameterSearchSpace(hyperparameter='patience',
                                                                        value_range=(5, 20),
                                                                        default_value=10,
                                                                        ),
        factor: HyperparameterSearchSpace = HyperparameterSearchSpace(hyperparameter='factor',
                                                                      value_range=(0.01, 0.9),
                                                                      default_value=0.1,
                                                                      )
    ) -> ConfigurationSpace:

        cs = ConfigurationSpace()

        add_hyperparameter(cs, mode, CategoricalHyperparameter)
        add_hyperparameter(cs, patience, UniformIntegerHyperparameter)
        add_hyperparameter(cs, factor, UniformFloatHyperparameter)

        return cs
