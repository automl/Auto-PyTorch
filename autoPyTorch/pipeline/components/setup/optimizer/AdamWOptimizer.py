from typing import Any, Dict, Optional, Union

from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import (
    UniformFloatHyperparameter,
)

import numpy as np

from torch.optim import AdamW

from autoPyTorch.datasets.base_dataset import BaseDatasetPropertiesType
from autoPyTorch.pipeline.components.setup.optimizer.base_optimizer import BaseOptimizerComponent
from autoPyTorch.utils.common import HyperparameterSearchSpace, add_hyperparameter


class AdamWOptimizer(BaseOptimizerComponent):
    """
    Implements AdamW  algorithm.

    Args:
        lr (float): learning rate (default: 1e-2)
        beta1 (float): coefficients used for computing running averages of gradient
        beta2 (float): coefficients used for computing running averages of square
        weight_decay (float): weight decay (L2 penalty)
        random_state (Optional[np.random.RandomState]): random state
    """

    def __init__(
        self,
        lr: float,
        beta1: float,
        beta2: float,
        weight_decay: float,
        random_state: Optional[np.random.RandomState] = None,
    ):
        super().__init__()
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.weight_decay = weight_decay
        self.random_state = random_state

    def fit(self, X: Dict[str, Any], y: Any = None) -> BaseOptimizerComponent:
        """
        Fits a component by using an input dictionary with pre-requisites

        Args:
            X (X: Dict[str, Any]): Dependencies needed by current component to perform fit
            y (Any): not used. To comply with sklearn API

        Returns:
            A instance of self
        """

        # Make sure that input dictionary X has the required
        # information to fit this stage
        self.check_requirements(X, y)

        self.optimizer = AdamW(
            params=X['network'].parameters(),
            lr=self.lr,
            betas=(self.beta1, self.beta2),
            weight_decay=self.weight_decay,
        )

        return self

    @staticmethod
    def get_properties(dataset_properties: Optional[Dict[str, BaseDatasetPropertiesType]] = None
                       ) -> Dict[str, Union[str, bool]]:
        return {
            'shortname': 'AdamW',
            'name': 'Adaptive Momentum Optimizer with decouple weight decay',
        }

    @staticmethod
    def get_hyperparameter_search_space(
        dataset_properties: Optional[Dict[str, BaseDatasetPropertiesType]] = None,
        lr: HyperparameterSearchSpace = HyperparameterSearchSpace(hyperparameter="lr",
                                                                  value_range=(1e-5, 1e-1),
                                                                  default_value=1e-2,
                                                                  log=True),
        beta1: HyperparameterSearchSpace = HyperparameterSearchSpace(hyperparameter="beta1",
                                                                     value_range=(0.85, 0.999),
                                                                     default_value=0.9),
        beta2: HyperparameterSearchSpace = HyperparameterSearchSpace(hyperparameter="beta2",
                                                                     value_range=(0.9, 0.9999),
                                                                     default_value=0.9),
        weight_decay: HyperparameterSearchSpace = HyperparameterSearchSpace(hyperparameter="weight_decay",
                                                                            value_range=(0.0, 0.1),
                                                                            default_value=0.0),
    ) -> ConfigurationSpace:
        cs = ConfigurationSpace()

        # The learning rate for the model
        add_hyperparameter(cs, lr, UniformFloatHyperparameter)
        add_hyperparameter(cs, beta1, UniformFloatHyperparameter)
        add_hyperparameter(cs, beta2, UniformFloatHyperparameter)
        add_hyperparameter(cs, weight_decay, UniformFloatHyperparameter)

        return cs
