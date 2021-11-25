from typing import Any, Dict, Optional, Union

from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import (
    UniformFloatHyperparameter,
)

import numpy as np

from torch.optim import RMSprop

from autoPyTorch.datasets.base_dataset import BaseDatasetPropertiesType
from autoPyTorch.pipeline.components.setup.optimizer.base_optimizer import BaseOptimizerComponent
from autoPyTorch.utils.common import HyperparameterSearchSpace, add_hyperparameter


class RMSpropOptimizer(BaseOptimizerComponent):
    """
    Implements RMSprop algorithm.
    The implementation here takes the square root of the gradient average
    before adding epsilon

    Args:
        lr (float): learning rate (default: 1e-2)
        momentum (float): momentum factor (default: 0)
        alpha (float): smoothing constant (default: 0.99)
        weight_decay (float): weight decay (L2 penalty) (default: 0)
        random_state (Optional[np.random.RandomState]): random state
    """

    def __init__(
        self,
        lr: float,
        momentum: float,
        alpha: float,
        weight_decay: float,
        random_state: Optional[np.random.RandomState] = None,
    ):
        super().__init__()
        self.lr = lr
        self.momentum = momentum
        self.alpha = alpha
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

        self.optimizer = RMSprop(
            params=X['network'].parameters(),
            lr=self.lr,
            alpha=self.alpha,
            weight_decay=self.weight_decay,
            momentum=self.momentum,
        )

        return self

    @staticmethod
    def get_properties(dataset_properties: Optional[Dict[str, BaseDatasetPropertiesType]] = None
                       ) -> Dict[str, Union[str, bool]]:
        return {
            'shortname': 'RMSprop',
            'name': 'RMSprop Optimizer',
        }

    @staticmethod
    def get_hyperparameter_search_space(
        dataset_properties: Optional[Dict[str, BaseDatasetPropertiesType]] = None,
        lr: HyperparameterSearchSpace = HyperparameterSearchSpace(hyperparameter="lr",
                                                                  value_range=(1e-5, 1e-1),
                                                                  default_value=1e-2,
                                                                  log=True),
        alpha: HyperparameterSearchSpace = HyperparameterSearchSpace(hyperparameter="alpha",
                                                                     value_range=(0.1, 0.99),
                                                                     default_value=0.99),
        weight_decay: HyperparameterSearchSpace = HyperparameterSearchSpace(hyperparameter="weight_decay",
                                                                            value_range=(0.0, 0.1),
                                                                            default_value=0.0),
        momentum: HyperparameterSearchSpace = HyperparameterSearchSpace(hyperparameter="momentum",
                                                                        value_range=(0.0, 0.99),
                                                                        default_value=0.0),
    ) -> ConfigurationSpace:
        cs = ConfigurationSpace()

        # The learning rate for the model
        add_hyperparameter(cs, lr, UniformFloatHyperparameter)
        add_hyperparameter(cs, alpha, UniformFloatHyperparameter)
        add_hyperparameter(cs, momentum, UniformFloatHyperparameter)
        add_hyperparameter(cs, weight_decay, UniformFloatHyperparameter)

        return cs
