from typing import Any, Dict, Optional

from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import (
    CategoricalHyperparameter,
    UniformFloatHyperparameter,
)

import numpy as np

from torch.optim import SGD

from autoPyTorch.pipeline.components.setup.optimizer.base_optimizer import BaseOptimizerComponent
from autoPyTorch.utils.common import HyperparameterSearchSpace, add_hyperparameter, get_hyperparameter


class SGDOptimizer(BaseOptimizerComponent):
    """
    Implements Stochstic Gradient Descend  algorithm.

    Args:
        lr (float): learning rate (default: 1e-2)
        momentum (float): momentum factor (default: 0)
        use_weight_decay (bool): flag for the activation of weight decay
        weight_decay (float): weight decay (L2 penalty) (default: 0)
        random_state (Optional[np.random.RandomState]): random state
    """
    def __init__(
        self,
        lr: float,
        momentum: float,
        use_weight_decay: bool,
        weight_decay: float = 0,
        random_state: Optional[np.random.RandomState] = None,
    ):

        super().__init__()
        self.lr = lr
        self.momentum = momentum
        self.use_weight_decay = use_weight_decay
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

        self.optimizer = SGD(
            params=X['network'].parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
            momentum=self.momentum,
        )

        return self

    @staticmethod
    def get_properties(dataset_properties: Optional[Dict[str, Any]] = None) -> Dict[str, str]:
        return {
            'shortname': 'SGD',
            'name': 'Stochastic gradient descent (optionally with momentum)',
        }

    @staticmethod
    def get_hyperparameter_search_space(
        dataset_properties: Optional[Dict] = None,
        lr: HyperparameterSearchSpace = HyperparameterSearchSpace(hyperparameter="lr",
                                                                  value_range=(1e-5, 1e-1),
                                                                  default_value=1e-2,
                                                                  log=True),
        use_weight_decay: HyperparameterSearchSpace = HyperparameterSearchSpace(hyperparameter="use_weight_decay",
                                                                                value_range=(True, False),
                                                                                default_value=True,
                                                                                ),
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
        add_hyperparameter(cs, momentum, UniformFloatHyperparameter)

        weight_decay = get_hyperparameter(weight_decay, UniformFloatHyperparameter)
        use_weight_decay = get_hyperparameter(use_weight_decay, CategoricalHyperparameter)
        cs.add_hyperparameters([use_weight_decay, weight_decay])

        cs.add_condition(
            ConfigurationSpace.EqualsCondition(
                weight_decay,
                use_weight_decay,
                True,
            )
        )

        return cs
