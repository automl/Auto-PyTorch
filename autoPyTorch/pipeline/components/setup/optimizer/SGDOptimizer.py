from typing import Any, Dict, Optional, Tuple

import ConfigSpace as CS
from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import (
    CategoricalHyperparameter,
    UniformFloatHyperparameter,
)

import numpy as np

from torch.optim import SGD

from autoPyTorch.pipeline.components.setup.optimizer.base_optimizer import BaseOptimizerComponent


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
    def get_hyperparameter_search_space(dataset_properties: Optional[Dict] = None,
                                        lr: Tuple[Tuple, float, bool] = ((1e-5, 1e-1), 1e-2, True),
                                        use_weight_decay: Tuple[Tuple, bool] = ((True, False), True),
                                        weight_decay: Tuple[Tuple, float, bool] = ((1E-7, 0.1), 0.0, True),
                                        momentum: Tuple[Tuple, float] = ((0.0, 0.99), 0.0),
                                        ) -> ConfigurationSpace:

        cs = ConfigurationSpace()

        # The learning rate for the model
        lr = UniformFloatHyperparameter('lr', lower=lr[0][0], upper=lr[0][1],
                                        default_value=lr[1], log=lr[2])

        use_wd = CategoricalHyperparameter(
            'use_weight_decay',
            choices=use_weight_decay[0],
            default_value=use_weight_decay[1],
        )

        weight_decay = UniformFloatHyperparameter('weight_decay', lower=weight_decay[0][0], upper=weight_decay[0][1],
                                                  default_value=weight_decay[1], log=weight_decay[2])

        momentum = UniformFloatHyperparameter('momentum', lower=momentum[0][0], upper=momentum[0][1],
                                              default_value=momentum[1])

        cs.add_hyperparameters([lr, weight_decay, momentum, use_wd])

        cs.add_condition(
            CS.EqualsCondition(
                weight_decay,
                use_wd,
                True,
            )
        )

        return cs
