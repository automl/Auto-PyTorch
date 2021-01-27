from typing import Any, Dict, Optional

from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import (
    UniformFloatHyperparameter,
)

import numpy as np

from torch.optim import RMSprop

from autoPyTorch.pipeline.components.setup.optimizer.base_optimizer import BaseOptimizerComponent


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
    def get_properties(dataset_properties: Optional[Dict[str, Any]] = None) -> Dict[str, str]:
        return {
            'shortname': 'RMSprop',
            'name': 'RMSprop Optimizer',
        }

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties: Optional[Dict] = None,
                                        lr=[(1e-5, 1e-1), 1e-2, True],
                                        alpha=[(0.1, 0.99), 0.99],
                                        weight_decay=[(0.0, 0.1), 0.0],
                                        momentum=[(0.0, 0.99), 0.0],
                                        ) -> ConfigurationSpace:

        cs = ConfigurationSpace()

        # The learning rate for the model
        lr = UniformFloatHyperparameter('lr', lower=lr[0][0], upper=lr[0][1],
                                        default_value=lr[1], log=lr[2])

        alpha = UniformFloatHyperparameter('alpha', lower=alpha[0][0], upper=alpha[0][1],
                                           default_value=alpha[1])

        weight_decay = UniformFloatHyperparameter('weight_decay', lower=weight_decay[0][0], upper=weight_decay[0][1],
                                                  default_value=weight_decay[1])

        momentum = UniformFloatHyperparameter('momentum', lower=momentum[0][0], upper=momentum[0][1],
                                              default_value=momentum[1])

        cs.add_hyperparameters([lr, alpha, weight_decay, momentum])

        return cs
