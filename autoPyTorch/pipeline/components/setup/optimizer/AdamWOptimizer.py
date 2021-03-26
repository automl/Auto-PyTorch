from typing import Any, Dict, Optional, Tuple

from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import (
    UniformFloatHyperparameter,
)

import numpy as np

from torch.optim import AdamW

from autoPyTorch.pipeline.components.setup.optimizer.base_optimizer import BaseOptimizerComponent


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
    def get_properties(dataset_properties: Optional[Dict[str, Any]] = None) -> Dict[str, str]:
        return {
            'shortname': 'AdamW',
            'name': 'Adaptive Momentum Optimizer with decouple weight decay',
        }

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties: Optional[Dict] = None,
                                        lr: Tuple[Tuple, float, bool] = ((1e-5, 1e-1), 1e-2, True),
                                        beta1: Tuple[Tuple, float] = ((0.85, 0.999), 0.9),
                                        beta2: Tuple[Tuple, float] = ((0.9, 0.9999), 0.9),
                                        weight_decay: Tuple[Tuple, float, bool] = ((0.0, 0.1), 0.0, True)
                                        ) -> ConfigurationSpace:

        cs = ConfigurationSpace()

        # The learning rate for the model
        lr = UniformFloatHyperparameter('lr', lower=lr[0][0], upper=lr[0][1],
                                        default_value=lr[1], log=lr[2])

        beta1 = UniformFloatHyperparameter('beta1', lower=beta1[0][0], upper=beta1[0][1],
                                           default_value=beta1[1])

        beta2 = UniformFloatHyperparameter('beta2', lower=beta2[0][0], upper=beta2[0][1],
                                           default_value=beta2[1])

        weight_decay = UniformFloatHyperparameter('weight_decay', lower=weight_decay[0][0], upper=weight_decay[0][1],
                                                  default_value=weight_decay[1], log=weight_decay[2])

        cs.add_hyperparameters([lr, beta1, beta2, weight_decay])

        return cs
