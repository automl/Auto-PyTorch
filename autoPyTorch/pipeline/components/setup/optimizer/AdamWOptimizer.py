from typing import Any, Dict, Optional

import ConfigSpace as CS
from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import (
    CategoricalHyperparameter,
    UniformFloatHyperparameter,
)

import numpy as np

from torch.optim import AdamW

from autoPyTorch.pipeline.components.setup.optimizer.base_optimizer import BaseOptimizerComponent
from autoPyTorch.utils.common import HyperparameterSearchSpace, add_hyperparameter, get_hyperparameter


class AdamWOptimizer(BaseOptimizerComponent):
    """
    Implements AdamW  algorithm.

    Args:
        lr (float): learning rate (default: 1e-2)
        beta1 (float): coefficients used for computing running averages of gradient
        beta2 (float): coefficients used for computing running averages of square
        use_weight_decay (bool): flag for the activation of weight decay
        weight_decay (float): weight decay (L2 penalty) (default: 0)
        random_state (Optional[np.random.RandomState]): random state
    """

    def __init__(
        self,
        lr: float,
        beta1: float,
        beta2: float,
        use_weight_decay: bool,
        weight_decay: float = 0,
        random_state: Optional[np.random.RandomState] = None,
    ):
        super().__init__()
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
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
    def get_hyperparameter_search_space(
        dataset_properties: Optional[Dict] = None,
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
        use_weight_decay: HyperparameterSearchSpace = HyperparameterSearchSpace(hyperparameter="use_weight_decay",
                                                                                value_range=(True, False),
                                                                                default_value=True,
                                                                                ),
        weight_decay: HyperparameterSearchSpace = HyperparameterSearchSpace(hyperparameter="weight_decay",
                                                                            value_range=(1E-7, 0.1),
                                                                            default_value=1E-4,
                                                                            log=False),
    ) -> ConfigurationSpace:
        cs = ConfigurationSpace()

        # The learning rate for the model
        add_hyperparameter(cs, lr, UniformFloatHyperparameter)
        add_hyperparameter(cs, beta1, UniformFloatHyperparameter)
        add_hyperparameter(cs, beta2, UniformFloatHyperparameter)

        weight_decay_flag = False
        if any(use_weight_decay.value_range):
            weight_decay_flag = True

        use_weight_decay = get_hyperparameter(use_weight_decay, CategoricalHyperparameter)
        cs.add_hyperparameter(use_weight_decay)

        if weight_decay_flag:
            weight_decay = get_hyperparameter(weight_decay, UniformFloatHyperparameter)
            cs.add_hyperparameter(weight_decay)
            cs.add_condition(
                CS.EqualsCondition(
                    weight_decay,
                    use_weight_decay,
                    True,
                )
            )

        return cs
