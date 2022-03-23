from typing import Any, Callable, Dict, Iterable, Optional, Tuple, Union

import ConfigSpace as CS
from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import (
    CategoricalHyperparameter,
    UniformFloatHyperparameter,
)

import numpy as np

from torch import clone
from torch.optim.optimizer import Optimizer

from autoPyTorch.datasets.base_dataset import BaseDatasetPropertiesType
from autoPyTorch.pipeline.components.setup.optimizer.base_optimizer import BaseOptimizerComponent
from autoPyTorch.utils.common import HyperparameterSearchSpace, add_hyperparameter, get_hyperparameter


class SGDW(Optimizer):
    r"""Implements stochastic gradient descent (optionally with momentum).
    Nesterov momentum is based on the formula from
    `On the importance of initialization and momentum in deep learning`__.
    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): learning rate
        momentum (float, optional): momentum factor (default: 0)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        dampening (float, optional): dampening for momentum (default: 0)
        nesterov (bool, optional): enables Nesterov momentum (default: False)
    Example:
        >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()
    __ http://www.cs.toronto.edu/%7Ehinton/absps/momentum.pdf
    .. note::
        The implementation of SGD with Momentum/Nesterov subtly differs from
        Sutskever et. al. and implementations in some other frameworks.
        Considering the specific case of Momentum, the update can be written as
        .. math::
                  v = \rho * v + g \\
                  p = p - lr * v
        where p, g, v and :math:`\rho` denote the parameters, gradient,
        velocity, and momentum respectively.
        This is in contrast to Sutskever et. al. and
        other frameworks which employ an update of the form
        .. math::
             v = \rho * v + lr * g \\
             p = p - v
        The Nesterov version is analogously modified.
    """
    def __init__(
            self,
            params: Iterable,
            lr: float,
            weight_decay: float = 0,
            momentum: float = 0,
            dampening: float = 0,
            nesterov: bool = False,
    ):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(SGDW, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(SGDW, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    def step(
        self,
        closure: Optional[Callable] = None
    ):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data

                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = clone(
                            d_p
                        ).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(d_p, alpha=1 - dampening)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf

                # Apply momentum
                p.data.add_(d_p, alpha=-group['lr'])

                # Apply weight decay
                if weight_decay != 0:
                    p.data.add_(weight_decay, alpha=-group['lr'])
        return loss


class SGDWOptimizer(BaseOptimizerComponent):
    """
    Implements Stochstic Gradient Descend  algorithm.

    Args:
        lr (float): learning rate (default: 1e-2)
        momentum (float): momentum factor (default: 0)
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

        self.optimizer = SGDW(
            params=X['network'].parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
            momentum=self.momentum,
        )

        return self

    @staticmethod
    def get_properties(dataset_properties: Optional[Dict[str, BaseDatasetPropertiesType]] = None
                       ) -> Dict[str, Union[str, bool]]:
        return {
            'shortname': 'SGDW',
            'name': 'Stochastic gradient descent (optionally with momentum) with decoupled weight decay',
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
                                                                            value_range=(1E-5, 0.1),
                                                                            default_value=1E-4,
                                                                            log=True),
        momentum: HyperparameterSearchSpace = HyperparameterSearchSpace(hyperparameter="momentum",
                                                                        value_range=(0.0, 0.99),
                                                                        default_value=0.0),
        ) -> ConfigurationSpace:

        cs = ConfigurationSpace()

         # The learning rate for the model
        add_hyperparameter(cs, lr, UniformFloatHyperparameter)
        add_hyperparameter(cs, momentum, UniformFloatHyperparameter)

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
