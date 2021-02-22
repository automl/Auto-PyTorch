from collections import defaultdict
import re
from typing import Dict, Tuple

from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import (
    UniformFloatHyperparameter,
    UniformIntegerHyperparameter
)

import torch
from torch.optim.optimizer import Optimizer


def update_model_state_dict_from_swa(model: torch.nn.Module, swa_state_dict: Dict) -> None:
    """
    swa model adds a module keyword to each parameter,
    this function updates the state dict of the model
    using the state dict of the swa model
    Args:
        model:
        swa_state_dict:

    Returns:

    """
    model_state = model.state_dict()
    for name, param in swa_state_dict.items():
        name = re.sub('module.', '', name)
        if name not in model_state.keys():
            continue
        model_state[name].copy_(param)


class Lookahead(Optimizer):
    r"""PyTorch implementation of the lookahead wrapper.
    Lookahead Optimizer: https://arxiv.org/abs/1907.08610
    """

    def __init__(self, optimizer, config):
        """optimizer: inner optimizer
        la_steps (int): number of lookahead steps
        la_alpha (float): linear interpolation factor. 1.0 recovers the inner optimizer.
        pullback_momentum (str): change to inner optimizer momentum on interpolation update
        """
        self.optimizer = optimizer
        self._la_step = 0  # counter for inner optimizer
        self.la_alpha = config[f"{self.__class__.__name__}:la_alpha"]
        self.la_alpha = torch.tensor(self.la_alpha)
        self._total_la_steps = config[f"{self.__class__.__name__}:la_steps"]
        # TODO possibly incorporate different momentum options when using SGD
        pullback_momentum = "none"
        pullback_momentum = pullback_momentum.lower()
        assert pullback_momentum in ["reset", "pullback", "none"]
        self.pullback_momentum = pullback_momentum

        self.state = defaultdict(dict)

        # Cache the current optimizer parameters
        for group in optimizer.param_groups:
            for p in group['params']:
                param_state = self.state[p]
                param_state['cached_params'] = torch.zeros_like(p.data)
                param_state['cached_params'].copy_(p.data)
                if self.pullback_momentum == "pullback":
                    param_state['cached_mom'] = torch.zeros_like(p.data)

    def __getstate__(self):
        return {
            'state': self.state,
            'optimizer': self.optimizer,
            'la_alpha': self.la_alpha,
            '_la_step': self._la_step,
            '_total_la_steps': self._total_la_steps,
            'pullback_momentum': self.pullback_momentum
        }

    def zero_grad(self):
        self.optimizer.zero_grad()

    def get_la_step(self):
        return self._la_step

    def state_dict(self):
        return self.optimizer.state_dict()

    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict)

    def _backup_and_load_cache(self):
        """Useful for performing evaluation on the slow weights (which typically generalize better)
        """
        for group in self.optimizer.param_groups:
            for p in group['params']:
                param_state = self.state[p]
                param_state['backup_params'] = torch.zeros_like(p.data)
                param_state['backup_params'].copy_(p.data)
                p.data.copy_(param_state['cached_params'])

    def _clear_and_load_backup(self):
        for group in self.optimizer.param_groups:
            for p in group['params']:
                param_state = self.state[p]
                p.data.copy_(param_state['backup_params'])
                del param_state['backup_params']

    @property
    def param_groups(self):
        return self.optimizer.param_groups

    def step(self, closure=None):
        """Performs a single Lookahead optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = self.optimizer.step(closure)
        self._la_step += 1

        if self._la_step >= self._total_la_steps:
            self._la_step = 0
            # Lookahead and cache the current optimizer parameters
            for group in self.optimizer.param_groups:
                for p in group['params']:
                    param_state = self.state[p]
                    p.data.mul_(self.la_alpha).add_(1.0 - self.la_alpha, param_state['cached_params'])  # crucial line
                    param_state['cached_params'].copy_(p.data)
                    if self.pullback_momentum == "pullback":
                        internal_momentum = self.optimizer.state[p]["momentum_buffer"]
                        self.optimizer.state[p]["momentum_buffer"] = internal_momentum.mul_(self.la_alpha).add_(
                            1.0 - self.la_alpha, param_state["cached_mom"])
                        param_state["cached_mom"] = self.optimizer.state[p]["momentum_buffer"]
                    elif self.pullback_momentum == "reset":
                        self.optimizer.state[p]["momentum_buffer"] = torch.zeros_like(p.data)

        return loss

    def to(self, device):

        self.la_alpha.to(device)
        for group in self.optimizer.param_groups:
            for p in group['params']:
                param_state = self.state[p]
                param_state['cached_params'] = param_state['cached_params'].to(device)
                param_state['cached_params'].copy_(p.data)
                if self.pullback_momentum == "pullback":
                    param_state['cached_mom'] = param_state['cached_mom'].to(device)

    @staticmethod
    def get_hyperparameter_search_space(
        la_steps: Tuple[Tuple, int, bool] = ((5, 10), 6, False),
        la_alpha: Tuple[Tuple, float, bool] = ((0.5, 0.8), 0.6, False),
    ):
        cs = ConfigurationSpace()
        la_steps = UniformIntegerHyperparameter('la_steps', lower=la_steps[0][0],
                                                upper=la_steps[0][1],
                                                default_value=la_steps[1],
                                                log=la_steps[2])
        la_alpha = UniformFloatHyperparameter('la_alpha', lower=la_alpha[0][0],
                                              upper=la_alpha[0][1],
                                              default_value=la_alpha[1],
                                              log=la_alpha[2])
        cs.add_hyperparameters([la_steps, la_alpha])

        return cs
