#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File which contains the optimizers.
"""
from collections import defaultdict

import torch
import torch.optim as optim
from torch.optim.optimizer import Optimizer
import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH

from autoPyTorch.utils.config_space_hyperparameter import get_hyperparameter, add_hyperparameter

__author__ = "Max Dippel, Michael Burkart and Matthias Urban"
__version__ = "0.0.1"
__license__ = "BSD"


@staticmethod
def get_boolean_variable_status_from_config(variable_config):
    variable_status = 'deactivated'
    if isinstance(variable_config, tuple):
        # multiple values are given
        if isinstance(variable_config[0], list):
            # set by the searchspace update
            predefined_values = variable_config[0]
            if len(predefined_values) > 1:
                variable_status = 'conditional'
            else:
                if predefined_values[0]:
                    variable_status = 'active'
        else:
            variable_status = 'conditional'
    else:
        if isinstance(variable_config, bool):
            if variable_config:
                variable_status = 'active'

    return variable_status


class AutoNetOptimizerBase(object):
    def __new__(cls, params, config):
        return cls._get_optimizer(cls, params, config)

    def _get_optimizer(self, params, config):
        raise ValueError('Override the method _get_optimizer and do not call the base class implementation')

    @staticmethod
    def get_config_space(*args, **kwargs):
        return CS.ConfigurationSpace()


class AdamOptimizer(AutoNetOptimizerBase):
    
    def _get_optimizer(self, params, config):
        if config['use_weight_decay']:
            weight_decay = config['weight_decay']
        else:
            weight_decay = 0

        return optim.Adam(params=params, lr=config['learning_rate'], weight_decay=weight_decay)
    
    @staticmethod
    def get_config_space(
        learning_rate=((1e-4, 0.1), True),
        weight_decay=(1e-5, 0.1),
        use_weight_decay=(True, False),
    ):
        cs = CS.ConfigurationSpace()
        add_hyperparameter(cs, CSH.UniformFloatHyperparameter, 'learning_rate', learning_rate)
        weight_decay_activation = add_hyperparameter(
            cs,
            CSH.CategoricalHyperparameter,
            'use_weight_decay',
            use_weight_decay
        )
        weight_decay_status = get_boolean_variable_status_from_config(use_weight_decay)

        if weight_decay_status == 'active' or weight_decay_status == 'conditional':
            weight_decay_value = add_hyperparameter(
                cs,
                CSH.UniformFloatHyperparameter,
                'weight_decay',
                weight_decay
            )
            if weight_decay_status == 'conditional':
                cs.add_condition(CS.EqualsCondition(weight_decay_value, weight_decay_activation, True))

        return cs


class AdamWOptimizer(AutoNetOptimizerBase):
    
    def _get_optimizer(self, params, config):
        if config['use_weight_decay']:
            weight_decay = config['weight_decay']
        else:
            weight_decay = 0

        return optim.AdamW(params=params, lr=config['learning_rate'], weight_decay=weight_decay)
    
    @staticmethod
    def get_config_space(
        learning_rate=((1e-4, 0.1), True),
        use_weight_decay=(True, False),
        weight_decay=(1e-5, 0.1),
    ):
        cs = CS.ConfigurationSpace()
        add_hyperparameter(cs, CSH.UniformFloatHyperparameter, 'learning_rate', learning_rate)
        weight_decay_activation = add_hyperparameter(
            cs,
            CSH.CategoricalHyperparameter,
            'use_weight_decay',
            use_weight_decay
        )
        validate_if_activated = False
        if isinstance(use_weight_decay, tuple):
            if isinstance(use_weight_decay[0], list):
                value_to_check = use_weight_decay[0]

            else:
                value_to_check = use_weight_decay
                validate_if_activated = True
        else:
            if isinstance(use_weight_decay, bool):
                value_to_check = use_weight_decay

        if True in value_to_check:
            weight_decay_value = add_hyperparameter(
                cs,
                CSH.UniformFloatHyperparameter,
                'weight_decay',
                weight_decay
            )
            if validate_if_activated:
                cs.add_condition(CS.EqualsCondition(weight_decay_value, weight_decay_activation, True))

        return cs


class SgdOptimizer(AutoNetOptimizerBase):
    
    def _get_optimizer(self, params, config):
        if config['use_weight_decay']:
            weight_decay = config['weight_decay']
        else:
            weight_decay = 0


        return optim.SGD(params=params, lr=config['learning_rate'], momentum=config['momentum'], weight_decay=weight_decay)
    
    @staticmethod
    def get_config_space(
        learning_rate=((1e-4, 0.1), True),
        momentum=((0.1, 0.99), True),
        use_weight_decay=(True, False),
        weight_decay=(1e-5, 0.1),
    ):
        cs = CS.ConfigurationSpace()
        add_hyperparameter(cs, CSH.UniformFloatHyperparameter, 'learning_rate', learning_rate)
        add_hyperparameter(cs, CSH.UniformFloatHyperparameter, 'momentum', momentum)
        weight_decay_activation = add_hyperparameter(
            cs,
            CSH.CategoricalHyperparameter,
            'use_weight_decay',
            use_weight_decay
        )
        validate_if_activated = False
        if isinstance(use_weight_decay, tuple):
            if isinstance(use_weight_decay[0], list):
                value_to_check = use_weight_decay[0]

            else:
                value_to_check = use_weight_decay
                validate_if_activated = True
        else:
            if isinstance(use_weight_decay, bool):
                value_to_check = use_weight_decay

        if True in value_to_check:
            weight_decay_value = add_hyperparameter(
                cs,
                CSH.UniformFloatHyperparameter,
                'weight_decay',
                weight_decay
            )
            if validate_if_activated:
                cs.add_condition(CS.EqualsCondition(weight_decay_value, weight_decay_activation, True))

        return cs


class RMSpropOptimizer(AutoNetOptimizerBase):
    
    def _get_optimizer(self, params, config):
        if config['use_weight_decay']:
            weight_decay = config['weight_decay']
        else:
            weight_decay = 0

        return optim.RMSprop(params=params, lr=config['learning_rate'], momentum=config['momentum'], weight_decay=weight_decay, centered=False)

    @staticmethod
    def get_config_space(
        learning_rate=((1e-4, 0.1), True),
        momentum=((0.1, 0.99), True),
        use_weight_decay=(True, False),
        weight_decay=(1e-5, 0.1),
        alpha=(0.1,0.99),
    ):
        cs = CS.ConfigurationSpace()
        add_hyperparameter(cs, CSH.UniformFloatHyperparameter, 'learning_rate', learning_rate)
        add_hyperparameter(cs, CSH.UniformFloatHyperparameter, 'momentum', momentum)
        weight_decay_activation = add_hyperparameter(
            cs,
            CSH.CategoricalHyperparameter,
            'use_weight_decay',
            use_weight_decay
        )
        validate_if_activated = False
        if isinstance(use_weight_decay, tuple):
            if isinstance(use_weight_decay[0], list):
                value_to_check = use_weight_decay[0]

            else:
                value_to_check = use_weight_decay
                validate_if_activated = True
        else:
            if isinstance(use_weight_decay, bool):
                value_to_check = use_weight_decay

        if True in value_to_check:
            weight_decay_value = add_hyperparameter(
                cs,
                CSH.UniformFloatHyperparameter,
                'weight_decay',
                weight_decay
            )
            if validate_if_activated:
                cs.add_condition(CS.EqualsCondition(weight_decay_value, weight_decay_activation, True))

        add_hyperparameter(cs, CSH.UniformFloatHyperparameter, 'alpha', alpha)
        return cs


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
        self.la_alpha = config["la_alpha"]
        self.la_alpha = torch.tensor(self.la_alpha)
        self._total_la_steps = config["la_steps"]
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
    def get_config_space(
            la_steps=((5, 10), False),
            la_alpha=((0.5, 0.8), False),
    ):
        cs = CS.ConfigurationSpace()
        add_hyperparameter(cs, CS.UniformIntegerHyperparameter, 'la_steps', la_steps)
        add_hyperparameter(cs, CSH.UniformFloatHyperparameter, 'la_alpha', la_alpha)

        return cs
