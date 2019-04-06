#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File which contains the optimizers.
"""

from autoPyTorch.utils.config_space_hyperparameter import get_hyperparameter, add_hyperparameter

import torch.optim as optim

import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH

__author__ = "Max Dippel, Michael Burkart and Matthias Urban"
__version__ = "0.0.1"
__license__ = "BSD"

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
        return optim.Adam(params=params, lr=config['learning_rate'], weight_decay=config['weight_decay'])
    
    @staticmethod
    def get_config_space(
        learning_rate=((0.0001, 0.1), True),
        weight_decay=(0.0001, 0.1)
    ):
        cs = CS.ConfigurationSpace()
        add_hyperparameter(cs, CSH.UniformFloatHyperparameter, 'learning_rate', learning_rate)
        add_hyperparameter(cs, CSH.UniformFloatHyperparameter, 'weight_decay', weight_decay)
        return cs


class SgdOptimizer(AutoNetOptimizerBase):
    
    def _get_optimizer(self, params, config):
        return optim.SGD(params=params, lr=config['learning_rate'], momentum=config['momentum'], weight_decay=config['weight_decay'])
    
    @staticmethod
    def get_config_space(
        learning_rate=((0.0001, 0.1), True),
        momentum=((0.1, 0.9), True),
        weight_decay=(0.0001, 0.1)
    ):
        cs = CS.ConfigurationSpace()
        add_hyperparameter(cs, CSH.UniformFloatHyperparameter, 'learning_rate', learning_rate)
        add_hyperparameter(cs, CSH.UniformFloatHyperparameter, 'momentum', momentum)
        add_hyperparameter(cs, CSH.UniformFloatHyperparameter, 'weight_decay', weight_decay)
        return cs
