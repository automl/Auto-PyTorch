#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File which contains the optimizers.
"""

import torch.optim as optim

import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH

from autoPyTorch.components.optimizer.optimizer_config import CSConfig

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
        # currently no use but might come in handy in the future
        return CS.ConfigurationSpace()


class AdamOptimizer(AutoNetOptimizerBase):
    
    def _get_optimizer(self, params, config):
        return optim.Adam(params=params, lr=config['learning_rate'], weight_decay=config['weight_decay'])
    
    @staticmethod
    def get_config_space(*args, **kwargs):
        cs = CS.ConfigurationSpace()
        config = CSConfig['adam_opt']
        cs.add_hyperparameter(CSH.UniformFloatHyperparameter('learning_rate', lower=config['learning_rate'][0], upper=config['learning_rate'][1], log=True))
        cs.add_hyperparameter(CSH.UniformFloatHyperparameter('weight_decay', lower=config['weight_decay'][0], upper=config['weight_decay'][1]))
        cs.add_configuration_space(prefix='', delimiter='', configuration_space=AutoNetOptimizerBase.get_config_space(*args, **kwargs))
        return cs


class AdamWOptimizer(AutoNetOptimizerBase):
    
    def _get_optimizer(self, params, config):
        return optim.AdamW(params=params, lr=config['learning_rate'], weight_decay=config['weight_decay'])
    
    @staticmethod
    def get_config_space(*args, **kwargs):
        cs = CS.ConfigurationSpace()
        config = CSConfig['adamw_opt']
        cs.add_hyperparameter(CSH.UniformFloatHyperparameter('learning_rate', lower=config['learning_rate'][0], upper=config['learning_rate'][1], log=True))
        cs.add_hyperparameter(CSH.UniformFloatHyperparameter('weight_decay', lower=config['weight_decay'][0], upper=config['weight_decay'][1]))
        cs.add_configuration_space(prefix='', delimiter='', configuration_space=AutoNetOptimizerBase.get_config_space(*args, **kwargs))
        return cs


class SgdOptimizer(AutoNetOptimizerBase):
    
    def _get_optimizer(self, params, config):
        return optim.SGD(params=params, lr=config['learning_rate'], momentum=config['momentum'], weight_decay=config['weight_decay'], nesterov=True)
    
    @staticmethod
    def get_config_space(*args, **kwargs):
        cs = CS.ConfigurationSpace()
        config = CSConfig['sgd_opt']
        cs.add_hyperparameter(CSH.UniformFloatHyperparameter('learning_rate', lower=config['learning_rate'][0], upper=config['learning_rate'][1], log=True))
        cs.add_hyperparameter(CSH.UniformFloatHyperparameter('momentum', lower=config['momentum'][0], upper=config['momentum'][1], log=False))
        cs.add_hyperparameter(CSH.UniformFloatHyperparameter('weight_decay', lower=config['weight_decay'][0], upper=config['weight_decay'][1], log=True))
        cs.add_configuration_space(prefix='', delimiter='', configuration_space=AutoNetOptimizerBase.get_config_space(*args, **kwargs))
        return cs


class RMSpropOptimizer(AutoNetOptimizerBase):
    
    def _get_optimizer(self, params, config):
        return optim.RMSprop(params=params, lr=config['learning_rate'], momentum=config['momentum'], weight_decay=config['weight_decay'], centered=False)
    
    @staticmethod
    def get_config_space(*args, **kwargs):
        cs = CS.ConfigurationSpace()
        config = CSConfig['rmsprop_opt']
        cs.add_hyperparameter(CSH.UniformFloatHyperparameter('learning_rate', lower=config['learning_rate'][0], upper=config['learning_rate'][1], log=True))
        cs.add_hyperparameter(CSH.UniformFloatHyperparameter('momentum', lower=config['momentum'][0], upper=config['momentum'][1], log=False))
        cs.add_hyperparameter(CSH.UniformFloatHyperparameter('weight_decay', lower=config['weight_decay'][0], upper=config['weight_decay'][1], log=True))
        cs.add_hyperparameter(CSH.UniformFloatHyperparameter('alpha', lower=config['alpha'][0], upper=config['alpha'][1], log=False))
        cs.add_configuration_space(prefix='', delimiter='', configuration_space=AutoNetOptimizerBase.get_config_space(*args, **kwargs))
        return cs
