#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This file contains the different learning rate schedulers of AutoNet.
"""

import torch
import torch.optim.lr_scheduler as lr_scheduler

import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH

from autoPyTorch.components.lr_scheduler.lr_schedulers_config import CSConfig

__author__ = "Max Dippel, Michael Burkart and Matthias Urban"
__version__ = "0.0.1"
__license__ = "BSD"

class AutoNetLearningRateSchedulerBase(object):
    def __new__(cls, params, config):
        scheduler = cls._get_scheduler(cls, params, config)
        if not hasattr(scheduler, "allows_early_stopping"):
            scheduler.allows_early_stopping = True
        if not hasattr(scheduler, "snapshot_before_restart"):
            scheduler.snapshot_before_restart = False
        return scheduler

    def _get_scheduler(self, optimizer, config):
        raise ValueError('Override the method _get_scheduler and do not call the base class implementation')

    @staticmethod
    def get_config_space(*args, **kwargs):
        # currently no use but might come in handy in the future
        return CS.ConfigurationSpace()

class SchedulerNone(AutoNetLearningRateSchedulerBase):

    def _get_scheduler(self, optimizer, config):
        return NoScheduling()

class SchedulerStepLR(AutoNetLearningRateSchedulerBase):

    def _get_scheduler(self, optimizer, config):
        return lr_scheduler.StepLR(optimizer=optimizer, step_size=config['step_size'], gamma=config['gamma'], last_epoch=-1)
    
    @staticmethod
    def get_config_space(*args, **kwargs):
        cs = CS.ConfigurationSpace()
        config = CSConfig['step_lr']
        cs.add_hyperparameter(CSH.UniformIntegerHyperparameter('step_size', lower=config['step_size'][0], upper=config['step_size'][1]))
        cs.add_hyperparameter(CSH.UniformFloatHyperparameter('gamma', lower=config['gamma'][0], upper=config['gamma'][1]))
        cs.add_configuration_space(prefix='', delimiter='', configuration_space=AutoNetLearningRateSchedulerBase.get_config_space(*args, **kwargs))
        return cs

class SchedulerExponentialLR(AutoNetLearningRateSchedulerBase):
    
    def _get_scheduler(self, optimizer, config):
        return lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=config['gamma'], last_epoch=-1)
    
    @staticmethod
    def get_config_space(*args, **kwargs):
        cs = CS.ConfigurationSpace()
        config = CSConfig['exponential_lr']
        cs.add_hyperparameter(CSH.UniformFloatHyperparameter('gamma', lower=config['gamma'][0], upper=config['gamma'][1]))
        cs.add_configuration_space(prefix='', delimiter='', configuration_space=AutoNetLearningRateSchedulerBase.get_config_space(*args, **kwargs))
        return cs

class SchedulerReduceLROnPlateau(AutoNetLearningRateSchedulerBase):
    
    def _get_scheduler(self, optimizer, config):
        return lr_scheduler.ReduceLROnPlateau(optimizer=optimizer)
    
    @staticmethod
    def get_config_space(*args, **kwargs):
        cs = CS.ConfigurationSpace()
        config = CSConfig['reduce_on_plateau']
        cs.add_hyperparameter(CSH.UniformFloatHyperparameter('factor', lower=config['factor'][0], upper=config['factor'][1]))
        cs.add_hyperparameter(CSH.UniformIntegerHyperparameter('patience', lower=config['patience'][0], upper=config['patience'][1]))
        cs.add_configuration_space(prefix='', delimiter='', configuration_space=AutoNetLearningRateSchedulerBase.get_config_space(*args, **kwargs))
        return cs

class SchedulerCyclicLR(AutoNetLearningRateSchedulerBase):

    def _get_scheduler(self, optimizer, config):
        maf = config['max_factor']
        mif = config['min_factor']
        cl = config['cycle_length']
        r = maf - mif
        def l(epoch):
            if int(epoch//cl) % 2 == 1:
                lr = mif + (r * (float(epoch % cl)/float(cl)))
            else:
                lr = maf - (r * (float(epoch % cl)/float(cl)))
            return lr
            
        return lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=l, last_epoch=-1)
    
    @staticmethod
    def get_config_space(*args, **kwargs):
        cs = CS.ConfigurationSpace()
        config = CSConfig['cyclic_lr']
        cs.add_hyperparameter(CSH.UniformFloatHyperparameter('max_factor', lower=config['max_factor'][0], upper=config['max_factor'][1]))
        cs.add_hyperparameter(CSH.UniformFloatHyperparameter('min_factor', lower=config['min_factor'][0], upper=config['min_factor'][1]))
        cs.add_hyperparameter(CSH.UniformIntegerHyperparameter('cycle_length', lower=config['cycle_length'][0], upper=config['cycle_length'][1]))
        cs.add_configuration_space(prefix='', delimiter='', configuration_space=AutoNetLearningRateSchedulerBase.get_config_space(*args, **kwargs))
        return cs

class SchedulerCosineAnnealingWithRestartsLR(AutoNetLearningRateSchedulerBase):

    def _get_scheduler(self, optimizer, config):
        scheduler = CosineAnnealingWithRestartsLR(optimizer, T_max=config['T_max'], T_mult=config['T_mult'],last_epoch=-1)
        scheduler.allows_early_stopping = False
        scheduler.snapshot_before_restart = True
        return scheduler
    
    @staticmethod
    def get_config_space(*args, **kwargs):
        cs = CS.ConfigurationSpace()
        config = CSConfig['cosine_annealing_lr']
        cs.add_hyperparameter(CSH.UniformIntegerHyperparameter('T_max', lower=config['T_max'][0], upper=config['T_max'][1]))
        cs.add_hyperparameter(CSH.UniformFloatHyperparameter('T_mult', lower=config['T_mult'][0], upper=config['T_mult'][1]))
        cs.add_configuration_space(prefix='', delimiter='', configuration_space=AutoNetLearningRateSchedulerBase.get_config_space(*args, **kwargs))
        return cs


class NoScheduling():

    def step(self, epoch):
        return


import math
class CosineAnnealingWithRestartsLR(torch.optim.lr_scheduler._LRScheduler):

    r"""Copyright: pytorch
    Set the learning rate of each parameter group using a cosine annealing
    schedule, where :math:`\eta_{max}` is set to the initial lr and
    :math:`T_{cur}` is the number of epochs since the last restart in SGDR:

    .. math::

        \eta_t = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})(1 +
        \cos(\frac{T_{cur}}{T_{max}}\pi))

    When last_epoch=-1, sets initial lr as lr.

    It has been proposed in
    `SGDR: Stochastic Gradient Descent with Warm Restarts`_. This implements
    the cosine annealing part of SGDR, the restarts and number of iterations multiplier.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        T_max (int): Maximum number of iterations.
        T_mult (float): Multiply T_max by this number after each restart. Default: 1.
        eta_min (float): Minimum learning rate. Default: 0.
        last_epoch (int): The index of last epoch. Default: -1.

    .. _SGDR\: Stochastic Gradient Descent with Warm Restarts:
        https://arxiv.org/abs/1608.03983
    """

    def __init__(self, optimizer, T_max, eta_min=0, last_epoch=-1, T_mult=1):
        self.T_max = T_max
        self.T_mult = T_mult
        self.restart_every = T_max
        self.eta_min = eta_min
        self.restarts = 0
        self.restarted_at = 0
        super().__init__(optimizer, last_epoch)
    
    def restart(self):
        self.restart_every *= self.T_mult
        self.restarted_at = self.last_epoch
    
    def cosine(self, base_lr):
        return self.eta_min + (base_lr - self.eta_min) * (1 + math.cos(math.pi * self.step_n / self.restart_every)) / 2
    
    @property
    def step_n(self):
        return self.last_epoch - self.restarted_at

    def get_lr(self):
        if self.step_n >= self.restart_every:
            self.restart()
        return [self.cosine(base_lr) for base_lr in self.base_lrs]  
