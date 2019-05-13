#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This file contains the different learning rate schedulers of AutoNet.
"""

from autoPyTorch.utils.config_space_hyperparameter import add_hyperparameter, get_hyperparameter

import torch
import torch.optim.lr_scheduler as lr_scheduler

import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH

__author__ = "Max Dippel, Michael Burkart and Matthias Urban"
__version__ = "0.0.1"
__license__ = "BSD"

class AutoNetLearningRateSchedulerBase(object):
    def __new__(cls, optimizer, config):
        """Get a new instance of the scheduler
        
        Arguments:
            cls {class} -- Type of scheduler
            optimizer {Optmizer} -- A PyTorch Optimizer
            config {dict} -- Sampled lr_scheduler config
        
        Returns:
            AutoNetLearningRateSchedulerBase -- The learning rate scheduler object
        """
        scheduler = cls._get_scheduler(cls, optimizer, config)
        if not hasattr(scheduler, "allows_early_stopping"):
            scheduler.allows_early_stopping = True
        if not hasattr(scheduler, "snapshot_before_restart"):
            scheduler.snapshot_before_restart = False
        return scheduler

    def _get_scheduler(self, optimizer, config):
        raise ValueError('Override the method _get_scheduler and do not call the base class implementation')

    @staticmethod
    def get_config_space():
        return CS.ConfigurationSpace()

class SchedulerNone(AutoNetLearningRateSchedulerBase):

    def _get_scheduler(self, optimizer, config):
        return NoScheduling(optimizer=optimizer)

class SchedulerStepLR(AutoNetLearningRateSchedulerBase):

    def _get_scheduler(self, optimizer, config):
        return lr_scheduler.StepLR(optimizer=optimizer, step_size=config['step_size'], gamma=config['gamma'], last_epoch=-1)
    
    @staticmethod
    def get_config_space(
        step_size=(1, 10),
        gamma=(0.001, 0.9)
    ):
        cs = CS.ConfigurationSpace()
        add_hyperparameter(cs, CSH.UniformIntegerHyperparameter, 'step_size', step_size)
        add_hyperparameter(cs, CSH.UniformFloatHyperparameter, 'gamma', gamma)
        return cs

class SchedulerExponentialLR(AutoNetLearningRateSchedulerBase):
    
    def _get_scheduler(self, optimizer, config):
        return lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=config['gamma'], last_epoch=-1)
    
    @staticmethod
    def get_config_space(
        gamma=(0.8, 0.9999)
    ):
        cs = CS.ConfigurationSpace()
        add_hyperparameter(cs, CSH.UniformFloatHyperparameter, 'gamma', gamma)
        return cs

class SchedulerReduceLROnPlateau(AutoNetLearningRateSchedulerBase):
    
    def _get_scheduler(self, optimizer, config):
        return lr_scheduler.ReduceLROnPlateau(optimizer=optimizer)
    
    @staticmethod
    def get_config_space(
        factor=(0.05, 0.5),
        patience=(3, 10)
    ):
        cs = CS.ConfigurationSpace()
        add_hyperparameter(cs, CSH.UniformFloatHyperparameter, 'factor', factor)
        add_hyperparameter(cs, CSH.UniformIntegerHyperparameter, 'patience', patience)
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
    def get_config_space(
        max_factor=(1.0, 2),
        min_factor=(0.001, 1.0),
        cycle_length=(3, 10)
    ):
        cs = CS.ConfigurationSpace()
        add_hyperparameter(cs, CSH.UniformFloatHyperparameter, 'max_factor', max_factor)
        add_hyperparameter(cs, CSH.UniformFloatHyperparameter, 'min_factor', min_factor)
        add_hyperparameter(cs, CSH.UniformIntegerHyperparameter, 'cycle_length', cycle_length)
        return cs

class SchedulerCosineAnnealingWithRestartsLR(AutoNetLearningRateSchedulerBase):

    def _get_scheduler(self, optimizer, config):
        scheduler = CosineAnnealingWithRestartsLR(optimizer, T_max=config['T_max'], T_mult=config['T_mult'],last_epoch=-1)
        scheduler.allows_early_stopping = False
        scheduler.snapshot_before_restart = True
        return scheduler
    
    @staticmethod
    def get_config_space(
        T_max=(1, 20),
        T_mult=(1.0, 2.0)
    ):
        cs = CS.ConfigurationSpace()
        add_hyperparameter(cs, CSH.UniformIntegerHyperparameter, 'T_max', T_max)
        add_hyperparameter(cs, CSH.UniformFloatHyperparameter, 'T_mult', T_mult)
        return cs


class NoScheduling():
    def __init__(self, optimizer):
        self.optimizer = optimizer

    def step(self, epoch):
        return
    
    def get_lr(self):
        try:
            return [self.optimizer.defaults["lr"]]
        except:
            return [None]


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
