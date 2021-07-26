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
        return lr_scheduler.ReduceLROnPlateau(  optimizer=optimizer, 
                                                factor=config['factor'], 
                                                patience=config['patience'])
    
    @staticmethod
    def get_config_space(*args, **kwargs):
        cs = CS.ConfigurationSpace()
        config = CSConfig['reduce_on_plateau']
        cs.add_hyperparameter(CSH.UniformFloatHyperparameter('factor', lower=config['factor'][0], upper=config['factor'][1]))
        cs.add_hyperparameter(CSH.UniformIntegerHyperparameter('patience', lower=config['patience'][0], upper=config['patience'][1]))
        cs.add_configuration_space(prefix='', delimiter='', configuration_space=AutoNetLearningRateSchedulerBase.get_config_space(*args, **kwargs))
        return cs


class SchedulerAdaptiveLR(AutoNetLearningRateSchedulerBase):
    
    def _get_scheduler(self, optimizer, config):
        from .adaptive_learningrate import AdaptiveLR
        return AdaptiveLR(optimizer=optimizer, T_max=config['T_max'], T_mul=config['T_mult'], patience=config['patience'], threshold=config['threshold'])
    
    @staticmethod
    def get_config_space(*args, **kwargs):
        cs = CS.ConfigurationSpace()
        config = CSConfig['adaptive_cosine_lr'] 
        cs.add_hyperparameter(CSH.UniformIntegerHyperparameter('T_max', lower=config['T_max'][0], upper=config['T_max'][1]))
        cs.add_hyperparameter(CSH.UniformIntegerHyperparameter('patience', lower=config['patience'][0], upper=config['patience'][1]))
        cs.add_hyperparameter(CSH.UniformFloatHyperparameter('T_mult', lower=config['T_mult'][0], upper=config['T_mult'][1]))
        cs.add_hyperparameter(CSH.UniformFloatHyperparameter('threshold', lower=config['threshold'][0], upper=config['threshold'][1]))
        cs.add_configuration_space(prefix='', delimiter='', configuration_space=AutoNetLearningRateSchedulerBase.get_config_space(*args, **kwargs))
        return cs

class SchedulerCyclicLR(AutoNetLearningRateSchedulerBase):

    def _get_scheduler(self, optimizer, config):
        from autoPyTorch.components.lr_scheduler.cyclic import CyclicLR
        return CyclicLR(optimizer=optimizer, 
                        min_factor=config['min_factor'], 
                        max_factor=config['max_factor'], 
                        cycle_length=config['cycle_length'], 
                        last_epoch=-1)
    
    @staticmethod
    def get_config_space(*args, **kwargs):
        cs = CS.ConfigurationSpace()
        config = CSConfig['cyclic_lr']
        cs.add_hyperparameter(CSH.UniformFloatHyperparameter('max_factor', lower=config['max_factor'][0], upper=config['max_factor'][1]))
        cs.add_hyperparameter(CSH.UniformFloatHyperparameter('min_factor', lower=config['min_factor'][0], upper=config['min_factor'][1]))
        cs.add_hyperparameter(CSH.UniformIntegerHyperparameter('cycle_length', lower=config['cycle_length'][0], upper=config['cycle_length'][1]))
        cs.add_configuration_space(prefix='', delimiter='', configuration_space=AutoNetLearningRateSchedulerBase.get_config_space(*args, **kwargs))
        return cs

class SchedulerCosineAnnealingLR(AutoNetLearningRateSchedulerBase):

    def _get_scheduler(self, optimizer, config):
        return lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=config['T_max'], eta_min=config['eta_min'], last_epoch=-1)

    @staticmethod
    def get_config_space(*args, **kwargs):
        cs = CS.ConfigurationSpace()
        cs.add_hyperparameter(CSH.UniformIntegerHyperparameter('T_max', lower=10, upper=1000))
        cs.add_hyperparameter(CSH.UniformFloatHyperparameter('eta_min', lower=0.0, upper=0.01))
        cs.add_configuration_space(prefix='', delimiter='', configuration_space=AutoNetLearningRateSchedulerBase.get_config_space(*args, **kwargs))
        return cs


class SchedulerCosineAnnealingWithRestartsLR(AutoNetLearningRateSchedulerBase):

    def _get_scheduler(self, optimizer, config):
        scheduler = CosineAnnealingWithRestartsLR(optimizer, T_max=config['T_max'], T_mult=config['T_mult'], last_epoch=-1)
        scheduler.allows_early_stopping = False
        scheduler.snapshot_before_restart = True
        return scheduler
    
    @staticmethod
    def get_config_space(*args, **kwargs):
        cs = CS.ConfigurationSpace()
        config = CSConfig['cosine_annealing_with_restarts_lr']
        cs.add_hyperparameter(CSH.UniformIntegerHyperparameter('T_max', lower=config['T_max'][0], upper=config['T_max'][1]))
        cs.add_hyperparameter(CSH.UniformFloatHyperparameter('T_mult', lower=config['T_mult'][0], upper=config['T_mult'][1]))
        cs.add_configuration_space(prefix='', delimiter='', configuration_space=AutoNetLearningRateSchedulerBase.get_config_space(*args, **kwargs))
        return cs


class NoScheduling():

    def step(self, epoch=None):
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

    def needs_checkpoint(self):
        return self.step_n + 1 >= self.restart_every


class SchedulerAlternatingCosineLR(AutoNetLearningRateSchedulerBase):
    
    def _get_scheduler(self, optimizer, config):
        from .alternating_cosine import AlternatingCosineLR
        scheduler = AlternatingCosineLR(optimizer, T_max=config['T_max'], T_mul=config['T_mult'], amplitude_reduction=config['amp_reduction'], last_epoch=-1)
        return scheduler
    
    @staticmethod
    def get_config_space(*args, **kwargs):
        cs = CS.ConfigurationSpace()
        config = CSConfig['alternating_cosine_lr']
        cs.add_hyperparameter(CSH.UniformIntegerHyperparameter('T_max', lower=config['T_max'][0], upper=config['T_max'][1]))
        cs.add_hyperparameter(CSH.UniformFloatHyperparameter('T_mult', lower=config['T_mult'][0], upper=config['T_mult'][1]))
        cs.add_hyperparameter(CSH.UniformFloatHyperparameter('amp_reduction', lower=config['amp_reduction'][0], upper=config['amp_reduction'][1]))
        cs.add_configuration_space(prefix='', delimiter='', configuration_space=AutoNetLearningRateSchedulerBase.get_config_space(*args, **kwargs))
        return cs

class SchedulerCustomLR(AutoNetLearningRateSchedulerBase):
    
    def _get_scheduler(self, optimizer, config):
        scheduler = CustomScheduler(optimizer, T_max=config['T_max'], T_mult=config['T_mult'], min_loss_increase=0.0001, loss_array_size=100, last_epoch=-1)
        scheduler.allows_early_stopping = False
        scheduler.snapshot_before_restart = True
        return scheduler
    
    @staticmethod
    def get_config_space(*args, **kwargs):
        cs = CS.ConfigurationSpace()
        config = CSConfig['custom_lr']
        cs.add_hyperparameter(CSH.UniformIntegerHyperparameter('T_max', lower=config['T_max'][0], upper=config['T_max'][1]))
        cs.add_hyperparameter(CSH.UniformFloatHyperparameter('T_mult', lower=config['T_mult'][0], upper=config['T_mult'][1]))
        cs.add_configuration_space(prefix='', delimiter='', configuration_space=AutoNetLearningRateSchedulerBase.get_config_space(*args, **kwargs))
        return cs


import numpy as np
from queue import Queue

# class CustomScheduler(torch.optim.lr_scheduler._LRScheduler):
    
#     def __init__(self, optimizer, T_max, T_mult, min_loss_increase, loss_array_size, amplitude_reduction=0.9, eta_min=0, last_epoch=-1):
#         self.T_max = T_max
#         self.T_mult = T_mult
#         self.restart_every = T_max
#         self.eta_min = eta_min
#         self.restarts = 0
#         self.restarted_at = 0
#         self.losses = []
#         self.base_lr_mult = 1
#         self.losses_max_length = loss_array_size
#         self.next_amplitude_reduction = amplitude_reduction
#         self.max_amplitude_reduction = amplitude_reduction
#         self.min_loss_increase = min_loss_increase
#         super(CustomScheduler, self).__init__(optimizer, last_epoch)
    
#     def restart(self):
#         self.restart_every *= self.T_mult
#         self.restarted_at = self.last_epoch
    
#     def cosine(self, base_lr):
#         return base_lr * self.base_lr_mult
    
#     def step(self, epoch=None, loss=float('inf')):

#         if len(self.losses) >= self.losses_max_length:
#             avg = np.sum(self.losses) / len(self.losses)
#             if avg - loss < self.min_loss_increase:
#                 self.base_lr_mult *= self.max_amplitude_reduction

#                 self.losses = []
#                 if self.base_lr_mult < 0.01:
#                     self.base_lr_mult = 1

#                     print('restart')

#                 print('descrease step', loss, ' - ', avg, 'mult', self.base_lr_mult)
#             else:
#                 self.losses = self.losses[1:]

#         self.losses.append(loss)
#         super(CustomScheduler, self).step(epoch=epoch)

#     # @property
#     # def step_n(self):
#     #     return self.last_epoch - self.restarted_at

#     def get_lr(self):
#         # if self.step_n >= self.restart_every:
#         #     self.restart_every *= self.T_mult
#         #     self.restarted_at = self.last_epoch + self.restart_every

#         #     self.base_lr_mult *= self.next_amplitude_reduction
#         #     self.next_amplitude_reduction = self.max_amplitude_reduction
            
#         #     self.losses = []
#         #     print('half_restart, amp:', self.restart_every, ' - ', self.base_lr_mult)

#         return [base_lr * self.base_lr_mult for base_lr in self.base_lrs]  


class CustomScheduler(torch.optim.lr_scheduler._LRScheduler):
    
    def __init__(self, optimizer, T_max, T_mult, min_loss_increase, loss_array_size, amplitude_reduction=0.9, eta_min=0, last_epoch=-1):
        self.T_max = T_max
        self.T_mult = T_mult
        self.restart_every = T_max
        self.eta_min = eta_min
        self.restarts = 0
        self.restarted_at = 0
        self.losses = []
        self.base_lr_mult = 1
        self.losses_max_length = loss_array_size
        self.next_amplitude_reduction = amplitude_reduction
        self.max_amplitude_reduction = amplitude_reduction
        self.min_loss_increase = min_loss_increase
        super(CustomScheduler, self).__init__(optimizer, last_epoch)
    
    def restart(self):
        self.restart_every *= self.T_mult
        self.restarted_at = self.last_epoch
    
    def cosine(self, base_lr):
        return self.eta_min + (base_lr * self.base_lr_mult - self.eta_min) * (1 + math.cos(math.pi * self.step_n / self.restart_every)) / 2
    
    def step(self, epoch=None, loss=float('inf')):

        if len(self.losses) >= self.losses_max_length:
            avg = np.sum(self.losses) / len(self.losses)
            if avg - loss < self.min_loss_increase:

                self.restarted_at = self.last_epoch
                self.losses = []

                if self.base_lr_mult < self.next_amplitude_reduction:
                    self.next_amplitude_reduction = self.base_lr_mult #(1 + self.base_lr_mult) / 2

                self.next_amplitude_reduction *= self.max_amplitude_reduction

                self.base_lr_mult = 1
                # print('restart', loss, ' - ', avg)
            else:
                #self.restart_every += 1

                self.losses = self.losses[1:]

        self.losses.append(loss)
        super(CustomScheduler, self).step(epoch=epoch)

    @property
    def step_n(self):
        return self.last_epoch - self.restarted_at

    def get_lr(self):
        if self.step_n >= self.restart_every:
            self.restart_every *= self.T_mult
            self.restarted_at = self.last_epoch + self.restart_every

            self.base_lr_mult *= self.next_amplitude_reduction
            self.next_amplitude_reduction = self.max_amplitude_reduction
            
            self.losses = []
            # print('half_restart, amp:', self.restart_every, ' - ', self.base_lr_mult)

        return [self.cosine(base_lr) for base_lr in self.base_lrs]  
