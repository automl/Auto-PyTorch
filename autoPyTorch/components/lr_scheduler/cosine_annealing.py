'''
This cosine decay scheduler will decay the learning rate using a cosine function
given a time limit for training. One decay step is performed after each batch iteration
and the learning rate is assured to become eta_min at each T_max,
'''

import math
from torch.optim.lr_scheduler import _LRScheduler


class CosineAnnealingRestartsLR(_LRScheduler):
    def __init__(self, optimizer, T_max, T_mul=1, eta_min=0, last_epoch=-1):
        '''
        Here last_epoch actually means last_step since the
        learning rate is decayed after each batch step.
        '''

        self.T_max = T_max
        self.T_mul = T_mul
        self.eta_min = eta_min
        self.cumulative_time = 0
        self.last_step = 0
        super(CosineAnnealingRestartsLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        '''
        Override this method to the existing get_lr() of the parent class
        '''
        if self.last_epoch >= self.T_max:
            self.T_max = self.T_max * self.T_mul
            self.last_epoch = 0
            self.cumulative_time = 0
        return [self.eta_min + (base_lr - self.eta_min) *
                        (1 + math.cos(math.pi * self.cumulative_time / self.T_max)) / 2
                        for base_lr in self.base_lrs]

    def needs_checkpoint(self):
        return (self.cumulative_time > self.T_max + self.cumulative_time - self.last_step + 5) and (self.last_step < 0)
