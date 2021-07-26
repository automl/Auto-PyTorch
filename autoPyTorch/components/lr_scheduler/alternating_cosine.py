
import math
from torch.optim.lr_scheduler import _LRScheduler


class AlternatingCosineLR(_LRScheduler):
    def __init__(self, optimizer, T_max, T_mul=1, amplitude_reduction=0.9, eta_min=0, last_epoch=-1):
        '''
        Here last_epoch actually means last_step since the
        learning rate is decayed after each batch step.
        '''

        self.T_max = T_max
        self.T_mul = T_mul
        self.eta_min = eta_min
        self.cumulative_time = 0
        self.amplitude_mult = amplitude_reduction
        self.base_lr_mult = 1
        self.frequency_mult = 1
        self.time_offset = 0
        self.last_step = 0
        super(AlternatingCosineLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        '''
        Override this method to the existing get_lr() of the parent class
        '''
        if self.last_epoch >= self.T_max:
            self.T_max = self.T_max * self.T_mul
            self.time_offset = self.T_max / 2
            self.last_epoch = 0
            self.base_lr_mult *= self.amplitude_mult
            self.frequency_mult = 2
            self.cumulative_time = 0
        return [self.eta_min + (base_lr * self.base_lr_mult - self.eta_min) *
                        (1 + math.cos(math.pi * (self.time_offset + self.cumulative_time) / self.T_max * self.frequency_mult)) / 2
                        for base_lr in self.base_lrs]
