
import math
from torch.optim.lr_scheduler import _LRScheduler


class CyclicLR(_LRScheduler):
    def __init__(self, optimizer, max_factor, min_factor, cycle_length, last_epoch=-1):

        self.maf = max_factor
        self.mif = min_factor
        self.cl = cycle_length
        self.r = max_factor - min_factor
        super(CyclicLR, self).__init__(optimizer, last_epoch)

    def lr_mult(self):
        if int(self.last_epoch//self.cl) % 2 == 1:
            lr = self.mif + (self.r * (float(self.last_epoch % self.cl)/float(self.cl)))
        else:
            lr = self.maf - (self.r * (float(self.last_epoch % self.cl)/float(self.cl)))
        return lr

    def get_lr(self):
        return [self.lr_mult() * base_lr for base_lr in self.base_lrs]
