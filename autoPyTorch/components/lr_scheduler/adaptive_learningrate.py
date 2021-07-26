import math
from torch.optim import Optimizer

class AdaptiveLR(object):

    def __init__(self, optimizer, mode='min', T_max=30, T_mul=2.0, eta_min=0, patience=3, threshold=0.1, min_lr=0, eps=1e-8, last_epoch=-1):

        if not isinstance(optimizer, Optimizer):
            raise TypeError('{} is not an Optimizer'.format(
                type(optimizer).__name__))
        self.optimizer = optimizer
        if last_epoch == -1:
            for group in optimizer.param_groups:
                group.setdefault('initial_lr', group['lr'])
        else:
            for i, group in enumerate(optimizer.param_groups):
                if 'initial_lr' not in group:
                    raise KeyError("param 'initial_lr' is not specified "
                                   "in param_groups[{}] when resuming an optimizer".format(i))
        self.base_lrs = list(map(lambda group: group['initial_lr'], optimizer.param_groups))
        self.last_epoch = last_epoch

        if isinstance(min_lr, list) or isinstance(min_lr, tuple):
            if len(min_lr) != len(optimizer.param_groups):
                raise ValueError("expected {} min_lrs, got {}".format(
                    len(optimizer.param_groups), len(min_lr)))
            self.min_lrs = list(min_lr)
        else:
            self.min_lrs = [min_lr] * len(optimizer.param_groups)

        self.T_max = T_max
        self.T_mul = T_mul
        self.eta_min = eta_min
        self.current_base_lrs = self.base_lrs
        self.metric_values = []
        self.threshold = threshold
        self.patience = patience
        self.steps = 0
        

    def step(self, metrics, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch
        
        self.metric_values.append(metrics)
        if len(self.metric_values) > self.patience:
            self.metric_values = self.metric_values[1:]

        if max(self.metric_values) - metrics > self.threshold:
            self.current_base_lrs = self.get_lr()
            self.steps = 0
        else:
            self.steps += 1
        
        self.last_metric_value = metrics

        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr

    def get_lr(self):
        '''
        Override this method to the existing get_lr() of the parent class
        '''
        if self.steps >= self.T_max:
            self.T_max = self.T_max * self.T_mul
            self.current_base_lrs = self.base_lrs
            self.metric_values = []
            self.steps = 0

        return [self.eta_min + (base_lr - self.eta_min) *
                        (1 + math.cos(math.pi * self.steps / self.T_max)) / 2
                        for base_lr in self.current_base_lrs]                    
