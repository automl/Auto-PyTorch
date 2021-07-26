__author__ = "Max Dippel, Michael Burkart and Matthias Urban"
__version__ = "0.0.1"
__license__ = "BSD"

CSConfig = dict()

# SchedulerStepLR
step_lr = dict()
step_lr['step_size'] = (1, 10)
step_lr['gamma'] = (0.001, 0.9)
CSConfig['step_lr'] = step_lr

# SchedulerExponentialLR
exponential_lr = dict()
exponential_lr['gamma'] = (0.8, 0.9999)
CSConfig['exponential_lr'] = exponential_lr

# SchedulerReduceLROnPlateau
reduce_on_plateau = dict()
reduce_on_plateau['factor'] = (0.05, 0.5)
reduce_on_plateau['patience'] = (3, 10)
CSConfig['reduce_on_plateau'] = reduce_on_plateau

# SchedulerCyclicLR
cyclic_lr = dict()
cyclic_lr['max_factor'] = (1.0, 2)
cyclic_lr['min_factor'] = (0.001, 1.0)
cyclic_lr['cycle_length'] = (3, 10)
CSConfig['cyclic_lr'] = cyclic_lr

# SchedulerCosineAnnealingWithRestartsLR
cosine_annealing_lr = dict()
cosine_annealing_lr['T_max'] = (1, 20)
cosine_annealing_lr['T_mult'] = (1.0, 2.0)
CSConfig['cosine_annealing_with_restarts_lr'] = cosine_annealing_lr


alternating_cosine_lr = dict()
alternating_cosine_lr['T_max'] = (10, 30)
alternating_cosine_lr['T_mult'] = (1.0, 2.0)
alternating_cosine_lr['amp_reduction'] = (0.1, 1)
CSConfig['alternating_cosine_lr'] = alternating_cosine_lr

adaptive_cosine_lr = dict()
adaptive_cosine_lr['T_max'] = (300, 1000)
adaptive_cosine_lr['T_mult'] = (1.0, 2.0)
adaptive_cosine_lr['patience'] = (2, 5)
adaptive_cosine_lr['threshold'] = (0.001, 0.5)
CSConfig['adaptive_cosine_lr'] = adaptive_cosine_lr


custom_lr = dict()
custom_lr['T_max'] = (1, 20)
custom_lr['T_mult'] = (1.0, 2.0)
CSConfig['custom_lr'] = custom_lr
