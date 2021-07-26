__author__ = "Max Dippel, Michael Burkart and Matthias Urban"
__version__ = "0.0.1"
__license__ = "BSD"

CSConfig = dict()

# AdamOptimizer
adam_opt = dict()
adam_opt['learning_rate'] = (1e-3, 0.5)
adam_opt['weight_decay'] = (1e-5, 1e-3)
CSConfig['adam_opt'] = adam_opt

# AdamWOptimizer
adamw_opt = dict()
adamw_opt['learning_rate'] = (1e-3, 0.5)
adamw_opt['weight_decay'] = (1e-5, 1e-3)
CSConfig['adamw_opt'] = adamw_opt

# SgdOptimizer
sgd_opt = dict()
sgd_opt['learning_rate'] = (1e-3, 0.5)
sgd_opt['weight_decay'] = (1e-5, 1e-3)
sgd_opt['momentum'] = (1e-3, 0.99)
CSConfig['sgd_opt'] = sgd_opt

# RMSprop
rmsprop_opt = dict()
rmsprop_opt['learning_rate'] = (1e-3, 0.5)
rmsprop_opt['weight_decay'] = (1e-5, 1e-3)
rmsprop_opt['momentum'] = (1e-3, 0.99)
rmsprop_opt['alpha'] = (0.8, 0.999)
CSConfig['rmsprop_opt'] = rmsprop_opt
