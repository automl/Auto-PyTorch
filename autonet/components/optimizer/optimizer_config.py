__author__ = "Max Dippel, Michael Burkart and Matthias Urban"
__version__ = "0.0.1"
__license__ = "BSD"

CSConfig = dict()

# AdamOptimizer
adam_opt = dict()
adam_opt['learning_rate'] = (0.0001, 0.1)
adam_opt['weight_decay'] = (0.0001, 0.1)
CSConfig['adam_opt'] = adam_opt

# SgdOptimizer
sgd_opt = dict()
sgd_opt['learning_rate'] = (0.0001, 0.1)
sgd_opt['weight_decay'] = (0.0001, 0.1)
sgd_opt['momentum'] = (0.1, 0.9)
CSConfig['sgd_opt'] = sgd_opt
