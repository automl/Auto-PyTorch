"""
File which contains the optimizers.
"""
import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
import torch.optim as optim

from autoPyTorch.utils.config_space_hyperparameter import add_hyperparameter

__author__ = "Max Dippel, Michael Burkart and Matthias Urban"
__version__ = "0.0.1"
__license__ = "BSD"


class AutoNetOptimizerBase(object):
    def __new__(cls, params, config):
        return cls._get_optimizer(cls, params, config)

    def _get_optimizer(self, params, config):
        raise ValueError('Override the method _get_optimizer and do not call the base class implementation')

    @staticmethod
    def get_config_space(*args, **kwargs):
        return CS.ConfigurationSpace()


class AdamOptimizer(AutoNetOptimizerBase):

    def _get_optimizer(self, params, config):
        if config['use_weight_decay']:
            weight_decay = config['weight_decay']
        else:
            weight_decay = 0

        return optim.Adam(params=params, lr=config['learning_rate'], weight_decay=weight_decay)

    @staticmethod
    def get_config_space(
            learning_rate=((1e-4, 0.1), True),
            weight_decay=(1e-5, 0.1),
            use_weight_decay=(True, False),
    ):
        cs = CS.ConfigurationSpace()
        add_hyperparameter(cs, CSH.UniformFloatHyperparameter, 'learning_rate', learning_rate)
        weight_decay_activation = add_hyperparameter(
            cs,
            CSH.CategoricalHyperparameter,
            'use_weight_decay',
            use_weight_decay
        )
        if True in use_weight_decay:
            weight_decay_value = add_hyperparameter(
                cs,
                CSH.UniformFloatHyperparameter,
                'weight_decay',
                weight_decay
            )

            cs.add_condition(CS.EqualsCondition(weight_decay_value, weight_decay_activation, True))

        return cs


class AdamWOptimizer(AutoNetOptimizerBase):

    def _get_optimizer(self, params, config):
        if config['use_weight_decay']:
            weight_decay = config['weight_decay']
        else:
            weight_decay = 0

        return optim.AdamW(params=params, lr=config['learning_rate'], weight_decay=weight_decay)

    @staticmethod
    def get_config_space(
            learning_rate=((1e-4, 0.1), True),
            use_weight_decay=(True, False),
            weight_decay=(1e-5, 0.1),
    ):
        cs = CS.ConfigurationSpace()
        add_hyperparameter(cs, CSH.UniformFloatHyperparameter, 'learning_rate', learning_rate)
        weight_decay_activation = add_hyperparameter(
            cs,
            CSH.CategoricalHyperparameter,
            'use_weight_decay',
            use_weight_decay
        )
        if True in use_weight_decay:
            weight_decay_value = add_hyperparameter(
                cs,
                CSH.UniformFloatHyperparameter,
                'weight_decay',
                weight_decay
            )

            cs.add_condition(CS.EqualsCondition(weight_decay_value, weight_decay_activation, True))

        return cs


class SgdOptimizer(AutoNetOptimizerBase):

    def _get_optimizer(self, params, config):
        if config['use_weight_decay']:
            weight_decay = config['weight_decay']
        else:
            weight_decay = 0

        return optim.SGD(params=params, lr=config['learning_rate'], momentum=config['momentum'],
                         weight_decay=weight_decay)

    @staticmethod
    def get_config_space(
            learning_rate=((1e-4, 0.1), True),
            momentum=((0.1, 0.99), True),
            use_weight_decay=(True, False),
            weight_decay=(1e-5, 0.1),
    ):
        cs = CS.ConfigurationSpace()
        add_hyperparameter(cs, CSH.UniformFloatHyperparameter, 'learning_rate', learning_rate)
        add_hyperparameter(cs, CSH.UniformFloatHyperparameter, 'momentum', momentum)
        weight_decay_activation = add_hyperparameter(
            cs,
            CSH.CategoricalHyperparameter,
            'use_weight_decay',
            use_weight_decay
        )
        if True in use_weight_decay:
            weight_decay_value = add_hyperparameter(
                cs,
                CSH.UniformFloatHyperparameter,
                'weight_decay',
                weight_decay
            )

            cs.add_condition(CS.EqualsCondition(weight_decay_value, weight_decay_activation, True))

        return cs


class RMSpropOptimizer(AutoNetOptimizerBase):

    def _get_optimizer(self, params, config):
        if config['use_weight_decay']:
            weight_decay = config['weight_decay']
        else:
            weight_decay = 0

        return optim.RMSprop(params=params, lr=config['learning_rate'], momentum=config['momentum'],
                             weight_decay=weight_decay, centered=False)

    @staticmethod
    def get_config_space(
            learning_rate=((1e-4, 0.1), True),
            momentum=((0.1, 0.99), True),
            use_weight_decay=(True, False),
            weight_decay=(1e-5, 0.1),
            alpha=(0.1, 0.99),
    ):
        cs = CS.ConfigurationSpace()
        add_hyperparameter(cs, CSH.UniformFloatHyperparameter, 'learning_rate', learning_rate)
        add_hyperparameter(cs, CSH.UniformFloatHyperparameter, 'momentum', momentum)
        weight_decay_activation = add_hyperparameter(
            cs,
            CSH.CategoricalHyperparameter,
            'use_weight_decay',
            use_weight_decay
        )
        if True in use_weight_decay:
            weight_decay_value = add_hyperparameter(
                cs,
                CSH.UniformFloatHyperparameter,
                'weight_decay',
                weight_decay
            )

            cs.add_condition(CS.EqualsCondition(weight_decay_value, weight_decay_activation, True))

        add_hyperparameter(cs, CSH.UniformFloatHyperparameter, 'alpha', alpha)
        return cs
