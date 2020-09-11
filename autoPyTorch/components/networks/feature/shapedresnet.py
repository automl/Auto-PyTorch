#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ResNets in fancy shapes.
"""

from copy import deepcopy

import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH

from autoPyTorch.utils.config_space_hyperparameter import add_hyperparameter, get_hyperparameter
from autoPyTorch.components.networks.feature.resnet import ResNet
from autoPyTorch.components.networks.feature.shapedmlpnet import get_shaped_neuron_counts

__author__ = "Max Dippel, Michael Burkart and Matthias Urban"
__version__ = "0.0.1"
__license__ = "BSD"

class ShapedResNet(ResNet):
    def __init__(self, config, in_features, out_features, *args, **kwargs):
        augmented_config = deepcopy(config)

        neuron_counts = get_shaped_neuron_counts(config['resnet_shape'],
                                                 in_features,
                                                 out_features,
                                                 config['max_units'],
                                                 config['num_groups']+2)[:-1]
        augmented_config.update(
                {"num_units_%d" % (i) : num for i, num in enumerate(neuron_counts)})
        

        if (config['use_dropout']):
            dropout_shape = get_shaped_neuron_counts(config['dropout_shape'], 0, 0, 1000, config['num_groups'])
            
            dropout_shape = [dropout / 1000 * config["max_dropout"] for dropout in dropout_shape]
        
            augmented_config.update(
                    {"dropout_%d" % (i+1) : dropout for i, dropout in enumerate(dropout_shape)})    

        super(ShapedResNet, self).__init__(augmented_config, in_features, out_features, *args, **kwargs)


    @staticmethod
    def get_config_space(
        num_groups=(1, 9),
        blocks_per_group=(1, 4),
        max_units=((10, 1024), True),
        activation=('sigmoid', 'tanh', 'relu'),
        max_shake_drop_probability=(0, 1),
        max_dropout=(0, 0.8),
        resnet_shape=('funnel', 'long_funnel', 'diamond', 'hexagon', 'brick', 'triangle', 'stairs'),
        dropout_shape=('funnel', 'long_funnel', 'diamond', 'hexagon', 'brick', 'triangle', 'stairs'),
        use_dropout=(True, False),
        use_batch_normalization=(True, False),
        use_skip_connection=(True, False),
        multibranch_choices=('none', 'shake-shake', 'shake-drop'),
    ):
        cs = CS.ConfigurationSpace()
        
        num_groups_hp = get_hyperparameter(CS.UniformIntegerHyperparameter, "num_groups", num_groups)
        cs.add_hyperparameter(num_groups_hp)
        blocks_per_group_hp = get_hyperparameter(CS.UniformIntegerHyperparameter, "blocks_per_group", blocks_per_group)
        cs.add_hyperparameter(blocks_per_group_hp)
        add_hyperparameter(cs, CSH.CategoricalHyperparameter, 'resnet_shape', resnet_shape)
        add_hyperparameter(cs, CSH.UniformIntegerHyperparameter, "max_units", max_units)

        add_hyperparameter(cs, CS.CategoricalHyperparameter, "activation", activation)
        use_dropout_hp = add_hyperparameter(cs, CS.CategoricalHyperparameter, "use_dropout", use_dropout)
        add_hyperparameter(cs, CS.CategoricalHyperparameter, "use_batch_normalization", use_batch_normalization)
        skip_connection_hp = add_hyperparameter(cs, CS.CategoricalHyperparameter, "use_skip_connection", use_skip_connection)

        multi_branch_regularization_hp = get_hyperparameter(
            CS.CategoricalHyperparameter,
            "multi_branch_regularization",
            multibranch_choices,
        )
        skip_connection_status = ShapedResNet.get_boolean_variable_status_from_config(
            use_skip_connection,
        )
        # we are interested in the status of shake drop since it has
        # a child hyperparameter.
        shake_drop_status = 'deactivated'
        if isinstance(multibranch_choices, tuple):
            if isinstance(multibranch_choices[0], list):
                multibranch_restricted_values = multibranch_choices[0]
            else:
                multibranch_restricted_values = multibranch_choices

            if 'shake-drop' in multibranch_restricted_values:
                if len(multibranch_restricted_values) > 1:
                    shake_drop_status = 'conditional'
                else:
                    shake_drop_status = 'active'



        # if skip connection is conditional/activated, add dependent hyperparameters
        if skip_connection_status == 'active' or skip_connection_status == 'conditional':

            cs.add_hyperparameter(multi_branch_regularization_hp)
            # if it is conditional
            if skip_connection_status == 'conditional':
                cs.add_condition(
                    CS.EqualsCondition(
                        multi_branch_regularization_hp,
                        skip_connection_hp,
                        True,
                    )
                )
            if shake_drop_status == 'active' or shake_drop_status =='conditional':
                shake_drop_prob_hp = add_hyperparameter(
                    cs,
                    CS.UniformFloatHyperparameter,
                    "max_shake_drop_probability",
                    max_shake_drop_probability,
                )
                if shake_drop_status == 'conditional':
                    cs.add_condition(
                        CS.EqualsCondition(
                            shake_drop_prob_hp,
                            multi_branch_regularization_hp,
                            'shake-drop',
                        )
                    )

        dropout_status = ShapedResNet.get_boolean_variable_status_from_config(
            use_dropout,
        )

        if dropout_status == 'active' or dropout_status == 'conditional':

            dropout_shape_hp = add_hyperparameter(
                cs,
                CSH.CategoricalHyperparameter,
                'dropout_shape',
                dropout_shape,
            )
            max_dropout_hp = add_hyperparameter(
                cs,
                CSH.UniformFloatHyperparameter,
                'max_dropout',
                max_dropout
            )
            if dropout_status == 'conditional':
                cs.add_condition(
                    CS.EqualsCondition(
                        dropout_shape_hp,
                        use_dropout_hp,
                        True
                    )
                )
                cs.add_condition(
                    CS.EqualsCondition(
                        max_dropout_hp,
                        use_dropout_hp,
                        True
                    )
                )

        return cs

    @staticmethod
    def get_boolean_variable_status_from_config(variable_config):

        variable_status = 'deactivated'
        if isinstance(variable_config, tuple):
            # multiple values are given
            if isinstance(variable_config[0], list):
                # set by the searchspace update
                predefined_values = variable_config[0]
                if len(predefined_values) > 1:
                    variable_status = 'conditional'
                else:
                    if predefined_values[0]:
                        variable_status = 'active'
            else:
                variable_status = 'conditional'
        else:
            if isinstance(variable_config, bool):
                if variable_config:
                    variable_status = 'active'

        return variable_status
