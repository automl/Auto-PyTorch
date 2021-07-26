#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ResNets in fancy shapes.
"""

from copy import deepcopy

import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH

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
    def get_config_space(user_updates=None):
        cs = CS.ConfigurationSpace()
        range_num_groups=(1, 9)
        range_blocks_per_group=(1, 4)
        range_max_num_units=(10, 1024)
        possible_activations=('sigmoid', 'tanh', 'relu')
        range_max_shake_drop_probability=(0, 1)
        range_max_dropout=(0, 0.8)
        possible_block_shapes=('funnel', 'long_funnel', 'diamond', 'hexagon', 'brick', 'triangle', 'stairs')
        possible_dropout_shapes=('funnel', 'long_funnel', 'diamond', 'hexagon', 'brick', 'triangle', 'stairs')
        
        num_groups = CS.UniformIntegerHyperparameter("num_groups", lower=range_num_groups[0], upper=range_num_groups[1])
        cs.add_hyperparameter(num_groups)
        num_res_blocks = CS.UniformIntegerHyperparameter("blocks_per_group", lower=range_blocks_per_group[0], upper=range_blocks_per_group[1])
        cs.add_hyperparameter(num_res_blocks)
        cs.add_hyperparameter(CS.CategoricalHyperparameter("activation", possible_activations))
        use_dropout = CS.CategoricalHyperparameter("use_dropout", [True, False], default_value=True)
        cs.add_hyperparameter(use_dropout)
        cs.add_hyperparameter(CS.CategoricalHyperparameter("use_shake_shake", [True, False], default_value=True))
        
        shake_drop = cs.add_hyperparameter(CS.CategoricalHyperparameter("use_shake_drop", [True, False], default_value=True))
        shake_drop_prob = cs.add_hyperparameter(CS.UniformFloatHyperparameter("max_shake_drop_probability",
            lower=range_max_shake_drop_probability[0], upper=range_max_shake_drop_probability[1]))
        cs.add_condition(CS.EqualsCondition(shake_drop_prob, shake_drop, True))
        
        resnet_shape = CSH.CategoricalHyperparameter('resnet_shape', possible_block_shapes)
        cs.add_hyperparameter(resnet_shape)
        
        max_units = CSH.UniformIntegerHyperparameter("max_units", lower=range_max_num_units[0], upper=range_max_num_units[1], log=True)
        cs.add_hyperparameter(max_units)

        dropout_shape = CSH.CategoricalHyperparameter('dropout_shape', possible_dropout_shapes)
        cs.add_hyperparameter(dropout_shape)
        max_dropout = CSH.UniformFloatHyperparameter("max_dropout", lower=range_max_dropout[0], upper=range_max_dropout[1])
        cs.add_hyperparameter(max_dropout)
        cs.add_condition(CS.EqualsCondition(dropout_shape, use_dropout, True))
        cs.add_condition(CS.EqualsCondition(max_dropout, use_dropout, True))
        return cs
