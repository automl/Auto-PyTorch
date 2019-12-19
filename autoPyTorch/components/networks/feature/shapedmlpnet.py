#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multilayer Perceptrons in fancy shapes.
"""

import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
import torch.nn as nn

from autoPyTorch.utils.config_space_hyperparameter import add_hyperparameter, get_hyperparameter
from autoPyTorch.components.networks.feature.mlpnet import MlpNet

__author__ = "Max Dippel, Michael Burkart and Matthias Urban"
__version__ = "0.0.1"
__license__ = "BSD"

class ShapedMlpNet(MlpNet):
    def __init__(self, *args, **kwargs):
        super(ShapedMlpNet, self).__init__(*args, **kwargs)

    def _build_net(self, in_features, out_features):
        layers = list()
        neuron_counts = get_shaped_neuron_counts(self.config['mlp_shape'],
                                                  in_features,
                                                  out_features,
                                                  self.config['max_units'],
                                                  self.config['num_layers'])
        if self.config["use_dropout"] and self.config["max_dropout"]>0.05:
            dropout_shape = get_shaped_neuron_counts( self.config['mlp_shape'], 0, 0, 1000, self.config['num_layers'])

        previous = in_features
        for i in range(self.config['num_layers']-1):
            if (i >= len(neuron_counts)):
                break
            dropout = dropout_shape[i] / 1000 * self.config["max_dropout"] if (self.config["use_dropout"] and self.config["max_dropout"]>0.05) else 0
            self._add_layer(layers, previous, neuron_counts[i], dropout)
            previous = neuron_counts[i]

        layers.append(nn.Linear(previous, out_features))
        return nn.Sequential(*layers)

    def _add_layer(self, layers, in_features, out_features, dropout):
        layers.append(nn.Linear(in_features, out_features))
        layers.append(self.activation())
        if self.config["use_dropout"] and self.config["max_dropout"]>0.05:
            layers.append(nn.Dropout(dropout))

    @staticmethod

    def get_config_space(
        num_layers=(1, 15),
        max_units=((10, 1024), True),
        activation=('sigmoid', 'tanh', 'relu'),
        mlp_shape=('funnel', 'long_funnel', 'diamond', 'hexagon', 'brick', 'triangle', 'stairs'),
        max_dropout=(0, 0.8),
        use_dropout=(True, False)
    ):
        cs = CS.ConfigurationSpace()
        
        mlp_shape_hp = get_hyperparameter(CSH.CategoricalHyperparameter, 'mlp_shape', mlp_shape)
        cs.add_hyperparameter(mlp_shape_hp)

        num_layers_hp = get_hyperparameter(CSH.UniformIntegerHyperparameter, 'num_layers', num_layers)
        cs.add_hyperparameter(num_layers_hp)
        max_units_hp = get_hyperparameter(CSH.UniformIntegerHyperparameter, "max_units", max_units)
        cs.add_hyperparameter(max_units_hp)

        use_dropout_hp = add_hyperparameter(cs, CS.CategoricalHyperparameter, "use_dropout", use_dropout)

        if True in use_dropout:
            max_dropout_hp = add_hyperparameter(cs, CSH.UniformFloatHyperparameter, "max_dropout", max_dropout)
        
            cs.add_condition(CS.EqualsCondition(max_dropout_hp, use_dropout_hp, True))

        add_hyperparameter(cs, CSH.CategoricalHyperparameter, 'activation', activation)
        return cs
        
        
def get_shaped_neuron_counts(shape, in_feat, out_feat, max_neurons, layer_count):
    counts = []

    if (layer_count <= 0):
        return counts

    if (layer_count == 1):
        counts.append(out_feat)
        return counts

    max_neurons = max(in_feat, max_neurons)
    # https://mikkokotila.github.io/slate/#shapes

    if shape == 'brick':
        #
        #   |        |
        #   |        |
        #   |        |
        #   |        |
        #   |        |
        #   |___  ___|
        #
        for _ in range(layer_count-1):
            counts.append(max_neurons)
        counts.append(out_feat)

    if shape == 'triangle':
        #
        #        /  \
        #       /    \
        #      /      \
        #     /        \
        #    /          \
        #   /_____  _____\
        #
        previous = in_feat
        step_size = int((max_neurons - previous) / (layer_count-1))
        step_size = max(0, step_size)
        for _ in range(layer_count-2):
            previous = previous + step_size
            counts.append(previous)
        counts.append(max_neurons)
        counts.append(out_feat)

    if shape == 'funnel':
        #
        #   \            /
        #    \          /
        #     \        /
        #      \      /
        #       \    /
        #        \  /
        #
        previous = max_neurons
        counts.append(previous)
        
        step_size = int((previous - out_feat) / (layer_count-1))
        step_size = max(0, step_size)
        for _ in range(layer_count-2):
            previous = previous - step_size
            counts.append(previous)

        counts.append(out_feat)

    if shape == 'long_funnel':
        #
        #   |        |
        #   |        |
        #   |        |
        #    \      /
        #     \    /
        #      \  /
        #
        brick_layer = int(layer_count / 2)
        funnel_layer = layer_count - brick_layer
        counts.extend(get_shaped_neuron_counts('brick', in_feat, max_neurons, max_neurons, brick_layer))
        counts.extend(get_shaped_neuron_counts('funnel', in_feat, out_feat, max_neurons, funnel_layer))
        
        if (len(counts) != layer_count):
            print("\nWarning: long funnel layer count does not match " + str(layer_count) + " != " + str(len(counts)) + "\n")
    
    if shape == 'diamond':
        #
        #     /  \
        #    /    \
        #   /      \
        #   \      /
        #    \    /
        #     \  /
        #
        triangle_layer = int(layer_count / 2) + 1
        funnel_layer = layer_count - triangle_layer
        counts.extend(get_shaped_neuron_counts('triangle', in_feat, max_neurons, max_neurons, triangle_layer))
        remove_triangle_layer = len(counts) > 1
        if (remove_triangle_layer):
            counts = counts[0:-2] # remove the last two layers since max_neurons == out_features (-> two layers with the same size)
        counts.extend(get_shaped_neuron_counts('funnel', max_neurons, out_feat, max_neurons, funnel_layer + (2 if remove_triangle_layer else 0)))

        if (len(counts) != layer_count):
            print("\nWarning: diamond layer count does not match " + str(layer_count) + " != " + str(len(counts)) + "\n")

    if shape == 'hexagon':
        #
        #     /  \
        #    /    \
        #   |      |
        #   |      |
        #    \    /
        #     \  /
        #
        triangle_layer = int(layer_count / 3) + 1
        funnel_layer = triangle_layer
        brick_layer = layer_count - triangle_layer - funnel_layer
        counts.extend(get_shaped_neuron_counts('triangle', in_feat, max_neurons, max_neurons, triangle_layer))
        counts.extend(get_shaped_neuron_counts('brick', max_neurons, max_neurons, max_neurons, brick_layer))
        counts.extend(get_shaped_neuron_counts('funnel', max_neurons, out_feat, max_neurons, funnel_layer))

        if (len(counts) != layer_count):
            print("\nWarning: hexagon layer count does not match " + str(layer_count) + " != " + str(len(counts)) + "\n")

    if shape == 'stairs':
        #
        #   |          |
        #   |_        _|
        #     |      |
        #     |_    _|
        #       |  |
        #       |  |
        #
        previous = max_neurons
        counts.append(previous)

        if layer_count % 2 == 1:
            counts.append(previous)

        step_size = 2 * int((max_neurons - out_feat) / (layer_count-1))
        step_size = max(0, step_size)
        for _ in range(int(layer_count / 2 - 1)):
            previous = previous - step_size
            counts.append(previous)
            counts.append(previous)

        counts.append(out_feat)
        
        if (len(counts) != layer_count):
            print("\nWarning: stairs layer count does not match " + str(layer_count) + " != " + str(len(counts)) + "\n")

    return counts
