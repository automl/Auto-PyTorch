#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Basic Implementation of a multi layer perceptron.
"""

from __future__ import division, print_function

import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
import torch.nn as nn

from autoPyTorch.components.networks.base_net import BaseFeatureNet

__author__ = "Max Dippel, Michael Burkart and Matthias Urban"
__version__ = "0.0.1"
__license__ = "BSD"

class MlpNet(BaseFeatureNet):
    activations = {
        'relu': nn.ReLU,
        'sigmoid': nn.Sigmoid,
        'tanh': nn.Tanh
    }


    def __init__(self, config, in_features, out_features, embedding, final_activation=None):
        super(MlpNet, self).__init__(config, in_features, out_features, embedding, final_activation)
        self.activation = self.activations[config['activation']]
        self.layers = self._build_net(self.n_feats, self.n_classes)

    def _build_net(self, in_features, out_features):
        layers = list()
        self._add_layer(layers, in_features, self.config["num_units_1"], 1)

        for i in range(2, self.config["num_layers"] + 1):
            self._add_layer(layers, self.config["num_units_%d" % (i-1)], self.config["num_units_%d" % i], i)

        layers.append(nn.Linear(self.config["num_units_%d" % self.config["num_layers"]], out_features))
        return nn.Sequential(*layers)

    def _add_layer(self, layers, in_features, out_features, layer_id):
        layers.append(nn.Linear(in_features, out_features))
        layers.append(self.activation())
        if self.config["use_dropout"]:
            layers.append(nn.Dropout(self.config["dropout_%d" % layer_id]))

    @staticmethod
    def get_config_space(user_updates=None):
        cs = CS.ConfigurationSpace()
        range_num_layers=(1, 15)
        range_num_units=(10, 1024)
        possible_activations=('sigmoid', 'tanh', 'relu')
        range_dropout=(0.0, 0.8)
        
        if user_updates is not None and 'num_layers' in user_updates:
            range_num_layers = user_updates['num_layers']

        num_layers = CSH.UniformIntegerHyperparameter('num_layers', lower=range_num_layers[0], upper=range_num_layers[1])
        cs.add_hyperparameter(num_layers)
        use_dropout = cs.add_hyperparameter(CS.CategoricalHyperparameter("use_dropout", [True, False], default_value=True))

        for i in range(1, range_num_layers[1] + 1):
            n_units = CSH.UniformIntegerHyperparameter("num_units_%d" % i,
                lower=range_num_units[0], upper=range_num_units[1], log=True)
            cs.add_hyperparameter(n_units)
            dropout = CSH.UniformFloatHyperparameter("dropout_%d" % i, lower=range_dropout[0], upper=range_dropout[1])
            cs.add_hyperparameter(dropout)
            dropout_condition_1 = CS.EqualsCondition(dropout, use_dropout, True)

            if i > range_num_layers[0]:
                cs.add_condition(CS.GreaterThanCondition(n_units, num_layers, i - 1))

                dropout_condition_2 = CS.GreaterThanCondition(dropout, num_layers, i - 1)
                cs.add_condition(CS.AndConjunction(dropout_condition_1, dropout_condition_2))
            else:
                cs.add_condition(dropout_condition_1)
        
        cs.add_hyperparameter(CSH.CategoricalHyperparameter('activation', possible_activations))
        return(cs)