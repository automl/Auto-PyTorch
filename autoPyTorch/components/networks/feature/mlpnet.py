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
from autoPyTorch.utils.config_space_hyperparameter import add_hyperparameter, get_hyperparameter

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
    def get_config_space(
        num_layers=((1, 15), False),
        num_units=((10, 1024), True),
        activation=('sigmoid', 'tanh', 'relu'),
        dropout=(0.0, 0.8),
        use_dropout=(True, False),
        **kwargs
    ):
        cs = CS.ConfigurationSpace()

        num_layers_hp = get_hyperparameter(CSH.UniformIntegerHyperparameter, 'num_layers', num_layers)
        cs.add_hyperparameter(num_layers_hp)
        use_dropout_hp = add_hyperparameter(cs, CS.CategoricalHyperparameter, "use_dropout", use_dropout)

        for i in range(1, num_layers[0][1] + 1):
            n_units_hp = get_hyperparameter(CSH.UniformIntegerHyperparameter, "num_units_%d" % i, kwargs.pop("num_units_%d" % i, num_units))
            cs.add_hyperparameter(n_units_hp)

            if i > num_layers[0][0]:
                cs.add_condition(CS.GreaterThanCondition(n_units_hp, num_layers_hp, i - 1))

            if True in use_dropout:
                dropout_hp = get_hyperparameter(CSH.UniformFloatHyperparameter, "dropout_%d" % i, kwargs.pop("dropout_%d" % i, dropout))
                cs.add_hyperparameter(dropout_hp)
                dropout_condition_1 = CS.EqualsCondition(dropout_hp, use_dropout_hp, True)

                if i > num_layers[0][0]:
                    dropout_condition_2 = CS.GreaterThanCondition(dropout_hp, num_layers_hp, i - 1)
                    cs.add_condition(CS.AndConjunction(dropout_condition_1, dropout_condition_2))
                else:
                    cs.add_condition(dropout_condition_1)
        
        add_hyperparameter(cs, CSH.CategoricalHyperparameter,'activation', activation)
        assert len(kwargs) == 0, "Invalid hyperparameter updates for mlpnet: %s" % str(kwargs)
        return(cs)