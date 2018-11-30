#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Imlementation of a ResNet with feature data.
"""

import ConfigSpace
import torch.nn as nn

from autoPyTorch.components.networks.base_net import BaseFeatureNet
from autoPyTorch.components.regularization.shake import (shake_drop,
                                                     shake_drop_get_bl,
                                                     shake_get_alpha_beta,
                                                     shake_shake)

__author__ = "Max Dippel, Michael Burkart and Matthias Urban"
__version__ = "0.0.1"
__license__ = "BSD"

class ResNet(BaseFeatureNet):
    activations = {
        'relu': nn.ReLU,
        'sigmoid': nn.Sigmoid,
        'tanh': nn.Tanh
    }
    
    def __init__(self, config, in_features, out_features, embedding, final_activation=None):
        super(ResNet, self).__init__(config, in_features, out_features, embedding, final_activation)
        self.activation = self.activations[config['activation']]
        self.layers = self._build_net(self.n_feats, self.n_classes)
        
    def _build_net(self, in_features, out_features):
        
        layers = list()
        layers.append(nn.Linear(in_features, self.config["num_units_0"]))

        # build num_groups-1 groups each consisting of blocks_per_group ResBlocks
        # the output features of each group is defined by num_units_i
        for i in range(1, self.config["num_groups"] + 1):
            layers.append(self._add_group(  in_features=self.config["num_units_%d" % (i-1)], 
                                            out_features=self.config["num_units_%d" % i], 
                                            last_block_index=(i-1) * self.config["blocks_per_group"], 
                                            dropout=self.config["use_dropout"] and self.config["dropout_%d" % i]))

        layers.append(nn.BatchNorm1d(self.config["num_units_%i" % self.config["num_groups"]]))
        layers.append(self.activation())

        layers.append(nn.Linear(self.config["num_units_%i" % self.config["num_groups"]], out_features))
        return nn.Sequential(*layers)
        
    # Stacking Residual Blocks on the same stage
    def _add_group(self, in_features, out_features, last_block_index, dropout):
        blocks = list()
        blocks.append(ResBlock(self.config, in_features, out_features, last_block_index, dropout, self.activation))
        for i in range(1, self.config["blocks_per_group"]):
            blocks.append(ResBlock(self.config, out_features, out_features, last_block_index+i, dropout, self.activation))
        return nn.Sequential(*blocks)

    @staticmethod
    def get_config_space(user_updates=None):
        cs = ConfigSpace.ConfigurationSpace()
        range_num_groups=(1, 9)
        range_blocks_per_group=(1, 4)
        range_num_units=(10, 1024)
        possible_activations=('sigmoid', 'tanh', 'relu')
        range_max_shake_drop_probability=(0, 1)
        range_dropout=(0, 0.8)

        if user_updates is not None and 'num_groups' in user_updates:
            range_num_groups = user_updates['num_groups']
        if user_updates is not None and 'blocks_per_group' in user_updates:
            range_blocks_per_group = user_updates['blocks_per_group']

        num_groups = ConfigSpace.UniformIntegerHyperparameter("num_groups", lower=range_num_groups[0], upper=range_num_groups[1])
        cs.add_hyperparameter(num_groups)
        num_res_blocks = ConfigSpace.UniformIntegerHyperparameter("blocks_per_group", lower=range_blocks_per_group[0], upper=range_blocks_per_group[1])
        cs.add_hyperparameter(num_res_blocks)
        cs.add_hyperparameter(ConfigSpace.CategoricalHyperparameter("activation", possible_activations))
        
        use_dropout = ConfigSpace.CategoricalHyperparameter("use_dropout", [True, False], default_value=True)
        cs.add_hyperparameter(use_dropout)
        cs.add_hyperparameter(ConfigSpace.CategoricalHyperparameter("use_shake_shake", [True, False], default_value=True))
        
        shake_drop = cs.add_hyperparameter(ConfigSpace.CategoricalHyperparameter("use_shake_drop", [True, False], default_value=True))
        shake_drop_prob = cs.add_hyperparameter(ConfigSpace.UniformFloatHyperparameter("max_shake_drop_probability",
            lower=range_max_shake_drop_probability[0], upper=range_max_shake_drop_probability[1]))
        cs.add_condition(ConfigSpace.EqualsCondition(shake_drop_prob, shake_drop, True))
        

        # it is the upper bound of the nr of groups, since the configuration will actually be sampled.
        for i in range(0, range_num_groups[1] + 1):

            n_units = ConfigSpace.UniformIntegerHyperparameter("num_units_%d" % i, lower=range_num_units[0], upper=range_num_units[1], log=True)
            cs.add_hyperparameter(n_units)
            dropout = ConfigSpace.UniformFloatHyperparameter("dropout_%d" % i, lower=range_dropout[0], upper=range_dropout[1])
            cs.add_hyperparameter(dropout)
            dropout_condition_1 = ConfigSpace.EqualsCondition(dropout, use_dropout, True)

            if i > 1:
                cs.add_condition(ConfigSpace.GreaterThanCondition(n_units, num_groups, i - 1))
                
                dropout_condition_2 = ConfigSpace.GreaterThanCondition(dropout, num_groups, i - 1)
                cs.add_condition(ConfigSpace.AndConjunction(dropout_condition_1, dropout_condition_2))
            else:
                cs.add_condition(dropout_condition_1)

        return cs


class ResBlock(nn.Module):
    
    def __init__(self, config, in_features, out_features, block_index, dropout, activation):
        super(ResBlock, self).__init__()
        self.config = config
        self.dropout = dropout
        self.activation = activation

        self.shortcut = None
        self.start_norm = None

        # if in != out the shortcut needs a linear layer to match the result dimensions
        # if the shortcut needs a layer we apply batchnorm and activation to the shortcut as well (start_norm)
        if in_features != out_features:
            self.shortcut = nn.Linear(in_features, out_features)
            self.start_norm = nn.Sequential(nn.BatchNorm1d(in_features), self.activation())

        self.block_index = block_index
        self.num_blocks = self.config["blocks_per_group"] * self.config["num_groups"]
        self.layers = self._build_block(in_features, out_features)

        if config["use_shake_shake"]:
            self.shake_shake_layers = self._build_block(in_features, out_features)
        

    # each bloack consists of two linear layers with batch norm and activation
    def _build_block(self, in_features, out_features):
        layers = list()
        
        if self.start_norm == None:
            layers.append(nn.BatchNorm1d(in_features))
            layers.append(self.activation())
        layers.append(nn.Linear(in_features, out_features))

        layers.append(nn.BatchNorm1d(out_features))
        layers.append(self.activation())
        
        if (self.config["use_dropout"]):
            layers.append(nn.Dropout(self.dropout))
        layers.append(nn.Linear(out_features, out_features))

        return nn.Sequential(*layers)

    def forward(self, x):
        residual = x

        # if shortcut is not none we need a layer such that x matches the output dimension
        if self.shortcut != None: # in this case self.start_norm is also != none
            # apply start_norm to x in order to have batchnorm+activation in front of shortcut and layers
            # note that in this case layers does not start with batchnorm+activation but with the first linear layer (see _build_block)
            # as a result if in_features == out_features -> result = x + W(~D(A(BN(W(A(BN(x))))))
            # if in_features != out_features -> result = W_shortcut(A(BN(x))) + W_2(~D(A(BN(W_1(A(BN(x))))))
            x = self.start_norm(x)
            residual = self.shortcut(x)
        
        if self.config["use_shake_shake"]:
            x1 = self.layers(x)
            x2 = self.shake_shake_layers(x)
            alpha, beta = shake_get_alpha_beta(self.training, x.is_cuda)
            x = shake_shake(x1, x2, alpha, beta)
        else:
            x = self.layers(x)
        
        if self.config["use_shake_drop"]:
            alpha, beta = shake_get_alpha_beta(self.training, x.is_cuda)
            bl = shake_drop_get_bl(self.block_index, 1 - self.config["max_shake_drop_probability"], self.num_blocks, self.training, x.is_cuda)
            x = shake_drop(x, alpha, beta, bl)

        x = x + residual
        return x
