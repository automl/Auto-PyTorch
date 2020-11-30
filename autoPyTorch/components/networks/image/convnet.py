#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Basic Implementation of a convolutional network.
"""

from __future__ import division, print_function

import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
import torch.nn as nn

from autoPyTorch.components.networks.base_net import BaseImageNet

__author__ = "Max Dippel, Michael Burkart and Matthias Urban"
__version__ = "0.0.1"
__license__ = "BSD"


class ConvNet(BaseImageNet):
    def __init__(self, config, in_features, out_features, final_activation, *args, **kwargs):
        super(ConvNet, self).__init__(config, in_features, out_features, final_activation)
        self.layers = self._build_net(self.n_classes)


    def forward(self, x):
        x = self.layers(x)
        x = x.reshape(x.size(0), -1)
        x = self.last_layer(x)
        if not self.training and self.final_activation is not None:
            x = self.final_activation(x)
        return x

    def _build_net(self, out_features):
        layers = list()
        init_filter = self.config["conv_init_filters"]
        self._add_layer(layers, self.channels, init_filter, 1)
        
        cw, ch = self._get_layer_size(self.iw, self.ih)
        self.dense_size = init_filter * cw * ch
        print(cw, ch, self.dense_size)
        for i in range(2, self.config["num_layers"]+1):
            cw, ch = self._get_layer_size(cw, ch)
            if cw == 0 or ch == 0:
                print("> reduce network size due to too small layers.")
                break
            self._add_layer(layers, init_filter, init_filter * 2, i)
            init_filter *= 2
            self.dense_size = init_filter * cw * ch
            print(cw, ch, self.dense_size)
            
        self.last_layer = nn.Linear(self.dense_size, out_features)
        nw = nn.Sequential(*layers)
        #print(nw)
        return nw
    
    def _get_layer_size(self, w, h):
        cw = ((w - self.config["conv_kernel_size"] + 2 * self.config["conv_kernel_padding"])
                //self.config["conv_kernel_stride"]) + 1
        ch = ((h - self.config["conv_kernel_size"] + 2 * self.config["conv_kernel_padding"])
                //self.config["conv_kernel_stride"]) + 1
        cw, ch = cw // self.config["pool_size"], ch // self.config["pool_size"]
        return cw, ch

    def _add_layer(self, layers, in_filters, out_filters, layer_id):
        layers.append(nn.Conv2d(in_filters, out_filters,
                                kernel_size=self.config["conv_kernel_size"],
                                stride=self.config["conv_kernel_stride"],
                                padding=self.config["conv_kernel_padding"]))
        layers.append(nn.BatchNorm2d(out_filters))
        layers.append(self.activation())
        layers.append(nn.MaxPool2d(kernel_size=self.config["pool_size"], stride=self.config["pool_size"]))

    @staticmethod
    def get_config_space(user_updates=None):
        cs = CS.ConfigurationSpace()
        
        cs.add_hyperparameter(CSH.CategoricalHyperparameter('activation', ['relu'])) #'sigmoid', 'tanh',
        num_layers = CSH.UniformIntegerHyperparameter('num_layers', lower=2, upper=5)
        cs.add_hyperparameter(num_layers)
        cs.add_hyperparameter(CSH.UniformIntegerHyperparameter('conv_init_filters', lower=16, upper=64))
        cs.add_hyperparameter(CSH.UniformIntegerHyperparameter('conv_kernel_size', lower=1, upper=5))
        cs.add_hyperparameter(CSH.UniformIntegerHyperparameter('conv_kernel_stride', lower=1, upper=3))
        cs.add_hyperparameter(CSH.UniformIntegerHyperparameter('conv_kernel_padding', lower=2, upper=3))
        cs.add_hyperparameter(CSH.UniformIntegerHyperparameter('pool_size', lower=2, upper=3))

        return(cs)
