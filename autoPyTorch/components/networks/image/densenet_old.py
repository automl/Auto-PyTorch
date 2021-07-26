#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Implementation of a Dense Net for image data.
"""

import torch
import torch.nn as nn
import math

import inspect
from autoPyTorch.components.networks.base_net import BaseImageNet
from autoPyTorch.utils.modules import Reshape
from autoPyTorch.components.networks.activations import all_activations, get_activation
from .utils.utils import get_layer_params

# https://github.com/liuzhuang13/DenseNet

__author__ = "Michael Burkart"
__version__ = "0.0.1"
__license__ = "BSD"

import logging
logger = logging.getLogger('autonet')


class _DenseLayer(nn.Sequential):
    def __init__(self, nChannels, growth_rate, drop_rate, bottleneck):
        super(_DenseLayer, self).__init__()
        
        self.add_module('norm1', nn.BatchNorm2d(nChannels))
        self.add_module('relu1', nn.ReLU(inplace=True))
        if bottleneck:
            self.add_module('conv1', nn.Conv2d(nChannels, 4 * growth_rate, kernel_size=1, stride=1, bias=False))
            nChannels = 4 * growth_rate
            if drop_rate > 0:
                self.add_module('drop', nn.Dropout2d(p=drop_rate, inplace=True))
                
            self.add_module('norm2', nn.BatchNorm2d(nChannels))
            self.add_module('relu2', nn.ReLU(inplace=True))
        self.add_module('conv2', nn.Conv2d(nChannels, growth_rate, kernel_size=3, stride=1, padding=1, bias=False))
        if drop_rate > 0:
            self.add_module('drop', nn.Dropout2d(p=drop_rate, inplace=True))

    def forward(self, x):
        new_features = super(_DenseLayer, self).forward(x)
        return torch.cat([x, new_features], 1)


class _DenseBlock(nn.Sequential):
    def __init__(self, N, nChannels, growth_rate, drop_rate, bottleneck):
        super(_DenseBlock, self).__init__()
        for i in range(N):
            self.add_module('denselayer%d' % (i + 1), _DenseLayer(nChannels, growth_rate, drop_rate, bottleneck))
            nChannels += growth_rate


class _Transition(nn.Sequential):
    def __init__(self, nChannels, nOutChannels, drop_rate, last, pool_size, kernel_size, stride, padding):
        super(_Transition, self).__init__()
        # self.add_module('p_transition', PrintNode("transition"))
        self.add_module('norm', nn.BatchNorm2d(nChannels))
        self.add_module('relu', nn.ReLU(inplace=True))
        # self.add_module('p_last', PrintNode("last transition " + str(last)))
        if last:
            self.add_module('pool', nn.AvgPool2d(kernel_size=pool_size, stride=pool_size))
            self.add_module('reshape', Reshape(nChannels))
        else:
            self.add_module('conv', nn.Conv2d(nChannels, nOutChannels, kernel_size=1, stride=1, bias=False))
            if drop_rate > 0:
                self.add_module('drop', nn.Dropout2d(p=drop_rate, inplace=True))
            self.add_module('pool', nn.AvgPool2d(kernel_size=kernel_size, stride=stride, padding=padding))
        

class DenseNet(BaseImageNet):

    def __init__(self, config, in_features, out_features, final_activation, *args, **kwargs):

        super(DenseNet, self).__init__(config, in_features, out_features, final_activation)

        growth_rate=config['growth_rate']
        bottleneck=config['bottleneck']
        channel_reduction=config['channel_reduction']


        image_size, min_image_size = min(self.iw, self.ih), 3

        nChannels= 2 * growth_rate

        self.features = nn.Sequential()
        
        self.features.add_module('conv0', nn.Conv2d(self.channels, nChannels, kernel_size=3, stride=1, padding=1, bias=False))
        # self.features.add_module('conv0', nn.Conv2d(self.channels, nChannels, kernel_size=7, stride=2, padding=3, bias=False))
        # image_size = math.floor((image_size + 1) / 2)

        self.features.add_module('norm0', nn.BatchNorm2d(nChannels))
        self.features.add_module('activ0', nn.ReLU(inplace=True))

        # self.features.add_module('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        # image_size = math.floor((image_size + 1) / 2)

        division_steps = math.floor(math.log2(image_size) - math.log2(min_image_size) - 1e-5)

        nOutChannels = nChannels
        
        for i in range(1, config['blocks']+1):
            nChannels = nOutChannels

            drop_rate = config['dropout_%d' % i] if config['use_dropout'] else 0

            block = _DenseBlock(N=config['layer_in_block_%d' % i], nChannels=nChannels, bottleneck=bottleneck, 
                                growth_rate=growth_rate, drop_rate=drop_rate)

            self.features.add_module('denseblock%d' % i, block)
            nChannels = nChannels + config['layer_in_block_%d' % i] * growth_rate
            nOutChannels = max(1, math.floor(nChannels * channel_reduction))

            transition = _Transition(   nChannels=nChannels, nOutChannels=nOutChannels, 
                                        drop_rate=drop_rate, last=(i == config['blocks']), 
                                        pool_size=image_size, # only used in last transition -> reduce to '1x1 image'
                                        kernel_size=2 if i > config['blocks'] - division_steps else 1, 
                                        stride=2 if i > config['blocks'] - division_steps else 1,
                                        padding=0)
            self.features.add_module('trans%d' % i, transition)

            if i > config['blocks'] - division_steps:
                image_size = math.floor(image_size / 2)


        # Linear layer
        self.classifier = nn.Linear(nChannels, out_features)

        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                self.matrix_init(m.weight, 'kaiming_normal')
            elif isinstance(m, nn.BatchNorm2d):
                self.matrix_init(m.weight, 'constant_1')
                self.matrix_init(m.bias, 'constant_0')
            elif isinstance(m, nn.Linear):
                self.matrix_init(m.bias, 'constant_0')

    
    def matrix_init(self, matrix, init_type):
        if init_type == 'kaiming_normal':
            nn.init.kaiming_normal_(matrix)
        elif init_type == 'constant_0':
            nn.init.constant_(matrix, 0)
        elif init_type == 'constant_1':
            nn.init.constant_(matrix, 1)
        elif init_type == 'constant_05':
            nn.init.constant_(matrix, 0.5)
        elif init_type == 'random':
            return
        else:
            raise ValueError('Init type ' + init_type + ' is not supported')
            

    def forward(self, x):
        out = self.features(x)
        out = self.classifier(out)
        if not self.training and self.final_activation is not None:
            out = self.final_activation(out)
        return out

    @staticmethod
    def get_config_space(growth_rate_range=(10, 128), nr_blocks=(1, 4), layer_range=(1, 20), **kwargs):

        import ConfigSpace as CS
        import ConfigSpace.hyperparameters as CSH
        from autoPyTorch.utils.config_space_hyperparameter import add_hyperparameter

        cs = CS.ConfigurationSpace()
        add_hyperparameter(cs, CSH.UniformIntegerHyperparameter, 'growth_rate', growth_rate_range)
        add_hyperparameter(cs,   CSH.UniformFloatHyperparameter, 'channel_reduction', [0.1, 0.9])
        add_hyperparameter(cs,    CSH.CategoricalHyperparameter, 'bottleneck', [True, False])

        blocks =        add_hyperparameter(cs, CSH.UniformIntegerHyperparameter, 'blocks', nr_blocks)
        use_dropout =   add_hyperparameter(cs,    CSH.CategoricalHyperparameter, 'use_dropout', [True, False])

        for i in range(1, nr_blocks[1]+1):
            layer =         add_hyperparameter(cs, CSH.UniformIntegerHyperparameter, 'layer_in_block_%d' % i, layer_range)
            dropout =       add_hyperparameter(cs,   CSH.UniformFloatHyperparameter, 'dropout_%d' % i, [0.0, 1.0])
            
            if i > nr_blocks[0]:
                cs.add_condition(CS.GreaterThanCondition(layer, blocks, i-1))
                cs.add_condition(CS.AndConjunction(CS.EqualsCondition(dropout, use_dropout, True), CS.GreaterThanCondition(dropout, blocks, i-1)))
            else:
                cs.add_condition(CS.EqualsCondition(dropout, use_dropout, True))

        return cs



