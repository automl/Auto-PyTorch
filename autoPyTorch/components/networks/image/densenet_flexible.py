#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Implementation of a Dense Net for image data.
"""

import torch
import torch.nn as nn
import math

import ConfigSpace
from autoPyTorch.components.networks.base_net import BaseImageNet
from autoPyTorch.utils.config_space_hyperparameter import add_hyperparameter, get_hyperparameter

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

class PrintNode(nn.Module):
    def __init__(self, msg):
        super(PrintNode, self).__init__()
        self.msg = msg

    def forward(self, x):
        logger.debug(self.msg)
        return x


class _DenseLayer(nn.Sequential):
    def __init__(self, nChannels, growth_rate, drop_rate, bottleneck, kernel_size, activation):
        super(_DenseLayer, self).__init__()
        # self.add_module('p_layer1', PrintNode("layer1"))
        self.add_module('norm1', nn.BatchNorm2d(nChannels))
        self.add_module('relu1', get_activation(activation, inplace=True))
        if bottleneck:
            self.add_module('conv1', nn.Conv2d(nChannels, 4 * growth_rate, kernel_size=1, stride=1, bias=False))
            nChannels = 4 * growth_rate
            if drop_rate > 0:
                self.add_module('drop', nn.Dropout2d(p=drop_rate, inplace=True))
            # self.add_module('p_layer2', PrintNode("layer2"))
            self.add_module('norm2', nn.BatchNorm2d(nChannels))
            self.add_module('relu2', get_activation(activation, inplace=True))
        self.add_module('conv2', nn.Conv2d(nChannels, growth_rate, kernel_size=kernel_size, stride=1, padding=int((kernel_size-1)/2), bias=False))
        if drop_rate > 0:
            self.add_module('drop', nn.Dropout2d(p=drop_rate, inplace=True))

    def forward(self, x):
        new_features = super(_DenseLayer, self).forward(x)
        # logger.debug('concat ' + str(x.shape) + ' and ' + str(new_features.shape))
        return torch.cat([x, new_features], 1)


class _DenseBlock(nn.Sequential):
    def __init__(self, N, nChannels, growth_rate, drop_rate, bottleneck, kernel_size, activation):
        super(_DenseBlock, self).__init__()
        for i in range(N):
            self.add_module('denselayer%d' % (i + 1), _DenseLayer(nChannels, growth_rate, drop_rate, bottleneck, kernel_size, activation))
            nChannels += growth_rate
        


class _Transition(nn.Sequential):
    def __init__(self, nChannels, nOutChannels, drop_rate, last, pool_size, kernel_size, stride, padding, activation):
        super(_Transition, self).__init__()
        # self.add_module('p_transition', PrintNode("transition"))
        self.add_module('norm', nn.BatchNorm2d(nChannels))
        self.add_module('relu', get_activation(activation, inplace=True))
        # self.add_module('p_last', PrintNode("last transition " + str(last)))
        if last:
            self.add_module('pool', nn.AvgPool2d(kernel_size=pool_size, stride=pool_size))
            self.add_module('reshape', Reshape(nChannels))
        else:
            self.add_module('conv', nn.Conv2d(nChannels, nOutChannels, kernel_size=1, stride=1, bias=False))
            if drop_rate > 0:
                self.add_module('drop', nn.Dropout2d(p=drop_rate, inplace=True))
            self.add_module('pool', nn.AvgPool2d(kernel_size=kernel_size, stride=stride, padding=padding))
        

class DenseNetFlexible(BaseImageNet):

    def __init__(self, config, in_features, out_features, final_activation, *args, **kwargs):

        super(DenseNetFlexible, self).__init__(config, in_features, out_features, final_activation)

        growth_rate=config['growth_rate']
        bottleneck=config['bottleneck']
        channel_reduction=config['channel_reduction']

        in_size = min(self.iw, self.ih)
        out_size = max(1, in_size * config['last_image_size'])
        size_reduction = math.pow(in_size / out_size, 1 / (config['blocks'] + 1))

        nChannels= 2 * growth_rate

        self.features = nn.Sequential()

        sizes = [max(1, round(in_size / math.pow(size_reduction, i+1))) for i in range(config['blocks'] + 2)]
        
        in_size, kernel_size, stride, padding = get_layer_params(in_size, sizes[0], config['first_conv_kernel'])
        self.features.add_module('conv0', nn.Conv2d(self.channels, nChannels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False))
        self.features.add_module('norm0', nn.BatchNorm2d(nChannels))
        self.features.add_module('activ0', get_activation(config['first_activation'], inplace=True))

        in_size, kernel_size, stride, padding = get_layer_params(in_size, sizes[1], config['first_pool_kernel'])
        self.features.add_module('pool0', nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=padding))
        # print(in_size)

        nOutChannels = nChannels
        # Each denseblock
        for i in range(1, config['blocks']+1):
            nChannels = nOutChannels

            drop_rate = config['dropout_%d' % i] if config['use_dropout'] else 0

            block = _DenseBlock(N=config['layer_in_block_%d' % i], nChannels=nChannels, bottleneck=bottleneck, 
                                growth_rate=growth_rate, drop_rate=drop_rate, kernel_size=config['conv_kernel_%d' % i],
                                activation=config['activation_%d' % i])

            self.features.add_module('denseblock%d' % i, block)
            nChannels = nChannels + config['layer_in_block_%d' % i] * growth_rate
            nOutChannels = max(1, math.floor(nChannels * channel_reduction))

            out_size, kernel_size, stride, padding = get_layer_params(in_size, sizes[i+1], config['pool_kernel_%d' % i])
            transition = _Transition(   nChannels=nChannels, nOutChannels=nOutChannels, 
                                        drop_rate=drop_rate, last=(i == config['blocks']), 
                                        pool_size=in_size, # only used in last transition -> reduce to '1x1 image'
                                        kernel_size=kernel_size, stride=stride, padding=padding,
                                        activation=config['activation_%d' % i])
            in_size = out_size

            self.features.add_module('trans%d' % i, transition)

        # Linear layer
        self.classifier = nn.Linear(nChannels, out_features)

        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                self.matrix_init(m.weight, config['conv_init'])
            elif isinstance(m, nn.BatchNorm2d):
                self.matrix_init(m.weight, config['batchnorm_weight_init'])
                self.matrix_init(m.bias, config['batchnorm_bias_init'])
            elif isinstance(m, nn.Linear):
                self.matrix_init(m.bias, config['linear_bias_init'])

        # logger.debug(print(self))

        self.layers = nn.Sequential(self.features)
    
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
    def get_config_space(   growth_rate_range=(5, 128), nr_blocks=(1, 5), kernel_range=(2, 7), 
                            layer_range=(5, 50), activations=all_activations.keys(),
                            conv_init=('random', 'kaiming_normal', 'constant_0', 'constant_1', 'constant_05'),
                            batchnorm_weight_init=('random', 'constant_0', 'constant_1', 'constant_05'),
                            batchnorm_bias_init=('random', 'constant_0', 'constant_1', 'constant_05'),
                            linear_bias_init=('random', 'constant_0', 'constant_1', 'constant_05'), **kwargs):

        import ConfigSpace as CS
        import ConfigSpace.hyperparameters as CSH
        from autoPyTorch.utils.config_space_hyperparameter import add_hyperparameter

        cs = CS.ConfigurationSpace()
        growth_rate_hp = get_hyperparameter(ConfigSpace.UniformIntegerHyperparameter, 'growth_rate', growth_rate_range)
        first_conv_kernel_hp = get_hyperparameter(ConfigSpace.UniformIntegerHyperparameter, 'first_conv_kernel', kernel_range)
        first_pool_kernel_hp = get_hyperparameter(ConfigSpace.UniformIntegerHyperparameter, 'first_pool_kernel', kernel_range)
        conv_init_hp = get_hyperparameter(ConfigSpace.CategoricalHyperparameter, 'conv_init', conv_init)
        batchnorm_weight_init_hp = get_hyperparameter(ConfigSpace.CategoricalHyperparameter, 'batchnorm_weight_init', batchnorm_weight_init)
        batchnorm_bias_init_hp = get_hyperparameter(ConfigSpace.CategoricalHyperparameter, 'batchnorm_bias_init', batchnorm_bias_init)
        linear_bias_init_hp = get_hyperparameter(ConfigSpace.CategoricalHyperparameter, 'linear_bias_init', linear_bias_init)
        first_activation_hp = get_hyperparameter(ConfigSpace.CategoricalHyperparameter, 'first_activation', list(set(activations).intersection(all_activations)))
        blocks_hp = get_hyperparameter(ConfigSpace.UniformIntegerHyperparameter, 'blocks', nr_blocks)

        cs.add_hyperparameter(growth_rate_hp)
        cs.add_hyperparameter(first_conv_kernel_hp)
        cs.add_hyperparameter(first_pool_kernel_hp)
        cs.add_hyperparameter(conv_init_hp)
        cs.add_hyperparameter(batchnorm_weight_init_hp)
        cs.add_hyperparameter(batchnorm_bias_init_hp)
        cs.add_hyperparameter(linear_bias_init_hp)
        cs.add_hyperparameter(first_activation_hp)
        cs.add_hyperparameter(blocks_hp)
        add_hyperparameter(cs,   CSH.UniformFloatHyperparameter, 'channel_reduction', [0.1, 0.9])
        add_hyperparameter(cs,   CSH.UniformFloatHyperparameter, 'last_image_size', [0, 1])
        add_hyperparameter(cs,    CSH.CategoricalHyperparameter, 'bottleneck', [True, False])
        use_dropout =   add_hyperparameter(cs,    CSH.CategoricalHyperparameter, 'use_dropout', [True, False])

        if type(nr_blocks[0]) == int:
            min_blocks = nr_blocks[0]
            max_blocks = nr_blocks[1]
        else:
            min_blocks = nr_blocks[0][0]
            max_blocks = nr_blocks[0][1]

        for i in range(1, max_blocks+1):
            layer_hp = get_hyperparameter(ConfigSpace.UniformIntegerHyperparameter, 'layer_in_block_%d' % i, layer_range)
            pool_kernel_hp = get_hyperparameter(ConfigSpace.UniformIntegerHyperparameter, 'pool_kernel_%d' % i, kernel_range)
            activation_hp = get_hyperparameter(ConfigSpace.CategoricalHyperparameter, 'activation_%d' % i, list(set(activations).intersection(all_activations)))
            cs.add_hyperparameter(layer_hp)
            cs.add_hyperparameter(pool_kernel_hp)
            cs.add_hyperparameter(activation_hp)
            dropout =       add_hyperparameter(cs,   CSH.UniformFloatHyperparameter, 'dropout_%d' % i, [0.0, 1.0])
            conv_kernel =   add_hyperparameter(cs,    CSH.CategoricalHyperparameter, 'conv_kernel_%d' % i, [3, 5, 7])

            
            if i > min_blocks:
                cs.add_condition(CS.GreaterThanCondition(layer_hp, blocks_hp, i-1))
                cs.add_condition(CS.GreaterThanCondition(conv_kernel, blocks_hp, i-1))
                cs.add_condition(CS.GreaterThanCondition(pool_kernel_hp, blocks_hp, i-1))
                cs.add_condition(CS.GreaterThanCondition(activation_hp, blocks_hp, i-1))
                cs.add_condition(CS.AndConjunction(CS.EqualsCondition(dropout, use_dropout, True), CS.GreaterThanCondition(dropout, blocks_hp, i-1)))
            else:
                cs.add_condition(CS.EqualsCondition(dropout, use_dropout, True))

        return cs
