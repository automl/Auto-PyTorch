import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from collections import OrderedDict

import ConfigSpace
from autoPyTorch.components.networks.base_net import BaseImageNet
from autoPyTorch.utils.config_space_hyperparameter import add_hyperparameter, get_hyperparameter

from autoPyTorch.components.networks.base_net import BaseImageNet

class _DenseLayer(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm2d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv2d(num_input_features, bn_size *
                        growth_rate, kernel_size=1, stride=1, bias=False)),
        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                        kernel_size=3, stride=1, padding=1, bias=False)),
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super(_DenseLayer, self).forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return torch.cat([x, new_features], 1)


class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i * growth_rate, growth_rate, bn_size, drop_rate)
            self.add_module('denselayer%d' % (i + 1), layer)


class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features, pool_size):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=pool_size, stride=pool_size))


class DenseNet(BaseImageNet):
    r"""Densenet-BC model class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_
    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
    """

    def __init__(self, config, in_features, out_features, final_activation, *args, **kwargs):

        super(DenseNet, self).__init__(config, in_features, out_features, final_activation)

        growth_rate = config['growth_rate']
        block_config=[config['layer_in_block_%d' % (i+1)] for i in range(config['blocks'])]
        num_init_features = 2 * growth_rate 
        bn_size = 4
        drop_rate = config['dropout'] if config['use_dropout'] else 0
        num_classes = self.n_classes
        
        image_size, min_image_size = min(self.iw, self.ih), 1

        import math
        division_steps = math.floor(math.log2(image_size) - math.log2(min_image_size) - 1e-5) + 1

        if division_steps > len(block_config) + 1:
            # First convolution
            self.features = nn.Sequential(OrderedDict([
                ('conv0', nn.Conv2d(self.channels, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)),
                ('norm0', nn.BatchNorm2d(num_init_features)),
                ('relu0', nn.ReLU(inplace=True)),
                ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
            ]))
            division_steps -= 2
        else:
            self.features = nn.Sequential(OrderedDict([
                ('conv0', nn.Conv2d(self.channels, num_init_features, kernel_size=3, stride=1, padding=1, bias=False))
            ]))


        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers=num_layers, num_input_features=num_features,
                                bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2, pool_size=2 if i > len(block_config) - division_steps else 1)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2

        # Final batch norm
        self.features.add_module('last_norm', nn.BatchNorm2d(num_features))

        # Linear layer
        self.classifier = nn.Linear(num_features, num_classes)

        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1)).view(features.size(0), -1)
        out = self.classifier(out)

        if not self.training and self.final_activation is not None:
            out = self.final_activation(out)
        return out

    @staticmethod
    def get_config_space(growth_rate_range=(12, 40), nr_blocks=(3, 4), layer_range=([1, 12], [6, 24], [12, 64], [12, 64]), num_init_features=(32, 128), **kwargs):

        import ConfigSpace as CS
        import ConfigSpace.hyperparameters as CSH
        from autoPyTorch.utils.config_space_hyperparameter import add_hyperparameter

        cs = CS.ConfigurationSpace()
        growth_rate_hp = get_hyperparameter(ConfigSpace.UniformIntegerHyperparameter, 'growth_rate', growth_rate_range)
        cs.add_hyperparameter(growth_rate_hp)
        # add_hyperparameter(cs,   CSH.UniformFloatHyperparameter, 'bn_size', [2, 4])
        # add_hyperparameter(cs, CSH.UniformIntegerHyperparameter, 'num_init_features', num_init_features, log=True)
        # add_hyperparameter(cs,    CSH.CategoricalHyperparameter, 'bottleneck', [True, False])

        blocks_hp = get_hyperparameter(ConfigSpace.UniformIntegerHyperparameter, 'blocks', nr_blocks)
        cs.add_hyperparameter(blocks_hp)
        use_dropout =   add_hyperparameter(cs,    CSH.CategoricalHyperparameter, 'use_dropout', [True, False])
        dropout =       add_hyperparameter(cs,   CSH.UniformFloatHyperparameter, 'dropout', [0.0, 1.0])
        cs.add_condition(CS.EqualsCondition(dropout, use_dropout, True))

        if type(nr_blocks[0]) == int:
            min_blocks = nr_blocks[0]
            max_blocks = nr_blocks[1]
        else:
            min_blocks = nr_blocks[0][0]
            max_blocks = nr_blocks[0][1]

        for i in range(1, max_blocks+1):
            layer_hp = get_hyperparameter(ConfigSpace.UniformIntegerHyperparameter, 'layer_in_block_%d' % i, layer_range[i-1])
            cs.add_hyperparameter(layer_hp)
            
            if i > min_blocks:
                cs.add_condition(CS.GreaterThanCondition(layer_hp, blocks_hp, i-1))

        return cs



