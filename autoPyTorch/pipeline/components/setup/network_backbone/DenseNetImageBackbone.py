import math
from collections import OrderedDict
from typing import Any, Dict, Optional, Tuple

import ConfigSpace as CS
from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import (
    CategoricalHyperparameter,
    UniformFloatHyperparameter,
    UniformIntegerHyperparameter
)

import torch
from torch import nn
from torch.nn import functional as F

from autoPyTorch.datasets.base_dataset import BaseDatasetPropertiesType
from autoPyTorch.pipeline.components.setup.network_backbone.base_network_backbone import NetworkBackboneComponent
from autoPyTorch.pipeline.components.setup.network_backbone.utils import _activations
from autoPyTorch.utils.common import HyperparameterSearchSpace, add_hyperparameter, get_hyperparameter


class _DenseLayer(nn.Sequential):
    def __init__(self,
                 num_input_features: int,
                 activation: str,
                 growth_rate: int,
                 bn_size: int,
                 drop_rate: float,
                 bn_args: Dict[str, Any]):
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm2d(num_input_features, **bn_args)),
        self.add_module('relu1', _activations[activation]()),
        self.add_module('conv1', nn.Conv2d(num_input_features, bn_size * growth_rate,
                                           kernel_size=1, stride=1, bias=False)),
        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate, **bn_args)),
        self.add_module('relu2', _activations[activation]()),
        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                                           kernel_size=3, stride=1, padding=1, bias=False)),
        self.drop_rate = drop_rate

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        new_features = super(_DenseLayer, self).forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return torch.cat([x, new_features], 1)


class _DenseBlock(nn.Sequential):
    def __init__(self,
                 num_layers: int,
                 num_input_features: int,
                 activation: str,
                 bn_size: int,
                 growth_rate: int,
                 drop_rate: float,
                 bn_args: Dict[str, Any]):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features=num_input_features + i * growth_rate,
                                activation=activation,
                                growth_rate=growth_rate,
                                bn_size=bn_size,
                                drop_rate=drop_rate,
                                bn_args=bn_args)
            self.add_module('denselayer%d' % (i + 1), layer)


class _Transition(nn.Sequential):
    def __init__(self,
                 num_input_features: int,
                 activation: str,
                 num_output_features: int,
                 pool_size: int,
                 bn_args: Dict[str, Any]):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features, **bn_args))
        self.add_module('relu', _activations[activation]())
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=pool_size, stride=pool_size))


class DenseNetBackbone(NetworkBackboneComponent):
    """
    Dense Net Backbone for images (see https://arxiv.org/pdf/1608.06993.pdf)
    """

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self.bn_args = {"eps": 1e-5, "momentum": 0.1}

    def build_backbone(self, input_shape: Tuple[int, ...]) -> nn.Module:
        channels, iw, ih = input_shape

        growth_rate = self.config['growth_rate']
        block_config = [self.config['layer_in_block_%d' % (i + 1)] for i in range(self.config['num_blocks'])]
        num_init_features = 2 * growth_rate
        bn_size = 4
        drop_rate = self.config['dropout'] if self.config['use_dropout'] else 0

        image_size, min_image_size = min(iw, ih), 1

        division_steps = math.floor(math.log2(image_size) - math.log2(min_image_size) - 1e-5) + 1

        if division_steps > len(block_config) + 1:
            # First convolution
            features = nn.Sequential(OrderedDict([
                ('conv0', nn.Conv2d(channels, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)),
                ('norm0', nn.BatchNorm2d(num_init_features, **self.bn_args)),
                ('relu0', nn.ReLU(inplace=True)),
                ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
            ]))
            division_steps -= 2
        else:
            features = nn.Sequential(OrderedDict([
                ('conv0', nn.Conv2d(channels, num_init_features, kernel_size=3, stride=1, padding=1, bias=False))
            ]))

        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers=num_layers,
                                activation=self.config["activation"],
                                num_input_features=num_features,
                                bn_size=bn_size,
                                growth_rate=growth_rate,
                                drop_rate=drop_rate,
                                bn_args=self.bn_args)
            features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features,
                                    activation=self.config["activation"],
                                    num_output_features=num_features // 2,
                                    pool_size=2 if i > len(block_config) - division_steps else 1,
                                    bn_args=self.bn_args)
                features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2

        # Final batch norm
        features.add_module('last_norm', nn.BatchNorm2d(num_features, **self.bn_args))
        self.backbone = features
        return features

    @staticmethod
    def get_properties(dataset_properties: Optional[Dict[str, BaseDatasetPropertiesType]] = None) -> Dict[str, Any]:
        return {
            'shortname': 'DenseNetBackbone',
            'name': 'DenseNetBackbone',
            'handles_tabular': False,
            'handles_image': True,
            'handles_time_series': False,
        }

    @staticmethod
    def get_hyperparameter_search_space(
        dataset_properties: Optional[Dict[str, BaseDatasetPropertiesType]] = None,
        num_layers: HyperparameterSearchSpace = HyperparameterSearchSpace(hyperparameter='num_layers',
                                                                          value_range=(4, 64),
                                                                          default_value=16,
                                                                          ),
        num_blocks: HyperparameterSearchSpace = HyperparameterSearchSpace(hyperparameter='num_blocks',
                                                                          value_range=(3, 4),
                                                                          default_value=3,
                                                                          ),
        growth_rate: HyperparameterSearchSpace = HyperparameterSearchSpace(hyperparameter='growth_rate',
                                                                           value_range=(12, 40),
                                                                           default_value=20,
                                                                           ),
        activation: HyperparameterSearchSpace = HyperparameterSearchSpace(hyperparameter='activation',
                                                                          value_range=tuple(_activations.keys()),
                                                                          default_value=list(_activations.keys())[0],
                                                                          ),
        use_dropout: HyperparameterSearchSpace = HyperparameterSearchSpace(hyperparameter='use_dropout',
                                                                           value_range=(True, False),
                                                                           default_value=False,
                                                                           ),
        dropout: HyperparameterSearchSpace = HyperparameterSearchSpace(hyperparameter='dropout',
                                                                       value_range=(0, 0.5),
                                                                       default_value=0.2,
                                                                       ),
    ) -> ConfigurationSpace:
        cs = CS.ConfigurationSpace()

        add_hyperparameter(cs, num_layers, UniformIntegerHyperparameter)
        add_hyperparameter(cs, growth_rate, UniformIntegerHyperparameter)

        min_num_blocks, max_num_blocks = num_blocks.value_range
        blocks_hp = get_hyperparameter(num_blocks, UniformIntegerHyperparameter)
        cs.add_hyperparameter(blocks_hp)

        add_hyperparameter(cs, activation, CategoricalHyperparameter)

        use_dropout = get_hyperparameter(use_dropout, CategoricalHyperparameter)

        dropout = get_hyperparameter(dropout, UniformFloatHyperparameter)

        cs.add_hyperparameters([use_dropout, dropout])
        cs.add_condition(CS.EqualsCondition(dropout, use_dropout, True))

        for i in range(1, int(max_num_blocks) + 1):

            layer_search_space = HyperparameterSearchSpace(hyperparameter='layer_in_block_%d' % i,
                                                           value_range=num_layers.value_range,
                                                           default_value=num_layers.default_value,
                                                           log=num_layers.log)
            layer_hp = get_hyperparameter(layer_search_space, UniformIntegerHyperparameter)

            cs.add_hyperparameter(layer_hp)
            if i > int(min_num_blocks):
                cs.add_condition(CS.GreaterThanCondition(layer_hp, blocks_hp, i - 1))

        return cs
