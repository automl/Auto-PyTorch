import math
from collections import OrderedDict
from typing import Any, Dict, Optional, Tuple

import ConfigSpace as CS
import torch
from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import (
    CategoricalHyperparameter,
    UniformFloatHyperparameter,
    UniformIntegerHyperparameter
)
from torch import nn
from torch.nn import functional as F

from autoPyTorch.pipeline.components.setup.network_backbone.base_network_backbone import NetworkBackboneComponent

_activations: Dict[str, nn.Module] = {
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid
}


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
        block_config = [self.config['layer_in_block_%d' % (i + 1)] for i in range(self.config['blocks'])]
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
    def get_properties(dataset_properties: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        return {
            'shortname': 'DenseNetBackbone',
            'name': 'DenseNetBackbone',
            'handles_tabular': False,
            'handles_image': True,
            'handles_time_series': False,
        }

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties: Optional[Dict] = None,
                                        num_blocks: Tuple[Tuple, int] = ((3, 4), 3),
                                        num_layers: Tuple[Tuple, int] = ((4, 64), 16),
                                        growth_rate: Tuple[Tuple, int] = ((12, 40), 20),
                                        activation: Tuple[Tuple, str] = (tuple(_activations.keys()),
                                                                         list(_activations.keys())[0]),
                                        use_dropout: Tuple[Tuple, bool] = ((True, False), False),
                                        dropout: Tuple[Tuple, float] = ((0, 0.5), 0.2)
                                        ) -> ConfigurationSpace:
        cs = CS.ConfigurationSpace()

        min_growth_rate, max_growth_rate = growth_rate[0]
        growth_rate_hp = UniformIntegerHyperparameter('growth_rate',
                                                      lower=min_growth_rate,
                                                      upper=max_growth_rate,
                                                      default_value=growth_rate[1])
        cs.add_hyperparameter(growth_rate_hp)

        min_num_blocks, max_num_blocks = num_blocks[0]
        blocks_hp = UniformIntegerHyperparameter('blocks',
                                                 lower=min_num_blocks,
                                                 upper=max_num_blocks,
                                                 default_value=num_blocks[1])
        cs.add_hyperparameter(blocks_hp)

        activation_hp = CategoricalHyperparameter('activation',
                                                  choices=activation[0],
                                                  default_value=activation[1])
        cs.add_hyperparameter(activation_hp)

        use_dropout = CategoricalHyperparameter('use_dropout',
                                                choices=use_dropout[0],
                                                default_value=use_dropout[1])

        min_dropout, max_dropout = dropout[0]
        dropout = UniformFloatHyperparameter('dropout',
                                             lower=min_dropout,
                                             upper=max_dropout,
                                             default_value=dropout[1])

        cs.add_hyperparameters([use_dropout, dropout])
        cs.add_condition(CS.EqualsCondition(dropout, use_dropout, True))

        for i in range(1, max_num_blocks + 1):
            min_num_layers, max_num_layers = num_layers[0]
            layer_hp = UniformIntegerHyperparameter('layer_in_block_%d' % i,
                                                    lower=min_num_layers,
                                                    upper=max_num_layers,
                                                    default_value=num_layers[1])
            cs.add_hyperparameter(layer_hp)

            if i > min_num_blocks:
                cs.add_condition(CS.GreaterThanCondition(layer_hp, blocks_hp, i - 1))

        return cs
