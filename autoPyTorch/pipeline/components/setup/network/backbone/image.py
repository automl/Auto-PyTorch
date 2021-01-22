import logging
import math
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple

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

from autoPyTorch.pipeline.components.setup.network.backbone.base_backbone import BaseBackbone

_activations: Dict[str, nn.Module] = {
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid
}


class ConvNetImageBackbone(BaseBackbone):
    supported_tasks = {"image_classification", "image_regression"}

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self.bn_args = {"eps": 1e-5, "momentum": 0.1}

    def _get_layer_size(self, w: int, h: int) -> Tuple[int, int]:
        cw = ((w - self.config["conv_kernel_size"] + 2 * self.config["conv_kernel_padding"])
              // self.config["conv_kernel_stride"]) + 1
        ch = ((h - self.config["conv_kernel_size"] + 2 * self.config["conv_kernel_padding"])
              // self.config["conv_kernel_stride"]) + 1
        cw, ch = cw // self.config["pool_size"], ch // self.config["pool_size"]
        return cw, ch

    def _add_layer(self, layers: List[nn.Module], in_filters: int, out_filters: int) -> None:
        layers.append(nn.Conv2d(in_filters, out_filters,
                                kernel_size=self.config["conv_kernel_size"],
                                stride=self.config["conv_kernel_stride"],
                                padding=self.config["conv_kernel_padding"]))
        layers.append(nn.BatchNorm2d(out_filters, **self.bn_args))
        layers.append(_activations[self.config["activation"]]())
        layers.append(nn.MaxPool2d(kernel_size=self.config["pool_size"], stride=self.config["pool_size"]))

    def build_backbone(self, input_shape: Tuple[int, ...]) -> nn.Module:
        channels, iw, ih = input_shape
        layers: List[nn.Module] = []
        init_filter = self.config["conv_init_filters"]
        self._add_layer(layers, channels, init_filter)

        cw, ch = self._get_layer_size(iw, ih)
        for i in range(2, self.config["num_layers"] + 1):
            cw, ch = self._get_layer_size(cw, ch)
            if cw == 0 or ch == 0:
                logging.info("> reduce network size due to too small layers.")
                break
            self._add_layer(layers, init_filter, init_filter * 2)
            init_filter *= 2
        backbone = nn.Sequential(*layers)
        self.backbone = backbone
        return backbone

    @staticmethod
    def get_properties(dataset_properties: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        return {
            'shortname': 'ConvNetImageBackbone',
            'name': 'ConvNetImageBackbone',
        }

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties: Optional[Dict[str, str]] = None,
                                        min_num_layers: int = 2,
                                        max_num_layers: int = 5,
                                        min_init_filters: int = 16,
                                        max_init_filters: int = 64,
                                        min_kernel_size: int = 2,
                                        max_kernel_size: int = 5,
                                        min_stride: int = 1,
                                        max_stride: int = 3,
                                        min_padding: int = 2,
                                        max_padding: int = 3,
                                        min_pool_size: int = 2,
                                        max_pool_size: int = 3) -> ConfigurationSpace:
        cs = CS.ConfigurationSpace()

        cs.add_hyperparameter(UniformIntegerHyperparameter('num_layers',
                                                           lower=min_num_layers,
                                                           upper=max_num_layers))
        cs.add_hyperparameter(CategoricalHyperparameter('activation',
                                                        choices=list(_activations.keys())))
        cs.add_hyperparameter(UniformIntegerHyperparameter('conv_init_filters',
                                                           lower=min_init_filters,
                                                           upper=max_init_filters))
        cs.add_hyperparameter(UniformIntegerHyperparameter('conv_kernel_size',
                                                           lower=min_kernel_size,
                                                           upper=max_kernel_size))
        cs.add_hyperparameter(UniformIntegerHyperparameter('conv_kernel_stride',
                                                           lower=min_stride,
                                                           upper=max_stride))
        cs.add_hyperparameter(UniformIntegerHyperparameter('conv_kernel_padding',
                                                           lower=min_padding,
                                                           upper=max_padding))
        cs.add_hyperparameter(UniformIntegerHyperparameter('pool_size',
                                                           lower=min_pool_size,
                                                           upper=max_pool_size))
        return cs


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


class DenseNetBackbone(BaseBackbone):
    supported_tasks = {"image_classification", "image_regression"}

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
        }

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties: Optional[Dict[str, str]] = None,
                                        min_growth_rate: int = 12,
                                        max_growth_rate: int = 40,
                                        min_num_blocks: int = 3,
                                        max_num_blocks: int = 4,
                                        min_num_layers: int = 4,
                                        max_num_layers: int = 64) -> ConfigurationSpace:
        cs = CS.ConfigurationSpace()
        growth_rate_hp = UniformIntegerHyperparameter('growth_rate',
                                                      lower=min_growth_rate,
                                                      upper=max_growth_rate)
        cs.add_hyperparameter(growth_rate_hp)

        blocks_hp = UniformIntegerHyperparameter('blocks',
                                                 lower=min_num_blocks,
                                                 upper=max_num_blocks)
        cs.add_hyperparameter(blocks_hp)

        activation_hp = CategoricalHyperparameter('activation',
                                                  choices=list(_activations.keys()))
        cs.add_hyperparameter(activation_hp)

        use_dropout = CategoricalHyperparameter('use_dropout', choices=[True, False])
        dropout = UniformFloatHyperparameter('dropout',
                                             lower=0.0,
                                             upper=1.0)
        cs.add_hyperparameters([use_dropout, dropout])
        cs.add_condition(CS.EqualsCondition(dropout, use_dropout, True))

        for i in range(1, max_num_blocks + 1):
            layer_hp = UniformIntegerHyperparameter('layer_in_block_%d' % i,
                                                    lower=min_num_layers,
                                                    upper=max_num_layers)
            cs.add_hyperparameter(layer_hp)

            if i > min_num_blocks:
                cs.add_condition(CS.GreaterThanCondition(layer_hp, blocks_hp, i - 1))

        return cs
