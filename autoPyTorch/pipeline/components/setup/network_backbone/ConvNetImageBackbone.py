from typing import Any, Dict, List, Optional, Tuple

import ConfigSpace as CS
from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import (
    CategoricalHyperparameter,
    UniformIntegerHyperparameter
)

from torch import nn

from autoPyTorch.datasets.base_dataset import BaseDatasetPropertiesType
from autoPyTorch.pipeline.components.setup.network_backbone.base_network_backbone import NetworkBackboneComponent
from autoPyTorch.pipeline.components.setup.network_backbone.utils import _activations
from autoPyTorch.utils.common import HyperparameterSearchSpace, add_hyperparameter


class ConvNetImageBackbone(NetworkBackboneComponent):
    """
    Standard Convolutional Neural Network backbone for images
    """

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
                break
            self._add_layer(layers, init_filter, init_filter * 2)
            init_filter *= 2
        backbone = nn.Sequential(*layers)
        self.backbone = backbone
        return backbone

    @staticmethod
    def get_properties(dataset_properties: Optional[Dict[str, BaseDatasetPropertiesType]] = None) -> Dict[str, Any]:
        return {
            'shortname': 'ConvNetImageBackbone',
            'name': 'ConvNetImageBackbone',
            'handles_tabular': False,
            'handles_image': True,
            'handles_time_series': False,
        }

    @staticmethod
    def get_hyperparameter_search_space(
        dataset_properties: Optional[Dict[str, BaseDatasetPropertiesType]] = None,
        num_layers: HyperparameterSearchSpace = HyperparameterSearchSpace(hyperparameter='num_layers',
                                                                          value_range=(2, 8),
                                                                          default_value=4,
                                                                          ),
        conv_init_filters: HyperparameterSearchSpace = HyperparameterSearchSpace(hyperparameter='conv_init_filters',
                                                                                 value_range=(16, 64),
                                                                                 default_value=32,
                                                                                 ),
        activation: HyperparameterSearchSpace = HyperparameterSearchSpace(hyperparameter='activation',
                                                                          value_range=tuple(_activations.keys()),
                                                                          default_value=list(_activations.keys())[0],
                                                                          ),
        conv_kernel_size: HyperparameterSearchSpace = HyperparameterSearchSpace(hyperparameter='conv_kernel_size',
                                                                                value_range=(3, 5),
                                                                                default_value=3,
                                                                                ),
        conv_kernel_stride: HyperparameterSearchSpace = HyperparameterSearchSpace(hyperparameter='conv_kernel_stride',
                                                                                  value_range=(1, 3),
                                                                                  default_value=1,
                                                                                  ),
        conv_kernel_padding: HyperparameterSearchSpace = HyperparameterSearchSpace(hyperparameter='conv_kernel_padding',
                                                                                   value_range=(2, 3),
                                                                                   default_value=2,
                                                                                   ),
        pool_size: HyperparameterSearchSpace = HyperparameterSearchSpace(hyperparameter='pool_size',
                                                                         value_range=(2, 3),
                                                                         default_value=2,
                                                                         )
    ) -> ConfigurationSpace:
        cs = CS.ConfigurationSpace()

        add_hyperparameter(cs, num_layers, UniformIntegerHyperparameter)
        add_hyperparameter(cs, conv_init_filters, UniformIntegerHyperparameter)
        add_hyperparameter(cs, activation, CategoricalHyperparameter)
        add_hyperparameter(cs, conv_kernel_size, UniformIntegerHyperparameter)
        add_hyperparameter(cs, conv_kernel_stride, UniformIntegerHyperparameter)
        add_hyperparameter(cs, conv_kernel_padding, UniformIntegerHyperparameter)
        add_hyperparameter(cs, pool_size, UniformIntegerHyperparameter)

        return cs
