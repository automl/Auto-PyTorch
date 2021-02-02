from typing import Any, Dict, List, Optional, Tuple

import ConfigSpace as CS
from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import (
    CategoricalHyperparameter,
    UniformIntegerHyperparameter
)

from torch import nn

from autoPyTorch.pipeline.components.setup.network_backbone.base_network_backbone import NetworkBackboneComponent
from autoPyTorch.pipeline.components.setup.network_backbone.utils import _activations


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
    def get_properties(dataset_properties: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        return {
            'shortname': 'ConvNetImageBackbone',
            'name': 'ConvNetImageBackbone',
            'handles_tabular': False,
            'handles_image': True,
            'handles_time_series': False,
        }

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties: Optional[Dict] = None,
                                        num_layers: Tuple[Tuple, int] = ((2, 8), 4),
                                        num_init_filters: Tuple[Tuple, int] = ((16, 64), 32),
                                        activation: Tuple[Tuple, str] = (tuple(_activations.keys()),
                                                                         list(_activations.keys())[0]),
                                        kernel_size: Tuple[Tuple, int] = ((3, 5), 3),
                                        stride: Tuple[Tuple, int] = ((1, 3), 1),
                                        padding: Tuple[Tuple, int] = ((2, 3), 2),
                                        pool_size: Tuple[Tuple, int] = ((2, 3), 2)
                                        ) -> ConfigurationSpace:
        cs = CS.ConfigurationSpace()

        min_num_layers, max_num_layers = num_layers[0]
        cs.add_hyperparameter(UniformIntegerHyperparameter('num_layers',
                                                           lower=min_num_layers,
                                                           upper=max_num_layers,
                                                           default_value=num_layers[1]))

        cs.add_hyperparameter(CategoricalHyperparameter('activation',
                                                        choices=activation[0],
                                                        default_value=activation[1]))

        min_init_filters, max_init_filters = num_init_filters[0]
        cs.add_hyperparameter(UniformIntegerHyperparameter('conv_init_filters',
                                                           lower=min_init_filters,
                                                           upper=max_init_filters,
                                                           default_value=num_init_filters[1]))

        min_kernel_size, max_kernel_size = kernel_size[0]
        cs.add_hyperparameter(UniformIntegerHyperparameter('conv_kernel_size',
                                                           lower=min_kernel_size,
                                                           upper=max_kernel_size,
                                                           default_value=kernel_size[1]))

        min_stride, max_stride = stride[0]
        cs.add_hyperparameter(UniformIntegerHyperparameter('conv_kernel_stride',
                                                           lower=min_stride,
                                                           upper=max_stride,
                                                           default_value=stride[1]))

        min_padding, max_padding = padding[0]
        cs.add_hyperparameter(UniformIntegerHyperparameter('conv_kernel_padding',
                                                           lower=min_padding,
                                                           upper=max_padding,
                                                           default_value=padding[1]))

        min_pool_size, max_pool_size = pool_size[0]
        cs.add_hyperparameter(UniformIntegerHyperparameter('pool_size',
                                                           lower=min_pool_size,
                                                           upper=max_pool_size,
                                                           default_value=pool_size[1]))

        return cs
