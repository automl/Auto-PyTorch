from typing import Any, Dict, List, Optional, Tuple

import ConfigSpace as CS
from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import CategoricalHyperparameter, UniformIntegerHyperparameter

import torch
from torch import nn

from autoPyTorch.pipeline.components.setup.network.head.base_head import BaseHead

_activations: Dict[str, nn.Module] = {
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid
}


class _FullyConvolutional2DHead(nn.Module):
    def __init__(self,
                 input_shape: Tuple[int, ...],
                 output_shape: Tuple[int, ...],
                 pooling_method: str,
                 activation: str,
                 num_layers: int,
                 num_channels: List[int]):
        super().__init__()

        layers = []
        in_channels = input_shape[0]
        for i in range(1, num_layers):
            layers.append(nn.Conv2d(in_channels=in_channels,
                                    out_channels=num_channels[i - 1],
                                    kernel_size=1))
            layers.append(_activations[activation]())
            in_channels = num_channels[i - 1]
        out_channels = output_shape[0]
        layers.append(nn.Conv2d(in_channels=in_channels,
                                out_channels=out_channels,
                                kernel_size=1))
        if pooling_method == "average":
            layers.append(nn.AdaptiveAvgPool2d(output_size=1))
        else:
            layers.append(nn.AdaptiveMaxPool2d(output_size=1))
        self.head = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.size()
        return self.head(x).view(B, -1)


class FullyConvolutional2DHead(BaseHead):
    """
    Head consisting of a number of 2d convolutional connected layers.
    Applies a global pooling operation in the end.
    """
    supported_tasks = {"image_classification", "image_regression"}

    def build_head(self, input_shape: Tuple[int, ...], output_shape: Tuple[int, ...]) -> nn.Module:
        return _FullyConvolutional2DHead(input_shape=input_shape,
                                         output_shape=output_shape,
                                         pooling_method=self.config["pooling_method"],
                                         activation=self.config.get("activation", None),
                                         num_layers=self.config["num_layers"],
                                         num_channels=[self.config[f"layer_{i}_filters"]
                                                       for i in range(1, self.config["num_layers"])])

    @staticmethod
    def get_properties(dataset_properties: Optional[Dict[str, Any]] = None) -> Dict[str, str]:
        return {
            'shortname': 'FullyConvolutionalHead',
            'name': 'FullyConvolutionalHead',
        }

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties: Optional[Dict[str, str]] = None,
                                        min_num_layers: int = 1,
                                        max_num_layers: int = 4,
                                        min_num_filters: int = 16,
                                        max_num_filters: int = 256) -> ConfigurationSpace:
        cs = ConfigurationSpace()

        num_layers_hp = UniformIntegerHyperparameter("num_layers",
                                                     lower=min_num_layers,
                                                     upper=max_num_layers)

        pooling_method_hp = CategoricalHyperparameter("pooling_method",
                                                      choices=["average", "max"])

        activation_hp = CategoricalHyperparameter('activation',
                                                  choices=list(_activations.keys()))

        cs.add_hyperparameters([num_layers_hp, pooling_method_hp, activation_hp])
        cs.add_condition(CS.GreaterThanCondition(activation_hp, num_layers_hp, 1))

        for i in range(1, max_num_layers):
            num_filters_hp = UniformIntegerHyperparameter(f"layer_{i}_filters",
                                                          lower=min_num_filters,
                                                          upper=max_num_filters)
            cs.add_hyperparameter(num_filters_hp)
            if i >= min_num_layers:
                cs.add_condition(CS.GreaterThanCondition(num_filters_hp, num_layers_hp, i))

        return cs
