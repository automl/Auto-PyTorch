from typing import Dict, List, Optional, Tuple, Union

import ConfigSpace as CS
from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import CategoricalHyperparameter, UniformIntegerHyperparameter

import torch
from torch import nn

from autoPyTorch.datasets.base_dataset import BaseDatasetPropertiesType
from autoPyTorch.pipeline.components.setup.network_head.base_network_head import NetworkHeadComponent
from autoPyTorch.pipeline.components.setup.network_head.utils import _activations
from autoPyTorch.utils.common import HyperparameterSearchSpace, add_hyperparameter, get_hyperparameter


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


class FullyConvolutional2DHead(NetworkHeadComponent):
    """
    Head consisting of a number of 2d convolutional connected layers.
    Applies a global pooling operation in the end.
    """

    def build_head(self, input_shape: Tuple[int, ...], output_shape: Tuple[int, ...]) -> nn.Module:
        return _FullyConvolutional2DHead(input_shape=input_shape,
                                         output_shape=output_shape,
                                         pooling_method=self.config["pooling_method"],
                                         activation=self.config.get("activation", None),
                                         num_layers=self.config["num_layers"],
                                         num_channels=[self.config[f"layer_{i}_filters"]
                                                       for i in range(1, self.config["num_layers"])])

    @staticmethod
    def get_properties(dataset_properties: Optional[Dict[str, BaseDatasetPropertiesType]] = None
                       ) -> Dict[str, Union[str, bool]]:
        return {
            'shortname': 'FullyConvolutional2DHead',
            'name': 'FullyConvolutional2DHead',
            'handles_tabular': False,
            'handles_image': True,
            'handles_time_series': False,
        }

    @staticmethod
    def get_hyperparameter_search_space(
        dataset_properties: Optional[Dict[str, BaseDatasetPropertiesType]] = None,
        num_layers: HyperparameterSearchSpace = HyperparameterSearchSpace(hyperparameter="num_layers",
                                                                          value_range=(1, 4),
                                                                          default_value=2),
        num_filters: HyperparameterSearchSpace = HyperparameterSearchSpace(hyperparameter="num_filters",
                                                                           value_range=(16, 256),
                                                                           default_value=32),
        activation: HyperparameterSearchSpace = HyperparameterSearchSpace(hyperparameter="activation",
                                                                          value_range=tuple(_activations.keys()),
                                                                          default_value=list(_activations.keys())[0]),
        pooling_method: HyperparameterSearchSpace = HyperparameterSearchSpace(hyperparameter="pooling_method",
                                                                              value_range=("average", "max"),
                                                                              default_value="max"),
    ) -> ConfigurationSpace:
        cs = ConfigurationSpace()

        min_num_layers, max_num_layers = num_layers.value_range
        num_layers_hp = get_hyperparameter(num_layers, UniformIntegerHyperparameter)

        add_hyperparameter(cs, pooling_method, CategoricalHyperparameter)

        activation_hp = get_hyperparameter(activation, CategoricalHyperparameter)

        cs.add_hyperparameters([num_layers_hp, activation_hp])
        cs.add_condition(CS.GreaterThanCondition(activation_hp, num_layers_hp, 1))

        for i in range(1, int(max_num_layers)):
            num_filters_search_space = HyperparameterSearchSpace(f"layer_{i}_filters",
                                                                 value_range=num_filters.value_range,
                                                                 default_value=num_filters.default_value,
                                                                 log=num_filters.log)
            num_filters_hp = get_hyperparameter(num_filters_search_space, UniformIntegerHyperparameter)
            cs.add_hyperparameter(num_filters_hp)
            if i >= int(min_num_layers):
                cs.add_condition(CS.GreaterThanCondition(num_filters_hp, num_layers_hp, i))

        return cs
