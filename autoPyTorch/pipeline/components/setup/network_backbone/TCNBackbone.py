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
from torch.nn.utils import weight_norm

from autoPyTorch.datasets.base_dataset import BaseDatasetPropertiesType
from autoPyTorch.pipeline.components.setup.network_backbone.base_network_backbone import NetworkBackboneComponent
from autoPyTorch.utils.common import HyperparameterSearchSpace, add_hyperparameter, get_hyperparameter


# _Chomp1d, _TemporalBlock and _TemporalConvNet copied from
# https://github.com/locuslab/TCN/blob/master/TCN/tcn.py, Carnegie Mellon University Locus Labs
# Paper: https://arxiv.org/pdf/1803.01271.pdf
class _Chomp1d(nn.Module):
    def __init__(self, chomp_size: int):
        super(_Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, :, :-self.chomp_size].contiguous()


class _TemporalBlock(nn.Module):
    def __init__(self,
                 n_inputs: int,
                 n_outputs: int,
                 kernel_size: int,
                 stride: int,
                 dilation: int,
                 padding: int,
                 dropout: float = 0.2):
        super(_TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = _Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = _Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(
            n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        # self.init_weights()

    def init_weights(self) -> None:
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class _TemporalConvNet(nn.Module):
    def __init__(self, num_inputs: int, num_channels: List[int], kernel_size: int = 2, dropout: float = 0.2):
        super(_TemporalConvNet, self).__init__()
        layers: List[Any] = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers += [_TemporalBlock(in_channels,
                                      out_channels,
                                      kernel_size,
                                      stride=1,
                                      dilation=dilation_size,
                                      padding=(kernel_size - 1) * dilation_size,
                                      dropout=dropout)]
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # swap sequence and feature dimensions for use with convolutional nets
        x = x.transpose(1, 2).contiguous()
        x = self.network(x)
        x = x.transpose(1, 2).contiguous()
        return x


class TCNBackbone(NetworkBackboneComponent):
    """
    Temporal Convolutional Network backbone for time series data (see https://arxiv.org/pdf/1803.01271.pdf).
    """

    def build_backbone(self, input_shape: Tuple[int, ...]) -> nn.Module:
        num_channels = [self.config["num_filters_0"]]
        for i in range(1, self.config["num_blocks"]):
            num_channels.append(self.config[f"num_filters_{i}"])
        backbone = _TemporalConvNet(input_shape[-1],
                                    num_channels,
                                    kernel_size=self.config["kernel_size"],
                                    dropout=self.config["dropout"] if self.config["use_dropout"] else 0.0
                                    )
        self.backbone = backbone
        return backbone

    @staticmethod
    def get_properties(dataset_properties: Optional[Dict[str, BaseDatasetPropertiesType]] = None
                       ) -> Dict[str, Any]:
        return {
            "shortname": "TCNBackbone",
            "name": "TCNBackbone",
            'handles_tabular': False,
            'handles_image': False,
            'handles_time_series': True,
        }

    @staticmethod
    def get_hyperparameter_search_space(
        dataset_properties: Optional[Dict[str, BaseDatasetPropertiesType]] = None,
        num_blocks: HyperparameterSearchSpace = HyperparameterSearchSpace(hyperparameter="num_blocks",
                                                                          value_range=(1, 10),
                                                                          default_value=5),
        num_filters: HyperparameterSearchSpace = HyperparameterSearchSpace(hyperparameter="num_filters",
                                                                           value_range=(4, 64),
                                                                           default_value=32),
        kernel_size: HyperparameterSearchSpace = HyperparameterSearchSpace(hyperparameter="kernel_size",
                                                                           value_range=(4, 64),
                                                                           default_value=32),
        use_dropout: HyperparameterSearchSpace = HyperparameterSearchSpace(hyperparameter="use_dropout",
                                                                           value_range=(True, False),
                                                                           default_value=False),
        dropout: HyperparameterSearchSpace = HyperparameterSearchSpace(hyperparameter="dropout",
                                                                       value_range=(0, 0.5),
                                                                       default_value=0.1),
    ) -> ConfigurationSpace:
        cs = ConfigurationSpace()

        min_num_blocks, max_num_blocks = num_blocks.value_range
        num_blocks_hp = get_hyperparameter(num_blocks, UniformIntegerHyperparameter)
        cs.add_hyperparameter(num_blocks_hp)

        add_hyperparameter(cs, kernel_size, UniformIntegerHyperparameter)

        use_dropout_hp = get_hyperparameter(use_dropout, CategoricalHyperparameter)
        cs.add_hyperparameter(use_dropout_hp)

        dropout_hp = get_hyperparameter(dropout, UniformFloatHyperparameter)
        cs.add_hyperparameter(dropout_hp)
        cs.add_condition(CS.EqualsCondition(dropout_hp, use_dropout_hp, True))

        for i in range(0, int(max_num_blocks)):
            num_filter_search_space = HyperparameterSearchSpace(f"num_filters_{i}",
                                                                value_range=num_filters.value_range,
                                                                default_value=num_filters.default_value,
                                                                log=num_filters.log)
            num_filters_hp = get_hyperparameter(num_filter_search_space, UniformIntegerHyperparameter)
            cs.add_hyperparameter(num_filters_hp)
            if i >= int(min_num_blocks):
                cs.add_condition(CS.GreaterThanCondition(
                    num_filters_hp, num_blocks_hp, i))

        return cs
