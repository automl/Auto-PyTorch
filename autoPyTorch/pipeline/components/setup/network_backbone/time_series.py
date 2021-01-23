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

from autoPyTorch.pipeline.components.setup.network_backbone.base_network_backbone import (
    NetworkBackboneComponent,
)


# Code inspired by https://github.com/hfawaz/InceptionTime
# Paper: https://arxiv.org/pdf/1909.04939.pdf
class _InceptionBlock(nn.Module):
    def __init__(self,
                 n_inputs: int,
                 n_filters: int,
                 kernel_size: int,
                 bottleneck: int = None):
        super(_InceptionBlock, self).__init__()
        self.n_filters = n_filters
        self.bottleneck = None \
            if bottleneck is None \
            else nn.Conv1d(n_inputs, bottleneck, kernel_size=1)

        kernel_sizes = [kernel_size // (2 ** i) for i in range(3)]
        n_inputs = n_inputs if bottleneck is None else bottleneck

        # create 3 conv layers with different kernel sizes which are applied in parallel
        self.pad1 = nn.ConstantPad1d(
            padding=self._padding(kernel_sizes[0]), value=0)
        self.conv1 = nn.Conv1d(n_inputs, n_filters, kernel_sizes[0])

        self.pad2 = nn.ConstantPad1d(
            padding=self._padding(kernel_sizes[1]), value=0)
        self.conv2 = nn.Conv1d(n_inputs, n_filters, kernel_sizes[1])

        self.pad3 = nn.ConstantPad1d(
            padding=self._padding(kernel_sizes[2]), value=0)
        self.conv3 = nn.Conv1d(n_inputs, n_filters, kernel_sizes[2])

        # create 1 maxpool and conv layer which are also applied in parallel
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=1, padding=1)
        self.convpool = nn.Conv1d(n_inputs, n_filters, 1)

        self.bn = nn.BatchNorm1d(4 * n_filters)

    def _padding(self, kernel_size: int) -> Tuple[int, int]:
        if kernel_size % 2 == 0:
            return kernel_size // 2, kernel_size // 2 - 1
        else:
            return kernel_size // 2, kernel_size // 2

    def get_n_outputs(self) -> int:
        return 4 * self.n_filters

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.bottleneck is not None:
            x = self.bottleneck(x)
        x1 = self.conv1(self.pad1(x))
        x2 = self.conv2(self.pad2(x))
        x3 = self.conv3(self.pad3(x))
        x4 = self.convpool(self.maxpool(x))
        x = torch.cat([x1, x2, x3, x4], dim=1)
        x = self.bn(x)
        return torch.relu(x)


class _ResidualBlock(nn.Module):
    def __init__(self, n_res_inputs: int, n_outputs: int):
        super(_ResidualBlock, self).__init__()
        self.shortcut = nn.Conv1d(n_res_inputs, n_outputs, 1, bias=False)
        self.bn = nn.BatchNorm1d(n_outputs)

    def forward(self, x: torch.Tensor, res: torch.Tensor) -> torch.Tensor:
        shortcut = self.shortcut(res)
        shortcut = self.bn(shortcut)
        x += shortcut
        return torch.relu(x)


class _InceptionTime(nn.Module):
    def __init__(self,
                 in_features: int,
                 config: Dict[str, Any]) -> None:
        super().__init__()
        self.config = config
        n_inputs = in_features
        n_filters = self.config["num_filters"]
        bottleneck_size = self.config["bottleneck_size"]
        kernel_size = self.config["kernel_size"]
        n_res_inputs = in_features
        for i in range(self.config["num_blocks"]):
            block = _InceptionBlock(n_inputs=n_inputs,
                                    n_filters=n_filters,
                                    bottleneck=bottleneck_size,
                                    kernel_size=kernel_size)
            self.__setattr__(f"inception_block_{i}", block)

            # add a residual block after every 3 inception blocks
            if i % 3 == 2:
                n_res_outputs = block.get_n_outputs()
                self.__setattr__(f"residual_block_{i}", _ResidualBlock(n_res_inputs=n_res_inputs,
                                                                       n_outputs=n_res_outputs))
                n_res_inputs = n_res_outputs
            n_inputs = block.get_n_outputs()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # swap sequence and feature dimensions for use with convolutional nets
        x = x.transpose(1, 2).contiguous()
        res = x
        for i in range(self.config["num_blocks"]):
            x = self.__getattr__(f"inception_block_{i}")(x)
            if i % 3 == 2:
                x = self.__getattr__(f"residual_block_{i}")(x, res)
                res = x
        x = x.transpose(1, 2).contiguous()
        return x


class InceptionTimeBackbone(NetworkBackboneComponent):
    supported_tasks = {"time_series_classification", "time_series_regression"}

    def build_backbone(self, input_shape: Tuple[int, ...]) -> nn.Module:
        backbone = _InceptionTime(in_features=input_shape[-1],
                                  config=self.config)
        self.backbone = backbone
        return backbone

    @staticmethod
    def get_properties(dataset_properties: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        return {
            'shortname': 'InceptionTimeBackbone',
            'name': 'InceptionTimeBackbone',
            'handles_tabular': False,
            'handles_image': False,
            'handles_time_series': True,
        }

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties: Optional[Dict[str, str]] = None,
                                        min_num_blocks: int = 1,
                                        max_num_blocks: int = 10,
                                        min_num_filters: int = 16,
                                        max_num_filters: int = 64,
                                        min_kernel_size: int = 32,
                                        max_kernel_size: int = 64,
                                        min_bottleneck_size: int = 16,
                                        max_bottleneck_size: int = 64,
                                        ) -> ConfigurationSpace:
        cs = ConfigurationSpace()

        num_blocks_hp = UniformIntegerHyperparameter("num_blocks",
                                                     lower=min_num_blocks,
                                                     upper=max_num_blocks)
        cs.add_hyperparameter(num_blocks_hp)

        num_filters_hp = UniformIntegerHyperparameter("num_filters",
                                                      lower=min_num_filters,
                                                      upper=max_num_filters)
        cs.add_hyperparameter(num_filters_hp)

        bottleneck_size_hp = UniformIntegerHyperparameter("bottleneck_size",
                                                          lower=min_bottleneck_size,
                                                          upper=max_bottleneck_size)
        cs.add_hyperparameter(bottleneck_size_hp)

        kernel_size_hp = UniformIntegerHyperparameter("kernel_size",
                                                      lower=min_kernel_size,
                                                      upper=max_kernel_size)
        cs.add_hyperparameter(kernel_size_hp)
        return cs


# Chomp1d, TemporalBlock and TemporalConvNet copied from
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
    supported_tasks = {"time_series_classification", "time_series_regression"}

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
    def get_properties(dataset_properties: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        return {
            "shortname": "TCNBackbone",
            "name": "TCNBackbone",
            'handles_tabular': False,
            'handles_image': False,
            'handles_time_series': True,
        }

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties: Optional[Dict[str, str]] = None,
                                        min_num_blocks: int = 1,
                                        max_num_blocks: int = 10,
                                        min_num_filters: int = 4,
                                        max_num_filters: int = 64,
                                        min_kernel_size: int = 4,
                                        max_kernel_size: int = 64,
                                        min_dropout: float = 0.0,
                                        max_dropout: float = 0.5
                                        ) -> ConfigurationSpace:
        cs = ConfigurationSpace()

        num_blocks_hp = UniformIntegerHyperparameter("num_blocks",
                                                     lower=min_num_blocks,
                                                     upper=max_num_blocks)
        cs.add_hyperparameter(num_blocks_hp)

        kernel_size_hp = UniformIntegerHyperparameter("kernel_size",
                                                      lower=min_kernel_size,
                                                      upper=max_kernel_size)
        cs.add_hyperparameter(kernel_size_hp)

        use_dropout_hp = CategoricalHyperparameter("use_dropout",
                                                   choices=[True, False])
        cs.add_hyperparameter(use_dropout_hp)

        dropout_hp = UniformFloatHyperparameter("dropout",
                                                lower=min_dropout,
                                                upper=max_dropout)
        cs.add_hyperparameter(dropout_hp)
        cs.add_condition(CS.EqualsCondition(dropout_hp, use_dropout_hp, True))

        for i in range(0, max_num_blocks):
            num_filters_hp = UniformIntegerHyperparameter(f"num_filters_{i}",
                                                          lower=min_num_filters,
                                                          upper=max_num_filters)
            cs.add_hyperparameter(num_filters_hp)
            if i >= min_num_blocks:
                cs.add_condition(CS.GreaterThanCondition(
                    num_filters_hp, num_blocks_hp, i))

        return cs
