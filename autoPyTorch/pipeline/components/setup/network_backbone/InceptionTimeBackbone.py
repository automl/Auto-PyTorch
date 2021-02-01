from typing import Any, Dict, Optional, Tuple

from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import (
    UniformIntegerHyperparameter
)

import torch
from torch import nn

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
    """
    InceptionTime backbone for time series data (see https://arxiv.org/pdf/1909.04939.pdf).
    """

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
    def get_hyperparameter_search_space(dataset_properties: Optional[Dict] = None,
                                        num_blocks: Tuple[Tuple, int] = ((1, 10), 5),
                                        num_filters: Tuple[Tuple, int] = ((4, 64), 32),
                                        kernel_size: Tuple[Tuple, int] = ((4, 64), 32),
                                        bottleneck_size: Tuple[Tuple, int] = ((16, 64), 32)
                                        ) -> ConfigurationSpace:
        cs = ConfigurationSpace()

        min_num_blocks, max_num_blocks = num_blocks[0]
        num_blocks_hp = UniformIntegerHyperparameter("num_blocks",
                                                     lower=min_num_blocks,
                                                     upper=max_num_blocks,
                                                     default_value=num_blocks[1])
        cs.add_hyperparameter(num_blocks_hp)

        min_num_filters, max_num_filters = num_filters[0]
        num_filters_hp = UniformIntegerHyperparameter("num_filters",
                                                      lower=min_num_filters,
                                                      upper=max_num_filters,
                                                      default_value=num_filters[1])
        cs.add_hyperparameter(num_filters_hp)

        min_bottleneck_size, max_bottleneck_size = bottleneck_size[0]
        bottleneck_size_hp = UniformIntegerHyperparameter("bottleneck_size",
                                                          lower=min_bottleneck_size,
                                                          upper=max_bottleneck_size,
                                                          default_value=bottleneck_size[1])
        cs.add_hyperparameter(bottleneck_size_hp)

        min_kernel_size, max_kernel_size = kernel_size[0]
        kernel_size_hp = UniformIntegerHyperparameter("kernel_size",
                                                      lower=min_kernel_size,
                                                      upper=max_kernel_size,
                                                      default_value=kernel_size[1])
        cs.add_hyperparameter(kernel_size_hp)
        return cs
