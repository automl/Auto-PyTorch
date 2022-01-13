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
from autoPyTorch.pipeline.components.setup.network_backbone.forecasting_backbone.base_forecasting_encoder import (
    BaseForecastingEncoder, EncoderNetwork
)
from autoPyTorch.datasets.base_dataset import BaseDatasetPropertiesType
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


class _TemporalConvNet(EncoderNetwork):
    def __init__(self, num_inputs: int, num_channels: List[int], kernel_size: List[int], dropout: float = 0.2):
        super(_TemporalConvNet, self).__init__()
        layers: List[Any] = []
        num_levels = len(num_channels)
        receptive_field = 1

        # stride_values = []

        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            stride = 1
            # stride_values.extend([stride, stride])
            layers += [_TemporalBlock(in_channels,
                                      out_channels,
                                      kernel_size[i],
                                      stride=stride,
                                      dilation=dilation_size,
                                      padding=(kernel_size[i] - 1) * dilation_size,
                                      dropout=dropout[i])]
            # receptive_field_block = 1 + (kernel_size - 1) * dilation_size * \
            #                        (int(np.prod(stride_values[:-2])) * (1 + stride_values[-2]))
            # stride = 1, we ignore stride computation
            receptive_field_block = 1 + 2 * (kernel_size[i] - 1) * dilation_size
            receptive_field += receptive_field_block
        self.receptive_field = receptive_field
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, output_seq=False) -> torch.Tensor:
        # swap sequence and feature dimensions for use with convolutional nets
        x = x.transpose(1, 2).contiguous()
        x = self.network(x)
        x = x.transpose(1, 2).contiguous()
        if output_seq:
            return x
        else:
            return x[:, -1, :]


class TCNEncoder(BaseForecastingEncoder):
    _receptive_field = 1
    """
    Temporal Convolutional Network backbone for time series data (see https://arxiv.org/pdf/1803.01271.pdf).
    """

    def build_encoder(self, input_shape: Tuple[int, ...]) -> nn.Module:
        num_channels = [self.config["num_filters_1"]]
        kernel_size = [self.config["kernel_size_1"]]
        dropout = [self.config[f"dropout_1"] if self.config["use_dropout"] else 0.0]
        for i in range(2, self.config["num_blocks"] + 1):
            num_channels.append(self.config[f"num_filters_{i}"])
            kernel_size.append(self.config[f"kernel_size_{i}"])
            dropout.append(self.config[f"dropout_{i}"] if self.config["use_dropout"] else 0.0)
        encoder = _TemporalConvNet(input_shape[-1],
                                   num_channels,
                                   kernel_size=kernel_size,
                                   dropout=dropout
                                   )
        self._receptive_field = encoder.receptive_field
        return encoder

    @staticmethod
    def allowed_decoders():
        """
        decoder that is compatible with the encoder
        """
        return ['MLPDecoder']

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

    def transform(self, X: Dict[str, Any]) -> Dict[str, Any]:
        X.update({'window_size': self._receptive_field})
        return super().transform(X)

    @staticmethod
    def get_hyperparameter_search_space(
            dataset_properties: Optional[Dict[str, BaseDatasetPropertiesType]] = None,
            num_blocks: HyperparameterSearchSpace = HyperparameterSearchSpace(hyperparameter="num_blocks",
                                                                              value_range=(1, 6),
                                                                              default_value=3),
            num_filters: HyperparameterSearchSpace = HyperparameterSearchSpace(hyperparameter="num_filters",
                                                                               value_range=(4, 64),
                                                                               default_value=32,
                                                                               log=True),
            kernel_size: HyperparameterSearchSpace = HyperparameterSearchSpace(hyperparameter="kernel_size",
                                                                               value_range=(2, 64),
                                                                               default_value=32,
                                                                               log=True),
            use_dropout: HyperparameterSearchSpace = HyperparameterSearchSpace(hyperparameter="use_dropout",
                                                                               value_range=(True, False),
                                                                               default_value=False),
            dropout: HyperparameterSearchSpace = HyperparameterSearchSpace(hyperparameter="dropout",
                                                                           value_range=(0, 0.5),
                                                                           default_value=0.1),
    ) -> ConfigurationSpace:
        cs = ConfigurationSpace()

        min_num_blocks, max_num_blocks = num_blocks.value_range
        num_blocks = get_hyperparameter(num_blocks, UniformIntegerHyperparameter)
        cs.add_hyperparameter(num_blocks)

        use_dropout = get_hyperparameter(use_dropout, CategoricalHyperparameter)
        cs.add_hyperparameter(use_dropout)

        for i in range(1, int(max_num_blocks) + 1):
            num_filter_search_space = HyperparameterSearchSpace(f"num_filters_{i}",
                                                                value_range=num_filters.value_range,
                                                                default_value=num_filters.default_value,
                                                                log=num_filters.log)
            kernel_size_search_space = HyperparameterSearchSpace(f"kernel_size_{i}",
                                                                 value_range=kernel_size.value_range,
                                                                 default_value=kernel_size.default_value,
                                                                 log=kernel_size.log)
            num_filters_hp = get_hyperparameter(num_filter_search_space, UniformIntegerHyperparameter)
            kernel_size_hp = get_hyperparameter(kernel_size_search_space, UniformIntegerHyperparameter)
            cs.add_hyperparameter(num_filters_hp)
            cs.add_hyperparameter(kernel_size_hp)
            if i > int(min_num_blocks):
                cs.add_conditions([
                    CS.GreaterThanCondition(num_filters_hp, num_blocks, i - 1),
                    CS.GreaterThanCondition(kernel_size_hp, num_blocks, i - 1)
                ])

            dropout_search_space = HyperparameterSearchSpace(hyperparameter='dropout_%d' % i,
                                                             value_range=dropout.value_range,
                                                             default_value=dropout.default_value,
                                                             log=dropout.log)
            dropout_hp = get_hyperparameter(dropout_search_space, UniformFloatHyperparameter)
            cs.add_hyperparameter(dropout_hp)

            dropout_condition_1 = CS.EqualsCondition(dropout_hp, use_dropout, True)

            if i > int(min_num_blocks):
                dropout_condition_2 = CS.GreaterThanCondition(dropout_hp, num_blocks, i - 1)
                cs.add_condition(CS.AndConjunction(dropout_condition_1, dropout_condition_2))
            else:
                cs.add_condition(dropout_condition_1)

        return cs
