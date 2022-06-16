# THE MIT License

# Copyright 2020 Jan Beitner

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

# This part of implementation follows pytorch-forecasting:
# https://github.com/jdb78/pytorch-forecasting/blob/master/pytorch_forecasting/models/nbeats/sub_modules.py

from typing import List, Tuple

import numpy as np

import torch
from torch import nn

from autoPyTorch.pipeline.components.setup.network_backbone.forecasting_backbone.forecasting_decoder.\
    NBEATSDecoder import NBEATSBlock


class TransposeLinear(nn.Module):
    def __init__(self, weights: torch.Tensor):
        super().__init__()
        self.register_buffer('weights', weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.mm(self.weights)


def linspace(backcast_length: int, forecast_length: int, centered: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """
    a function to generate a linear space to encode the positions of the components. For details. We refer to
    Oreshkin et al. N-BEATS: Neural basis expansion analysis for interpretable time series forecasting
    https://arxiv.org/abs/1905.10437
    """
    if centered:
        norm = max(backcast_length, forecast_length)
        start = -backcast_length
        stop = forecast_length - 1
    else:
        norm = backcast_length + forecast_length
        start = 0
        stop = backcast_length + forecast_length - 1
    lin_space = np.linspace(start / norm, stop / norm, backcast_length + forecast_length, dtype=np.float32)
    b_ls = lin_space[:backcast_length]
    f_ls = lin_space[backcast_length:]
    return b_ls, f_ls


def get_generic_heads(block_width: int, thetas_dim: int,
                      forecast_length: int, backcast_length: int) -> Tuple[nn.Module, nn.Module]:
    backcast_head = nn.Sequential(nn.Linear(block_width, thetas_dim, bias=False),
                                  nn.Linear(thetas_dim, backcast_length, bias=False))
    forecast_head = nn.Sequential(nn.Linear(block_width, thetas_dim, bias=False),
                                  nn.Linear(thetas_dim, forecast_length, bias=False))
    return backcast_head, forecast_head


def get_trend_heads(block_width: int, thetas_dim: int,
                    forecast_length: int, backcast_length: int) -> Tuple[nn.Module, nn.Module]:
    base_layer = nn.Linear(block_width, thetas_dim, bias=False)

    backcast_linspace, forecast_linspace = linspace(backcast_length, forecast_length, centered=True)
    norm = np.sqrt(forecast_length / thetas_dim)  # ensure range of predictions is comparable to input

    coefficients_backcast = torch.tensor([backcast_linspace ** i for i in range(thetas_dim)], dtype=torch.float32)

    coefficients_forecast = torch.tensor([forecast_linspace ** i for i in range(thetas_dim)], dtype=torch.float32)

    backcast_head = nn.Sequential(base_layer,
                                  TransposeLinear(coefficients_backcast * norm))
    forecast_head = nn.Sequential(base_layer,
                                  TransposeLinear(coefficients_forecast * norm))

    return backcast_head, forecast_head


def get_seasonality_heads(block_width: int, thetas_dim: int,
                          forecast_length: int, backcast_length: int) -> Tuple[nn.Module, nn.Module]:
    base_layer = nn.Linear(block_width, forecast_length, bias=False)

    backcast_linspace, forecast_linspace = linspace(backcast_length, forecast_length, centered=False)

    def get_frequencies(n: int) -> np.ndarray:
        return np.linspace(0, (backcast_length + forecast_length) / thetas_dim, n)

    p1, p2 = (forecast_length // 2, forecast_length // 2) if forecast_length % 2 == 0 else \
        (forecast_length // 2, forecast_length // 2 + 1)

    s1_b = torch.tensor(
        [np.cos(2 * np.pi * i * backcast_linspace) for i in get_frequencies(p1)], dtype=torch.float32)  # H/2-1
    s2_b = torch.tensor(
        [np.sin(2 * np.pi * i * backcast_linspace) for i in get_frequencies(p2)], dtype=torch.float32)

    s1_f = torch.tensor(
        [np.cos(2 * np.pi * i * forecast_linspace) for i in get_frequencies(p1)], dtype=torch.float32
    )  # H/2-1
    s2_f = torch.tensor(
        [np.sin(2 * np.pi * i * forecast_linspace) for i in get_frequencies(p2)], dtype=torch.float32
    )

    backcast_head = nn.Sequential(base_layer,
                                  TransposeLinear(torch.cat([s1_b, s2_b])))
    forecast_head = nn.Sequential(base_layer,
                                  TransposeLinear(torch.cat([s1_f, s2_f])))
    return backcast_head, forecast_head


def build_NBEATS_network(nbeats_decoder: List[List[NBEATSBlock]],
                         output_shape: Tuple[int]) -> nn.ModuleList:
    nbeats_blocks = []
    for stack_idx, stack in enumerate(nbeats_decoder):
        for block_idx, block in enumerate(nbeats_decoder[stack_idx]):
            stack_type = block.stack_type
            if stack_type == 'generic':
                backcast_head, forecast_head = get_generic_heads(block_width=block.width,
                                                                 thetas_dim=block.expansion_coefficient_length,
                                                                 forecast_length=np.product(output_shape).item(),
                                                                 backcast_length=block.n_in_features)
            elif stack_type == 'trend':
                backcast_head, forecast_head = get_trend_heads(block_width=block.width,
                                                               thetas_dim=block.expansion_coefficient_length,
                                                               forecast_length=np.product(output_shape).item(),
                                                               backcast_length=block.n_in_features)
            elif stack_type == 'seasonality':
                backcast_head, forecast_head = get_seasonality_heads(block_width=block.width,
                                                                     thetas_dim=block.expansion_coefficient_length,
                                                                     forecast_length=np.product(
                                                                         output_shape).item(),
                                                                     backcast_length=block.n_in_features)
            else:
                raise ValueError(f"Unsupported stack_type {stack_type}")
            block.backcast_head = backcast_head
            block.forecast_head = forecast_head

            nbeats_blocks.append(block)
        if nbeats_blocks[-1].weight_sharing:
            block = nbeats_blocks[-1]
            for _ in range(block.num_blocks - 1):
                nbeats_blocks.append(nbeats_blocks[-1])
    return nn.ModuleList(nbeats_blocks)
