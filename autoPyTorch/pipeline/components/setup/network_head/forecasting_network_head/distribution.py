# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# or in the "license" file accompanying this file. This file is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.

# This part of codes mainly follow the implementation in gluonts:
# https://github.com/awslabs/gluon-ts/blob/master/src/gluonts/torch/modules/distribution_output.py
# However, we don't simply follow their implementation mainly due to the different network backbone.
# Additionally, scale information is not presented here to avoid

from abc import abstractmethod
from typing import Any, Dict, NamedTuple, Tuple, Type

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Beta, Distribution, Gamma, Normal, Poisson, StudentT


class ProjectionLayer(nn.Module):
    """
    A projection layer that project features to a torch distribution
    """

    value_in_support = 0.0

    # https://github.com/awslabs/gluon-ts/blob/master/src/gluonts/torch/modules/distribution_output.py

    def __init__(
        self,
        num_in_features: int,
        output_shape: Tuple[int, ...],
        n_prediction_heads: int,
        decoder_has_local_layer: bool,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)

        # we consider all the prediction steps holistically. thus, the output of the poj layer is
        # n_prediction_steps * dim *output_shape

        def build_single_proj_layer(arg_dim: int) -> nn.Module:
            """
            build a single proj layer given the input dims, the output is unflattened to fit the required output_shape
            and n_prediction_steps.
            we note that output_shape's first dimensions is always n_prediction_steps
            Args:
                arg_dim (int):
                    dimension of the target distribution

            Returns:
                proj_layer (nn.Module):
                    projection layer that maps the decoder output to parameterize distributions
            """
            if decoder_has_local_layer:
                return nn.Sequential(
                    nn.Linear(num_in_features, np.prod(output_shape).item() * arg_dim),
                    nn.Unflatten(-1, (*output_shape, arg_dim)),
                )
            else:
                return nn.Sequential(
                    nn.Linear(
                        num_in_features,
                        n_prediction_heads * np.prod(output_shape).item() * arg_dim,
                    ),
                    nn.Unflatten(-1, (n_prediction_heads, *output_shape, arg_dim)),
                )

        self.proj = nn.ModuleList(
            [build_single_proj_layer(dim) for dim in self.arg_dims.values()]
        )

    def forward(self, x: torch.Tensor) -> torch.distributions:
        """
        get a target distribution
        Args:
            x: input tensor ([batch_size, in_features]):
                input tensor, acquired by the base header, have the shape [batch_size, in_features]

        Returns:
            dist: torch.distributions ([batch_size, n_prediction_steps, output_shape]):
                an output torch distribution with shape (batch_size, n_prediction_steps, output_shape)
        """
        params_unbounded = [proj(x) for proj in self.proj]
        return self.dist_cls(*self.domain_map(*params_unbounded))

    @property
    @abstractmethod
    def arg_dims(self) -> Dict[str, int]:
        raise NotImplementedError

    @abstractmethod
    def domain_map(self, *args: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        raise NotImplementedError

    @property
    @abstractmethod
    def dist_cls(self) -> Type[Distribution]:
        raise NotImplementedError


class NormalOutput(ProjectionLayer):
    @property
    def arg_dims(self) -> Dict[str, int]:
        return {"loc": 1, "scale": 1}

    def domain_map(self, loc: torch.Tensor, scale: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:  # type: ignore
        scale = F.softplus(scale) + 1e-10
        return loc.squeeze(-1), scale.squeeze(-1)

    @property
    def dist_cls(self) -> Type[Distribution]:
        return Normal  # type: ignore[no-any-return]


class StudentTOutput(ProjectionLayer):
    @property
    def arg_dims(self) -> Dict[str, int]:
        return {"df": 1, "loc": 1, "scale": 1}

    def domain_map(  # type: ignore[override]
        self, df: torch.Tensor, loc: torch.Tensor, scale: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        scale = F.softplus(scale) + 1e-10
        df = 2.0 + F.softplus(df)
        return df.squeeze(-1), loc.squeeze(-1), scale.squeeze(-1)

    @property
    def dist_cls(self) -> Type[Distribution]:
        return StudentT   # type: ignore[no-any-return]


class BetaOutput(ProjectionLayer):
    value_in_support = 0.5

    @property
    def arg_dims(self) -> Dict[str, int]:
        return {"concentration1": 1, "concentration0": 1}

    def domain_map(  # type: ignore[override]
        self, concentration1: torch.Tensor, concentration0: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # TODO we need to adapt epsilon value given the datatype of this module
        epsilon = 1e-10
        concentration1 = F.softplus(concentration1) + epsilon
        concentration0 = F.softplus(concentration0) + epsilon
        return concentration1.squeeze(-1), concentration0.squeeze(-1)

    @property
    def dist_cls(self) -> Type[Distribution]:
        # TODO consider constraints on Beta!!!
        return Beta   # type: ignore[no-any-return]


class GammaOutput(ProjectionLayer):
    value_in_support = 0.5

    @property
    def arg_dims(self) -> Dict[str, int]:
        return {"concentration": 1, "rate": 1}

    def domain_map(  # type: ignore[override]
        self, concentration: torch.Tensor, rate: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # TODO we need to adapt epsilon value given the datatype of this module
        epsilon = 1e-10
        concentration = F.softplus(concentration) + epsilon
        rate = F.softplus(rate) + epsilon
        return concentration.squeeze(-1), rate.squeeze(-1)

    @property
    def dist_cls(self) -> Type[Distribution]:
        return Gamma  # type: ignore[no-any-return]


class PoissonOutput(ProjectionLayer):
    @property
    def arg_dims(self) -> Dict[str, int]:
        return {"rate": 1}

    def domain_map(self, rate: torch.Tensor) -> Tuple[torch.Tensor]:  # type: ignore[override]
        rate_pos = F.softplus(rate).clone()
        return (rate_pos.squeeze(-1),)

    @property
    def dist_cls(self) -> Type[Distribution]:
        return Poisson  # type: ignore[no-any-return]


ALL_DISTRIBUTIONS = {
    "studentT": StudentTOutput,
    "normal": NormalOutput,
    # 'beta': BetaOutput,
    # 'gamma': GammaOutput,
    # 'poisson': PoissonOutput
}  # type: Dict[str, Type[ProjectionLayer]]


class DisForecastingStrategy(NamedTuple):
    dist_cls: str
    forecast_strategy: str = "sample"
    num_samples: int = 100
    aggregation: str = "mean"


# TODO find components that are compatible with beta, gamma and poisson distribution!

# TODO consider how to implement NegativeBinomialOutput without scale information
