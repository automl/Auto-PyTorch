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


from typing import Dict, Tuple

from abc import abstractmethod
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import (
    Beta,
    Distribution,
    Gamma,
    NegativeBinomial,
    Normal,
    Poisson,
    StudentT,
)


class ProjectionLayer(nn.Module):
    """
    A projection layer that
    """
    def __init__(self,
                 in_features: int,
                 n_prediction_steps: int,
                 **kwargs,):
        super().__init__(**kwargs)
        # we consider all the prediction steps holistically. thus, the output of the poj layer is
        # n_prediction_steps * dim
        self.proj = nn.ModuleList(
            [nn.Linear(in_features, n_prediction_steps * dim) for dim in self.args_dim.values()]
        )

    def forward(self, x: torch.Tensor) -> torch.distributions:
        params_unbounded = [proj(x) for proj in self.proj]
        return self.dist_cls(self.domain_map(*params_unbounded))

    @property
    @abstractmethod
    def arg_dims(self) -> Dict[str, int]:
        raise NotImplementedError

    @abstractmethod
    def domain_map(self, *args: torch.Tensor) -> Tuple[torch.Tensor]:
        raise NotImplementedError

    @property
    @abstractmethod
    def dist_cls(self) -> type(Distribution):
        raise NotImplementedError


class NormalOutput(ProjectionLayer):
    @property
    def arg_dims(self) -> Dict[str, int]:
        return {"loc": 1, "scale": 1}

    def domain_map(self, loc: torch.Tensor, scale: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        scale = F.softplus(scale)
        return loc.squeeze(-1), scale.squeeze(-1)

    @property
    def dist_cls(self) -> type(Distribution):
        return Normal


class StudentTOutput(ProjectionLayer):
    @property
    def arg_dims(self) -> Dict[str, int]:
        return {"df": 1, "loc": 1, "scale": 1}

    def domain_map(self, cls, df: torch.Tensor, loc: torch.Tensor, scale: torch.Tensor)\
            -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        scale = F.softplus(scale)
        df = 2.0 + F.softplus(df)
        return df.squeeze(-1), loc.squeeze(-1), scale.squeeze(-1)

    @property
    def dist_cls(self) -> type(Distribution):
        return StudentT


class BetaOutput(ProjectionLayer):
    @property
    def arg_dims(self) -> Dict[str, int]:
        return {"concentration1": 1, "concentration0": 1}

    def domain_map(self, concentration1: torch.Tensor, concentration0: torch.Tensor)\
            -> Tuple[torch.Tensor, torch.Tensor]:
        # TODO we need to adapt epsilon value given the datatype of this module
        epsilon = 1e-10
        concentration1 = F.softplus(concentration1) + epsilon
        concentration0 = F.softplus(concentration0) + epsilon
        return concentration1.squeeze(dim=-1), concentration0.squeeze(dim=-1)

    @property
    def dist_cls(self) -> type(Distribution):
        return Beta


class GammaOutput(ProjectionLayer):
    @property
    def arg_dims(self) -> Dict[str, int]:
        return {"concentration": 1, "rate": 1}

    def domain_map(self, concentration: torch.Tensor, rate: torch.Tensor)\
            -> Tuple[torch.Tensor, torch.Tensor]:
        # TODO we need to adapt epsilon value given the datatype of this module
        epsilon = 1e-10
        concentration = F.softplus(concentration) + epsilon
        rate = F.softplus(rate) + epsilon
        return concentration.squeeze(dim=-1), rate.squeeze(dim=-1)

    @property
    def dist_cls(self) -> type(Distribution):
        return Gamma


class PoissonOutput(ProjectionLayer):
    @property
    def arg_dims(self) -> Dict[str, int]:
        return {"rate": 1}

    def domain_map(self, rate: torch.Tensor) -> Tuple[torch.Tensor,]:
        rate_pos = F.softplus(rate).clone()
        return rate_pos.squeeze(-1),

    @property
    def dist_cls(self) -> type(Distribution):
        return Poisson


ALL_DISTRIBUTIONS = {'normal': NormalOutput,
                     'studentT': StudentTOutput,
                     'beta': BetaOutput,
                     'gamma': GammaOutput,
                     'poisson': PoissonOutput}  # type: Dict[str, type(ProjectionLayer)]


#TODO consider how to implement NegativeBinomialOutput without scale information
# class NegativeBinomialOutput(ProjectionLayer):
