# This part mainly follows the implementation in gluonts:
# https://github.com/awslabs/gluon-ts/blob/master/src/gluonts/torch/modules/distribution_output.py
# However, we don't simply follow their implementation mainly due to the different network backbone.
# Additionally, we rescale the output in the later phases to avoid

from typing import Callable, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import (
    AffineTransform,
    Beta,
    Distribution,
    Gamma,
    NegativeBinomial,
    Normal,
    Poisson,
    StudentT,
    TransformedDistribution,
)


class ProjectionLayer(nn.Module):
    """
    A projection layer that
    """
    def __init__(self,
                 in_features: int,
                 n_prediction_steps: int,
                 args_dims: [int],
                 domain_map: Callable[..., Tuple[torch.Tensor]],
                 **kwargs,):
        super().__init__(**kwargs)

