from abc import abstractmethod
from typing import Any, Dict, Iterable, Tuple, List, Optional

import numpy as np
import torch
from torch import nn

from autoPyTorch.pipeline.components.base_component import BaseEstimator
from autoPyTorch.pipeline.components.setup.network_head.base_network_head import NetworkHeadComponent
from autoPyTorch.pipeline.components.setup.network_backbone.utils import get_output_shape
from autoPyTorch.utils.common import FitRequirement

from autoPyTorch.pipeline.components.setup.network_head.distributed_network_head.distribution import ALL_DISTRIBUTIONS


class DistributionNetworkHeadComponents(NetworkHeadComponent):
    """
    Base class for network heads used for distribution output.
     Holds the head module and the config which was used to create it.
    """
    _required_properties = ["name", "shortname", "handles_tabular", "handles_image", "handles_time_series",
                            "n_prediction_steps", "train_with_log_prob"]

    def __init__(self,
                 **kwargs: Any):
        super().__init__()
        self.add_fit_requirements([
            FitRequirement('input_shape', (Iterable,), user_defined=True, dataset_property=True),
            FitRequirement('task_type', (str,), user_defined=True, dataset_property=True),
            FitRequirement('train_with_log_prob', (str,), user_defined=True, dataset_property=True),
            FitRequirement('n_prediction_steps', (int,), user_defined=True, dataset_property=True),
            FitRequirement('output_shape', (Iterable, int), user_defined=True, dataset_property=True),
            FitRequirement('window_size', (int,), user_defined=False, dataset_property=False)
        ])
        self.head: Optional[nn.Module] = None
        self.config = kwargs

    def fit(self, X: Dict[str, Any], y: Any = None) -> BaseEstimator:
        """
        Builds the head component and assigns it to self.head

        Args:
            X (X: Dict[str, Any]): Dependencies needed by current component to perform fit
            y (Any): not used. To comply with sklearn API
        Returns:
            Self
        """
        input_shape = X['dataset_properties']['input_shape']
        output_shape = X['dataset_properties']['output_shape']

        auto_regressive = self.config.get("auto_regressive", False)
        X.update({"auto_regressive": auto_regressive})
        # TODO consider Auto-regressive model on vanilla network head
        if auto_regressive:
            output_shape[0] = 1
        mlp_backbone = X.get("MLP_backbone", False)
        network_output_tuple = X.get("network_output_tuple", False)
        if mlp_backbone:
            input_shape = (X["window_size"], input_shape[-1])
        self.head = self.build_head(
            input_shape=get_output_shape(X['network_backbone'], input_shape=input_shape,
                                         network_output_tuple=network_output_tuple),
            output_shape=output_shape,
        )
        return self

    def build_head(self, input_shape: Tuple[int, ...], output_shape: Tuple[int, ...]) -> nn.Module:
        """
        Builds the head module and returns it

        Args:
            input_shape (Tuple[int, ...]): shape of the input to the head (usually the shape of the backbone output)
            output_shape (Tuple[int, ...]): shape of the output of the head

        Returns:
            nn.Module: head module
        """
        base_header_layer, num_head_base_output_features = self._build_head(input_shape)
        # TODO consider other form of proj layers
        proj_layer = self.build_proj_layer(dist_cls=self.config["dist_cls"],
                                           num_head_base_output_features=num_head_base_output_features,
                                           output_shape=output_shape,
                                           )
        return nn.Sequential(*base_header_layer, proj_layer)

    @abstractmethod
    def _build_head(self, input_shape: Tuple[int, ...]) -> Tuple[List[nn.Module], int]:
        """
        Builds the head module and returns it

        Args:
            input_shape (Tuple[int, ...]): shape of the input to the head (usually the shape of the backbone output)
            output_shape (Tuple[int, ...]): shape of the output of the head
            n_prediction_steps (int): how many steps need to be predicted in advance

        Returns:
            nn.Module: head module
        """
        raise NotImplementedError()

    @staticmethod
    def build_proj_layer(dist_cls: str,
                         num_head_base_output_features: int,
                         output_shape: Tuple[int, ...],) -> \
            torch.distributions.Distribution:
        """
        Builds a layer that maps the head output features to a torch distribution
        """
        if dist_cls not in ALL_DISTRIBUTIONS.keys():
            raise ValueError(f'Unsupported distribution class type: {dist_cls}')
        proj_layer = ALL_DISTRIBUTIONS[dist_cls](num_in_features=num_head_base_output_features,
                                                 output_shape=output_shape,)
        return proj_layer
