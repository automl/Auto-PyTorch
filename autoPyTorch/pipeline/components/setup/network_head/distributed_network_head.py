from abc import abstractmethod
from typing import Any, Dict, Iterable, Tuple, Optional

import torch
from torch import nn

from autoPyTorch.pipeline.components.base_component import BaseEstimator
from autoPyTorch.pipeline.components.setup.network_head.base_network_head import NetworkHeadComponent
from autoPyTorch.pipeline.components.setup.network_backbone.utils import get_output_shape
from autoPyTorch.utils.common import FitRequirement

from autoPyTorch.datasets.base_dataset import BaseDatasetPropertiesType
from autoPyTorch.pipeline.components.setup.network_head.distribution import ALL_DISTRIBUTIONS,NormalOutput, \
    StudentTOutput, BetaOutput, GammaOutput, PoissonOutput

from autoPyTorch.pipeline.components.setup.network_head.fully_connected import FullyConnectedHead



class DistributedNetworkComponents(NetworkHeadComponent):
    """
    Base class for network heads. Holds the head module and the config which was used to create it.
    """
    _required_properties = ["name", "shortname", "handles_tabular", "handles_image", "handles_time_series",
                            "n_prediction_steps", "train_with_log_prob"]

    def __init__(self,
                 **kwargs: Any):
        super().__init__()
        self.add_fit_requirements([
            FitRequirement('input_shape', (Iterable,), user_defined=True, dataset_property=True),
            FitRequirement('task_type', (str,), user_defined=True, dataset_property=True),
            FitRequirement('train_with_log_prob', (str, ), user_defined=True, dataset_property=True),
            FitRequirement('n_prediction_steps', (int,), user_defined=True, dataset_property=True),
            FitRequirement('output_shape', (Iterable, int), user_defined=True, dataset_property=True),
        ])
        self.head: nn.Module = None
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
        n_prediction_steps = X['dataset_properties']['n_prediction_steps']

        self.head = self.build_head(
            input_shape=get_output_shape(X['network_backbone'], input_shape=input_shape),
            output_shape=output_shape,
            n_prediction_steps=n_prediction_steps,
        )
        return self

    @abstractmethod
    def build_head(self, input_shape: Tuple[int, ...], output_shape: Tuple[int, ...],
                   n_prediction_steps: int =1) -> nn.Module:
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

    def _build_head(self, input_shape: Tuple[int, ...], output_shape: Tuple[int, ...],
                   n_prediction_steps: int =1) -> nn.Module:
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

    def build_proj_layer(self, dist_cls: str, head_base_output_features: int, n_prediction_steps: int) ->\
            torch.distributions.Distribution:
        """
        Builds a layer that maps the head output features to a torch distribution
        """
        if dist_cls not in ALL_DISTRIBUTIONS.keys():
            raise ValueError(f'Unsupported distribution class type: {dist_cls}')
        proj_layer = ALL_DISTRIBUTIONS[dist_cls](in_features=head_base_output_features,
                                                 n_prediction_steps=n_prediction_steps)
        return proj_layer

