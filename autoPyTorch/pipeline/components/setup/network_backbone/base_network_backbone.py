from abc import abstractmethod
from typing import Any, Dict, Tuple

import torch
from torch import nn

from autoPyTorch.pipeline.components.base_component import BaseEstimator
from autoPyTorch.pipeline.components.base_component import (
    autoPyTorchComponent,
)


class NetworkBackboneComponent(autoPyTorchComponent):
    """
    Base class for network backbones. Holds the backbone module and the config which was used to create it.
    """
    _required_properties = ["name", "shortname", "handles_tabular", "handles_image", "handles_time_series"]

    def __init__(self,
                 **kwargs: Any):
        super().__init__()
        self.backbone: nn.Module = None
        self.config = kwargs

    def fit(self, X: Dict[str, Any], y: Any = None) -> BaseEstimator:
        """
        Builds the backbone component and assigns it to self.backbone

        Args:
            X (X: Dict[str, Any]): Dependencies needed by current component to perform fit
            y (Any): not used. To comply with sklearn API
        Returns:
            Self
        """

        input_shape = X['X_train'].shape[1:]

        self.backbone = self.build_backbone(
            input_shape=input_shape,
        )
        return self

    def transform(self, X: Dict[str, Any]) -> Dict[str, Any]:
        """
        Adds the network head into the fit dictionary 'X' and returns it.

        Args:
            X (Dict[str, Any]): 'X' dictionary
        Returns:
            (Dict[str, Any]): the updated 'X' dictionary
        """
        X.update({'network_backbone': self.backbone})
        return X

    @abstractmethod
    def build_backbone(self, input_shape: Tuple[int, ...]) -> nn.Module:
        """
        Builds the backbone module and returns it

        Args:
            input_shape (Tuple[int, ...]): shape of the input to the backbone

        Returns:
            nn.Module: backbone module
        """
        raise NotImplementedError()

    def get_output_shape(self, input_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        """
        Run a dummy forward pass to get the output shape of the backbone.
        Can and should be overridden by subclasses that know the output shape
        without running a dummy forward pass.

        Args:
            input_shape (Tuple[int, ...]): shape of the input

        Returns:
            output_shape (Tuple[int, ...]): shape of the backbone output
        """
        placeholder = torch.randn((2, *input_shape), dtype=torch.float)
        with torch.no_grad():
            output = self.backbone(placeholder)
        return tuple(output.shape[1:])

    @classmethod
    def get_name(cls) -> str:
        """
        Get the name of the backbone

        Args:
            None

        Returns:
            str: Name of the backbone
        """
        return cls.get_properties()["shortname"]
