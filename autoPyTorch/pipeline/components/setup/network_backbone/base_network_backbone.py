from abc import abstractmethod
from typing import Any, Dict, Set, Tuple

import torch
from torch import nn

from autoPyTorch.pipeline.components.base_component import BaseEstimator
from autoPyTorch.pipeline.components.base_component import (
    autoPyTorchComponent,
)


class NetworkBackboneComponent(autoPyTorchComponent):
    """
    Backbone base class
    """
    supported_tasks: Set = set()

    def __init__(self,
                 **kwargs: Any):
        super().__init__()
        self.backbone: nn.Module = None
        self.config = kwargs

    def fit(self, X: Dict[str, Any], y: Any = None) -> BaseEstimator:
        """
        Not used. Just for API compatibility.
        """
        input_shape = X['X_train'].shape[1:]

        self.backbone = self.build_backbone(
            input_shape=input_shape,
        )
        return self

    def transform(self, X: Dict[str, Any]) -> Dict[str, Any]:
        """
        Adds the scheduler into the fit dictionary 'X' and returns it.
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

        Builds the backbone module and assigns it to self.backbone

        :param input_shape: shape of the input
        :return: the backbone module
        """
        raise NotImplementedError()

    def get_output_shape(self, input_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        """
        Run a dummy forward pass to get the output shape of the backbone.
        Can and should be overridden by subclasses that know the output shape
        without running a dummy forward pass.

        :param input_shape: shape of the input
        :return: output_shape
        """
        placeholder = torch.randn((2, *input_shape), dtype=torch.float)
        with torch.no_grad():
            output = self.backbone(placeholder)
        return tuple(output.shape[1:])

    @classmethod
    def get_name(cls) -> str:
        """
        Get the name of the backbone
        :return: name of the backbone
        """
        return cls.get_properties()["shortname"]
