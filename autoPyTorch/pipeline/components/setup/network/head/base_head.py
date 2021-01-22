from abc import abstractmethod
from typing import Any, Dict, Set, Tuple

import torch.nn as nn

from autoPyTorch.pipeline.components.base_component import BaseEstimator, autoPyTorchComponent


class BaseHead(autoPyTorchComponent):
    """
    Head base class
    """
    supported_tasks: Set = set()

    def __init__(self,
                 **kwargs: Any):
        super().__init__()
        self.head: nn.Module = None
        self.config = kwargs

    def fit(self, X: Dict[str, Any], y: Any = None) -> BaseEstimator:
        """
        Not used. Just for API compatibility.
        """
        return self

    @abstractmethod
    def build_head(self, input_shape: Tuple[int, ...], output_shape: Tuple[int, ...]) -> nn.Module:
        """

        Builds the head module and assigns it to self.head

        :param input_shape: shape of the input (usually the shape of the backbone output)
        :param output_shape: shape of the output
        :return: the head module
        """
        raise NotImplementedError()

    @classmethod
    def get_name(cls) -> str:
        """
        Get the name of the head
        :return: name of the head
        """
        return cls.get_properties()["shortname"]
