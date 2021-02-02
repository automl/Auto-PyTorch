from abc import abstractmethod
from typing import Any, Dict, Tuple

import torch.nn as nn

from autoPyTorch.constants import CLASSIFICATION_TASKS, STRING_TO_TASK_TYPES
from autoPyTorch.pipeline.components.base_component import BaseEstimator, autoPyTorchComponent
from autoPyTorch.pipeline.components.setup.network_backbone.utils import get_output_shape


class NetworkHeadComponent(autoPyTorchComponent):
    """
    Base class for network heads. Holds the head module and the config which was used to create it.
    """
    _required_properties = ["name", "shortname", "handles_tabular", "handles_image", "handles_time_series"]

    def __init__(self,
                 **kwargs: Any):
        super().__init__()
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
        input_shape = X['X_train'].shape[1:]
        output_shape = (X['dataset_properties']['num_classes'],) if \
            STRING_TO_TASK_TYPES[X['dataset_properties']['task_type']] in \
            CLASSIFICATION_TASKS else X['dataset_properties']['output_shape']

        self.head = self.build_head(
            input_shape=get_output_shape(X['network_backbone'], input_shape=input_shape),
            output_shape=output_shape,
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
        X.update({'network_head': self.head})
        return X

    @abstractmethod
    def build_head(self, input_shape: Tuple[int, ...], output_shape: Tuple[int, ...]) -> nn.Module:
        """
        Builds the head module and returns it

        Args:
            input_shape (Tuple[int, ...]): shape of the input to the head (usually the shape of the backbone output)
            output_shape (Tuple[int, ...]): shape of the output of the head

        Returns:
            nn.Module: head module
        """
        raise NotImplementedError()

    @classmethod
    def get_name(cls) -> str:
        """
        Get the name of the head

        Args:
            None

        Returns:
            str: Name of the head
        """
        return cls.get_properties()["shortname"]
