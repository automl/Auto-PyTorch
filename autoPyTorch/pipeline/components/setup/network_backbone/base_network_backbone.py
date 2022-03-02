from abc import abstractmethod
from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np

import pandas as pd

from scipy.sparse import spmatrix

import torch
from torch import nn

from autoPyTorch.pipeline.components.base_component import BaseEstimator
from autoPyTorch.pipeline.components.base_component import (
    autoPyTorchComponent,
)
from autoPyTorch.pipeline.components.setup.network_backbone.utils import get_output_shape
from autoPyTorch.utils.common import FitRequirement


class NetworkBackboneComponent(autoPyTorchComponent):
    """
    Base class for network backbones. Holds the backbone module and the config which was used to create it.
    """
    _required_properties = ["name", "shortname", "handles_tabular", "handles_image", "handles_time_series"]

    def __init__(self,
                 **kwargs: Any):
        super().__init__()
        self.add_fit_requirements([
            FitRequirement('is_small_preprocess', (bool,), user_defined=True, dataset_property=True),
            FitRequirement('X_train', (np.ndarray, pd.DataFrame, spmatrix), user_defined=True,
                           dataset_property=False),
            FitRequirement('input_shape', (Iterable,), user_defined=True, dataset_property=True),
            FitRequirement('tabular_transformer', (BaseEstimator,), user_defined=False, dataset_property=False),
            FitRequirement('network_embedding', (nn.Module,), user_defined=False, dataset_property=False)
        ])
        self.backbone: nn.Module = None
        self.config = kwargs
        self.input_shape: Optional[Iterable] = None

    def fit(self, X: Dict[str, Any], y: Any = None) -> BaseEstimator:
        """
        Builds the backbone component and assigns it to self.backbone

        Args:
            X (X: Dict[str, Any]): Dependencies needed by current component to perform fit
            y (Any): not used. To comply with sklearn API
        Returns:
            Self
        """
        self.check_requirements(X, y)
        X_train = X['X_train']

        if X["dataset_properties"]["is_small_preprocess"]:
            input_shape = X_train.shape[1:]
        else:
            # get input shape by transforming first two elements of the training set
            column_transformer = X['tabular_transformer'].preprocessor
            input_shape = column_transformer.transform(X_train[:1]).shape[1:]

        input_shape = get_output_shape(X['network_embedding'], input_shape=input_shape)
        self.input_shape = input_shape

        self.backbone = self.build_backbone(
            input_shape=input_shape,
        )
        return self

    def transform(self, X: Dict[str, Any]) -> Dict[str, Any]:
        """
        Adds the network backbone into the fit dictionary 'X' and returns it.
        Also, updates the input shape as from this point only the shape of
        the transformed dataset is used
        Args:
            X (Dict[str, Any]): 'X' dictionary
        Returns:
            (Dict[str, Any]): the updated 'X' dictionary
        """
        X['dataset_properties'].update({'input_shape': self.input_shape})
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
        return str(cls.get_properties()["shortname"])
