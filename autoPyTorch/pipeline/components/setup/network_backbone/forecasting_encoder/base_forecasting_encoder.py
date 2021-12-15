import numpy as np

import pandas as pd

from scipy.sparse import csr_matrix

import torch
import torchvision
from autoPyTorch.utils.common import FitRequirement
from torch import nn
from abc import abstractmethod
from typing import Any, Dict, Iterable, Optional, Tuple, List


from autoPyTorch.pipeline.components.base_component import BaseEstimator
from autoPyTorch.pipeline.components.setup.network_backbone.utils import get_output_shape
from autoPyTorch.pipeline.components.base_component import (
    autoPyTorchComponent,
)


class EncoderNetwork(nn.Module):
    def forward(self, x: torch.Tensor, output_seq: bool = False):
        """
        Base forecasting network, its output needs to be a 2-d or 3-d Tensor:
        When the decoder is an auto-regressive model, then it needs to output a 3-d Tensor, in which case, output_seq
         needs to be set as True
        When the decoder is a seq2seq model, the network needs to output a 2-d Tensor (B, N), in which case,
        output_seq needs to be set as False

        Args:
            x: torch.Tensor(B, L_in, N)
            output_seq (bool), if the network outputs a sequence tensor. If it is set True,
            output will be a 3-d Tensor (B, L_out, N). L_out = L_in if encoder_properties['recurrent'] is True.
            If this value is set as False, the network only returns the last item of the sequence.
        Returns:
            net_output: torch.Tensor with shape either (B, N) or (B, L_out, N)

        """
        raise NotImplementedError


class BaseForecastingEncoder(autoPyTorchComponent):
    """
    Base class for network backbones. Holds the backbone module and the config which was used to create it.
    """
    _required_properties = ["name", "shortname", "handles_tabular", "handles_image", "handles_time_series"]

    def __init__(self,
                 **kwargs: Any):
        autoPyTorchComponent.__init__(self)
        self.add_fit_requirements(
            self._required_fit_arguments
        )
        self.encoder: nn.Module = None
        self.config = kwargs
        self.input_shape: Optional[Iterable] = None

    @property
    def _required_fit_arguments(self) -> List[FitRequirement]:
        return [
                FitRequirement('is_small_preprocess', (bool,), user_defined=True, dataset_property=True),
                FitRequirement('X_train', (np.ndarray, pd.DataFrame, csr_matrix), user_defined=True,
                               dataset_property=False),
                FitRequirement('time_series_transformer', (BaseEstimator,), user_defined=False, dataset_property=False),
                FitRequirement('input_shape', (Iterable,), user_defined=True, dataset_property=True),
            ]

    def fit(self, X: Dict[str, Any], y: Any = None) -> BaseEstimator:
        self.check_requirements(X, y)
        X_train = X['X_train']

        input_shape = X["dataset_properties"]['input_shape']

        if not X["dataset_properties"]["is_small_preprocess"]:
            # get input shape by transforming first two elements of the training set
            transforms = torchvision.transforms.Compose(X['preprocess_transforms'])
            X_train = X_train[:1, np.newaxis, ...]
            input_shape = transforms(X_train).shape[1:]

        if 'network_embedding' in X.keys():
            input_shape = get_output_shape(X['network_embedding'], input_shape=input_shape)
        self.input_shape = input_shape

        self.encoder = self.build_encoder(
            input_shape=input_shape,
        )

        return self

    def transform(self, X: Dict[str, Any]) -> Dict[str, Any]:
        X['dataset_properties'].update({'input_shape': self.input_shape})
        X.update({'network_encoder': self.encoder})
        X.update({'encoder_properties': self.encoder_properties})
        return X

    @abstractmethod
    def build_encoder(self, input_shape: Tuple[int, ...]) -> nn.Module:
        """
        Builds the backbone module and returns it

        Args:
            input_shape (Tuple[int, ...]): shape of the input to the backbone

        Returns:
            nn.Module: backbone module
        """
        raise NotImplementedError()

    @property
    def encoder_properties(self):
        """
        Encoder properties, this determines how the data flows over the forecasting networks

        has_hidden_states, it determines if the network contains hidden states and thus return or accept the hidden
        states
        bijective_seq_output, determines if the network returns a sequence with the same sequence length as the input
        sequence when output_seq is set True
        fix_input_shape if the input shape is fixed, this is useful for building network head
        """
        # TODO make use of bijective_seq_output in trainer!!!
        encoder_properties = {'has_hidden_states': False,
                              'bijective_seq_output': False,
                              'fixed_input_seq_length': False
                              }
        return encoder_properties

