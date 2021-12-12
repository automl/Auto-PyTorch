from abc import abstractmethod
from autoPyTorch.pipeline.components.setup.network_backbone.base_network_backbone import NetworkBackboneComponent

import torch
from torch import nn
from abc import abstractmethod
from typing import Any, Dict, Iterable, Optional, Tuple, List


from autoPyTorch.pipeline.components.base_component import BaseEstimator


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


class BaseForecastingNetworkBackbone(NetworkBackboneComponent):
    def fit(self, X: Dict[str, Any], y: Any = None) -> BaseEstimator:
        return super().fit(X, y)

    def transform(self, X: Dict[str, Any]) -> Dict[str, Any]:
        X = super().transform(X)
        X.update({'encoder_properties': self.encoder_properties})
        return X

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

