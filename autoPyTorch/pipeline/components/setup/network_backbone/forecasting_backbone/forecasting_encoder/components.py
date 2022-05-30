from enum import Enum
from typing import NamedTuple, Tuple

import torch
from torch import nn


class EncoderProperties(NamedTuple):
    has_hidden_states: bool = False
    bijective_seq_output: bool = True
    fixed_input_seq_length: bool = False
    lagged_input: bool = False
    causality: bool = True  # this value indicates if the output of the model only depends on the past targets


class EncoderBlockInfo(NamedTuple):
    encoder: nn.Module
    encoder_properties: EncoderProperties
    encoder_input_shape: Tuple[int, ...]
    encoder_output_shape: Tuple[int, ...]
    n_hidden_states: int


class EncoderNetwork(nn.Module):
    def forward(self,
                x: torch.Tensor,
                output_seq: bool = False) -> torch.Tensor:
        """
        Base forecasting network, its output needs to be a 2-d or 3-d Tensor:
        When the decoder is an auto-regressive model, then it needs to output a 3-d Tensor, in which case, output_seq
         needs to be set as True
        When the decoder is a seq2seq model, the network needs to output a 2-d Tensor (B, N), in which case,
        output_seq needs to be set as False

        Args:
            x: torch.Tensor(B, L_in, N)
            output_seq (bool): if the network outputs a sequence tensor. If it is set True,
                output will be a 3-d Tensor (B, L_out, N). L_out = L_in if encoder_properties['recurrent'] is True.
                If this value is set as False, the network only returns the last item of the sequence.
            hx (Optional[torch.Tensor]): addational input to the network, this could be a hidden states or a sequence
                from previous inputs

        Returns:
            net_output: torch.Tensor with shape either (B, N) or (B, L_out, N)

        """
        raise NotImplementedError

    def get_last_seq_value(self, x: torch.Tensor) -> torch.Tensor:
        """
        get the last value of the sequential output
        Args:
            x: torch.Tensor(B, L, N): a sequential value output by the network, usually this value needs to be fed
                to the decoder (or a 2D tensor for a flat encoder)
        Returns:
            output: torch.Tensor(B, 1, M): last element of the sequential value (or a 2D tensor for flat encoder)

        """
        raise NotImplementedError


class EncoderOutputForm(Enum):
    NoOutput = 0
    HiddenStates = 1  # RNN -> RNN
    Sequence = 2  # Transformer -> Transformer
    SequenceLast = 3  # RNN/TCN/Transformer -> MLP
