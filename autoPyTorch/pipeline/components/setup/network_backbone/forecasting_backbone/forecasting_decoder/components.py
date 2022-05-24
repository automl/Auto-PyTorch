from typing import Optional, Tuple, NamedTuple

import torch
from torch import nn


class DecoderProperties(NamedTuple):
    has_hidden_states: bool = False
    has_local_layer: bool = True
    recurrent: bool = False
    lagged_input: bool = False
    multi_blocks: bool = False
    mask_on_future_target: bool = False


class DecoderBlockInfo(NamedTuple):
    decoder: nn.Module
    decoder_properties: DecoderProperties
    decoder_output_shape: Tuple[int, ...]
    decoder_input_shape: Tuple[int, ...]


class DecoderNetwork(nn.Module):
    def forward(self, x_future: torch.Tensor, encoder_output: torch.Tensor, pos_idx: Optional[Tuple[int]] = None):
        """
        Base forecasting Decoder Network, its output needs to be a 3-d Tensor:


        Args:
            x_future: torch.Tensor(B, L_future, N_out), the future features
            encoder_output: torch.Tensor(B, L_encoder, N), output of the encoder network, or the hidden states
        Returns:
            net_output: torch.Tensor with shape either (B, L_future, N)

        """
        raise NotImplementedError
