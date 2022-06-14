from typing import NamedTuple, Optional, Tuple

import torch
from torch import nn


class DecoderProperties(NamedTuple):
    """
    Decoder properties

    Args:
        has_hidden_states (bool):
            if the decoder has hidden states. A decoder with hidden states might have additional output and requires
            additional inputs
        has_local_layer (bool):
            if the decoder has local layer, in which case the output is also a 3D sequential feature
        recurrent (bool):
            if the decoder is recurrent. This determines if decoders can be auto-regressive
        lagged_input (bool):
            if the decoder accepts past targets as additional features
        multi_blocks (bool):
            If the decoder is stacked by multiple blocks (only for N-BEATS)
    """
    has_hidden_states: bool = False
    has_local_layer: bool = True
    recurrent: bool = False
    lagged_input: bool = False
    multi_blocks: bool = False


class DecoderBlockInfo(NamedTuple):
    """
    Decoder block infos

    Args:
        decoder (nn.Module):
            decoder network
        decoder_properties (EncoderProperties):
            decoder properties
        decoder_output_shape (Tuple[int, ...]):
            output shape that the decoder ought to output

        decoder_input_shape (Tuple[int, ...]):
            requried input shape of the decoder

    """
    decoder: nn.Module
    decoder_properties: DecoderProperties
    decoder_output_shape: Tuple[int, ...]
    decoder_input_shape: Tuple[int, ...]


class DecoderNetwork(nn.Module):
    def forward(self, x_future: torch.Tensor,
                encoder_output: torch.Tensor,
                pos_idx: Optional[Tuple[int]] = None) -> torch.Tensor:
        """
        Base forecasting Decoder Network, its output needs to be a 3-d Tensor:


        Args:
            x_future: torch.Tensor(B, L_future, N_out), the future features
            encoder_output: torch.Tensor(B, L_encoder, N), output of the encoder network, or the hidden states
            pos_idx: positional index, indicating the position of the forecasted tensor, used for transformer
        Returns:
            net_output: torch.Tensor with shape either (B, L_future, N)

        """
        raise NotImplementedError
