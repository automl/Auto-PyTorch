from typing import Dict, Any, Optional
import torch
from torch import nn
from functools import partial
import torch.nn.functional as F
import math

from autoPyTorch.pipeline.components.setup.network_backbone.forecasting_backbone. \
    forecasting_encoder.base_forecasting_encoder import (
    NetworkStructure,
    EncoderBlockInfo
)
from autoPyTorch.pipeline.components.setup.network_backbone.forecasting_backbone. \
    forecasting_decoder.base_forecasting_decoder import DecoderBlockInfo

from pytorch_forecasting.models.temporal_fusion_transformer.sub_modules import (
    TimeDistributed, TimeDistributedInterpolation, GatedLinearUnit, ResampleNorm, AddNorm, GateAddNorm,
    GatedResidualNetwork, VariableSelectionNetwork, InterpretableMultiHeadAttention,
)
from pytorch_forecasting.utils import create_mask


class AddLayer(nn.Module):
    def __init__(self, input_size: int, skip_size: int):
        super().__init__()
        if input_size == skip_size:
            self.fc = nn.Linear(skip_size, input_size)
        self.norm = nn.LayerNorm(input_size)

    def forward(self, input: torch.Tensor, skip: torch.Tensor):
        if hasattr(self, 'fc'):
            return self.norm(input + self.fc(skip))
        else:
            return self.norm(input)


class TemporalFusionLayer(nn.Module):
    """
    (Lim et al.
    Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting,
    https://arxiv.org/abs/1912.09363)
    we follow the implementation from pytorch forecasting:
    https://github.com/jdb78/pytorch-forecasting/blob/master/pytorch_forecasting/models/temporal_fusion_transformer/__init__.py
    """

    def __init__(self,
                 window_size: int,
                 n_prediction_steps: int,
                 network_structure: NetworkStructure,
                 network_encoder: Dict[str, EncoderBlockInfo],
                 n_decoder_output_features: int,
                 d_model: int,
                 n_head: int,
                 dropout: Optional[float] = None):
        super().__init__()
        num_blocks = network_structure.num_blocks
        last_block = f'block_{num_blocks}'
        n_encoder_output = network_encoder[last_block].encoder_output_shape_[-1]
        self.window_size = window_size
        self.n_prediction_steps = n_prediction_steps
        self.timestep = window_size + n_prediction_steps

        if n_decoder_output_features != n_encoder_output:
            self.decoder_proj_layer = nn.Linear(n_decoder_output_features, n_encoder_output, bias=False)
        else:
            self.decoder_proj_layer = None
        if network_structure.variable_selection:
            if network_structure.skip_connection:
                # static feature selector needs to generate the same number of features as the output of the encoder
                n_encoder_output_first = network_encoder['block_1'].encoder_output_shape_[-1]
                self.static_context_enrichment = GatedResidualNetwork(
                    n_encoder_output_first, n_encoder_output_first, n_encoder_output_first, dropout
                )
                self.enrichment = GatedResidualNetwork(
                    input_size=n_encoder_output,
                    hidden_size=n_encoder_output,
                    output_size=d_model,
                    dropout=dropout,
                    context_size=n_encoder_output_first,
                    residual=True,
                )
                self.enrich_with_static = True
        if not hasattr(self, 'enrichment'):
            self.enrichment = GatedResidualNetwork(
                input_size=n_encoder_output,
                hidden_size=n_encoder_output,
                output_size=d_model,
                dropout=self.dropout_rate if self.use_dropout else None,
                residual=True,
            )
            self.enrich_with_static = False

        self.attention_fusion = InterpretableMultiHeadAttention(
            d_model=d_model,
            n_head=n_head,
            dropout=dropout
        )
        self.post_attn_gate_norm = GateAddNorm(d_model, dropout=dropout, trainable_add=False)
        self.pos_wise_ff = GatedResidualNetwork(input_size=d_model, hidden_size=d_model,
                                                output_size=d_model, dropout=self.hparams.dropout)

        self.network_structure = network_structure
        if network_structure.skip_connection:
            if network_structure.skip_connection_type == 'add':
                self.residual_connection = AddLayer(d_model, n_encoder_output)
            elif network_structure.skip_connection_type == 'gate_add_norm':
                self.residual_connection = GateAddNorm(d_model, skip_size=n_encoder_output,
                                                       dropout=None, trainable_add=False)

    def forward(self, encoder_output: torch.Tensor, decoder_output: torch.Tensor, encoder_lengths: torch.LongTensor,
                static_embedding: Optional[torch.Tensor] = None):
        """
        Args:
            encoder_output: the output of the last layer of encoder network
            decoder_output: the output of the last layer of decoder network
            encoder_lengths: length of encoder network
            static_embedding: output of static variable selection network (if applible)
        """
        if self.decoder_proj_layer is not None:
            decoder_output = self.decoder_proj_layer(decoder_output)
        network_output = torch.cat([encoder_output, decoder_output], dim=1)

        if self.enrich_with_static:
            static_context_enrichment = self.static_context_enrichment(static_embedding)
            attn_input = self.enrichment(
                network_output, static_context_enrichment[:, None].expand(-1, self.timesteps, -1)
            )
        else:
            attn_input = self.enrichment(network_output)

        # Attention
        attn_output, attn_output_weights = self.attention_fusion(
            q=attn_input[:, self.window_size:],  # query only for predictions
            k=attn_input,
            v=attn_input,
            mask=self.get_attention_mask(
                encoder_lengths=encoder_lengths, decoder_length=self.n_prediction_steps
            ),
        )
        # skip connection over attention
        attn_output = self.post_attn_gate_norm(attn_output, attn_input[:, self.window_size:])
        output = self.pos_wise_ff(attn_output)

        if self.network_structure.skip_connection:
            return self.residual_connection(output, decoder_output)
        else:
            return output

    def get_attention_mask(self, encoder_lengths: torch.LongTensor, decoder_length: int):
        """
        https://github.com/jdb78/pytorch-forecasting/blob/master/pytorch_forecasting/models/
        temporal_fusion_transformer/__init__.py
        """
        # indices to which is attended
        attend_step = torch.arange(decoder_length, device=self.device)
        # indices for which is predicted
        predict_step = torch.arange(0, decoder_length, device=self.device)[:, None]
        # do not attend to steps to self or after prediction
        # todo: there is potential value in attending to future forecasts if they are made with knowledge currently
        #   available
        #   one possibility is here to use a second attention layer for future attention (assuming different effects
        #   matter in the future than the past)
        #   or alternatively using the same layer but allowing forward attention - i.e. only masking out non-available
        #   data and self
        decoder_mask = attend_step >= predict_step
        # do not attend to steps where data is padded
        encoder_mask = create_mask(encoder_lengths.max(), encoder_lengths)
        # combine masks along attended time - first encoder and then decoder
        mask = torch.cat(
            (
                encoder_mask.unsqueeze(1).expand(-1, decoder_length, -1),
                decoder_mask.unsqueeze(0).expand(encoder_lengths.size(0), -1, -1),
            ),
            dim=2,
        )
        return mask


def build_transformer_layers(d_model: int, config: Dict[str, Any], layer_type='encoder'):
    nhead = 2 ** config['n_head_log']
    dim_feedforward = 2 ** config['d_feed_forward_log']
    dropout = config.get('dropout', 0.0)
    activation = config['activation']
    layer_norm_eps = config['layer_norm_eps']
    if layer_type == 'encoder':
        return nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
                                          dropout=dropout, activation=activation,
                                          layer_norm_eps=layer_norm_eps, batch_first=True)
    elif layer_type == 'decoder':
        return nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
                                          dropout=dropout, activation=activation,
                                          layer_norm_eps=layer_norm_eps, batch_first=True)
    else:
        raise ValueError('layer_type must be encoder or decoder!')


# https://github.com/pytorch/examples/blob/master/word_language_model/model.py
class PositionalEncoding(nn.Module):
    r"""
        NOTE: different from the raw implementation, this model is designed for the batch_first inputs!
        Inject some information about the relative or absolute position of the tokens
        in the sequence. The positional encodings have the same dimension as
        the embeddings, so that the two can be summed. Here, we use sine and cosine
        functions of different frequencies.
    .. math::
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    Examples:
        >>> pos_encoder = PositionalEncoding(d_model)
    """

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [batch size, sequence length embed dim]
            output: [batch size, sequence length, embed dim]
        Examples:
            >>> output = pos_encoder(x)
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)
