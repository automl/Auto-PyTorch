from typing import Dict, Any, Optional
import torch
from torch import nn
from functools import partial
import torch.nn.functional as F
import math
from pytorch_forecasting.models.temporal_fusion_transformer.sub_modules import (
    TimeDistributed, TimeDistributedInterpolation, GatedLinearUnit, ResampleNorm, AddNorm, GateAddNorm,
    GatedResidualNetwork, VariableSelectionNetwork, InterpretableMultiHeadAttention
)


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


class TunableAddNorm(AddNorm):
    def __init__(self, input_size: int, skip_size: int = None, trainable_add: bool = True,
                 layer_norm_eps: float = 1e-5):
        super(TunableAddNorm, self).__init__(input_size, skip_size, trainable_add)
        self.norm = nn.LayerNorm(self.input_size, eps=layer_norm_eps)


class TunableGateAddNorm(GateAddNorm):
    def __init__(self, input_size: int, hidden_size: int = None, skip_size: int = None, trainable_add: bool = False,
                 dropout: Optional[float] = None, layer_norm_eps: float = 1e-5):
        super().__init__(input_size, hidden_size, skip_size, trainable_add, dropout)
        self.add_norm = TunableAddNorm(self.hidden_size, skip_size=self.skip_size, trainable_add=trainable_add,
                                       layer_norm_eps=layer_norm_eps)


class TunableGatedResidualNetwork(GatedResidualNetwork):
    def __init__(self, input_size: int, hidden_size: int, output_size: int, dropout: float = 0.1,
                 context_size: int = None, residual: bool = False, layer_norm_eps: float = 1e-5):
        super().__init__(input_size, hidden_size, output_size, dropout, context_size, residual)
        self.gate_norm = TunableGateAddNorm(
            input_size=self.hidden_size,
            skip_size=self.output_size,
            hidden_size=self.output_size,
            dropout=self.dropout,
            trainable_add=False,
            layer_norm_eps=layer_norm_eps
        )


class InterpretableMultiAttentionEncoderLayer(nn.Module):
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1,
                 layer_norm_eps: float = 1e-5, device=None, dtype=None) -> None:
        self.multi_attention = InterpretableMultiHeadAttention(n_head=nhead, d_model=d_model, dropout=dropout)
        self.post_attn_gate_norm = TunableGateAddNorm(input_size=d_model,
                                                      hidden_size=dim_feedforward,
                                                      dropout=dropout,
                                                      trainable_add=False,
                                                      layer_norm_eps=layer_norm_eps
                                                      )
        self.pos_wise_ff = TunableGatedResidualNetwork(
            self.hparams.hidden_size, self.hparams.hidden_size, self.hparams.hidden_size, dropout=self.hparams.dropout
        )


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
