from typing import Dict, Any
import torch
from torch import nn
import math


def build_transformer_layers(d_model: int, config: Dict[str, Any], layer_id: int, layer_type='encoder'):
    nhead = 2 ** config['num_head_log_%d' % layer_id]
    dim_feedforward = 2 ** config['d_feed_forward_log_%d' % layer_id]
    dropout = config['dropout_%d' % layer_id]
    activation = config['activation']
    layer_norm_eps = config['layer_norm_eps_%d' % layer_id]
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
