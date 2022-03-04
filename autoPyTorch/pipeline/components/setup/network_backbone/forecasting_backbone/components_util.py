import math
from sklearn.base import BaseEstimator

from typing import Any, Dict, NamedTuple

import torch
from torch import nn


class NetworkStructure(NamedTuple):
    num_blocks: int = 1
    variable_selection: bool = False
    share_single_variable_networks: bool = False
    use_temporal_fusion: bool = False,
    skip_connection: bool = False
    skip_connection_type: str = "add"  # could be 'add' or 'gate_add_norm'
    grn_dropout_rate: float = 0.0


class ForecastingNetworkStructure(BaseEstimator):
    def __init__(self,
                 num_blocks: int = 1,
                 variable_selection: bool = False,
                 share_single_variable_networks: bool = False,
                 use_temporal_fusion: bool = False,
                 skip_connection: bool = False,
                 skip_connection_type: str = "add",
                 grn_dropout_rate: float = 0.0,
                 ) -> None:
        super().__init__()
        self.network_structure = NetworkStructure(num_blocks=num_blocks,
                                                  variable_selection=variable_selection,
                                                  share_single_variable_networks=share_single_variable_networks,
                                                  use_temporal_fusion=use_temporal_fusion,
                                                  skip_connection=skip_connection,
                                                  skip_connection_type=skip_connection_type,
                                                  grn_dropout_rate=grn_dropout_rate)

    def fit(self, X: Dict[str, Any], y: Any = None) -> "ForecastingNetworkStructure":
        return self

    def transform(self, X: Dict[str, Any]) -> Dict[str, Any]:
        X.update({
            'network_structure': self.network_structure,
        })
        return X

    def __str__(self) -> str:
        """ Allow a nice understanding of what components where used """
        string = self.__class__.__name__
        return string


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
