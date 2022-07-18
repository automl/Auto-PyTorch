from typing import Any, Dict, List, Optional, Tuple, Union

import ConfigSpace as CS
from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import Constant

import numpy as np

import torch
from torch import nn

from autoPyTorch.datasets.base_dataset import BaseDatasetPropertiesType
from autoPyTorch.pipeline.components.base_component import BaseEstimator
from autoPyTorch.pipeline.components.setup.network_backbone.forecasting_backbone.forecasting_decoder.\
    base_forecasting_decoder import BaseForecastingDecoder, DecoderProperties
from autoPyTorch.pipeline.components.setup.network_backbone.forecasting_backbone.forecasting_decoder.components import \
    DecoderNetwork
from autoPyTorch.utils.common import FitRequirement


class RNN_Module(DecoderNetwork):
    def __init__(self,
                 in_features: int,
                 hidden_size: int,
                 num_layers: int,
                 cell_type: str,
                 dropout: float,
                 lagged_value: Optional[Union[List, np.ndarray]] = None):
        super().__init__()
        if cell_type == 'lstm':
            cell = nn.LSTM
        else:
            cell = nn.GRU
        self.lagged_value = lagged_value
        in_features = in_features
        self.lstm = cell(input_size=in_features,
                         hidden_size=hidden_size,
                         num_layers=num_layers,
                         dropout=dropout,
                         bidirectional=False,
                         batch_first=True)

    def forward(self,
                x_future: torch.Tensor,
                encoder_output: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                pos_idx: Optional[Tuple[int]] = None) -> Tuple[torch.Tensor, ...]:
        if x_future.ndim == 2:
            x_future = x_future.unsqueeze(1)
        outputs, hidden_state, = self.lstm(x_future, encoder_output)
        return outputs, hidden_state


class ForecastingRNNDecoder(BaseForecastingDecoder):
    """
    Standard searchable RNN decoder for time series data, only works when the encoder is an RNN encoder
    """

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        # RNN is naturally auto-regressive. However, we will not consider it as a decoder for deep AR model
        self.rnn_kwargs: Optional[Dict] = None
        self.lagged_value = [1, 2, 3, 4, 5, 6, 7]
        self.add_fit_requirements([FitRequirement('rnn_kwargs', (Dict,), user_defined=False, dataset_property=False)])

    def _build_decoder(self,
                       encoder_output_shape: Tuple[int, ...],
                       future_variable_input: Tuple[int, ...],
                       n_prediction_heads: int,
                       dataset_properties: Dict) -> Tuple[nn.Module, int]:
        assert self.rnn_kwargs is not None
        # RNN decoder only allows RNN encoder, these parameters need to exists.
        hidden_size = self.rnn_kwargs['hidden_size']
        num_layers = self.rnn_kwargs['num_layers']
        cell_type = self.rnn_kwargs['cell_type']
        dropout = self.rnn_kwargs['dropout']
        decoder = RNN_Module(in_features=future_variable_input[-1],
                             hidden_size=hidden_size,
                             num_layers=num_layers,
                             cell_type=cell_type,
                             dropout=dropout,
                             lagged_value=self.lagged_value
                             )
        return decoder, hidden_size

    @property
    def fitted_encoder(self) -> List[str]:
        return ['RNNEncoder']

    @staticmethod
    def decoder_properties() -> DecoderProperties:
        decoder_properties = DecoderProperties(has_hidden_states=True,
                                               recurrent=True,
                                               lagged_input=True)
        return decoder_properties

    def fit(self, X: Dict[str, Any], y: Any = None) -> BaseEstimator:
        self.rnn_kwargs = X['rnn_kwargs']
        if 'lagged_value' in X['dataset_properties']:
            self.lagged_value = X['dataset_properties']['lagged_value']
        return super().fit(X, y)

    @staticmethod
    def get_properties(dataset_properties: Optional[Dict[str, BaseDatasetPropertiesType]] = None) -> Dict[str, Any]:
        return {
            'shortname': 'RNNDecoder',
            'name': 'RNNDecoder',
            'handles_tabular': False,
            'handles_image': False,
            'handles_time_series': True,
        }

    @staticmethod
    def get_hyperparameter_search_space(
            dataset_properties: Optional[Dict] = None,
    ) -> ConfigurationSpace:
        cs = CS.ConfigurationSpace()
        cs.add_hyperparameter(Constant('decoder_type', 'RNNDecoder'))  # this helps the encoder to recognize the decoder
        return cs
