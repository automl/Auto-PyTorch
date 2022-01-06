from typing import Any, Dict, Optional, Tuple, List, Union
import warnings

import ConfigSpace as CS
from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import Constant


import torch
from torch import nn

import numpy as np
from gluonts.time_feature.lag import get_lags_for_frequency

from autoPyTorch.datasets.base_dataset import BaseDatasetPropertiesType
from autoPyTorch.pipeline.components.base_component import BaseEstimator
from autoPyTorch.pipeline.components.setup.network_backbone.forecasting_backbone.forecasting_decoder.base_forecasting_decoder import \
    BaseForecastingDecoder

from autoPyTorch.utils.common import HyperparameterSearchSpace, get_hyperparameter, FitRequirement
from autoPyTorch.utils.forecasting_time_features import FREQUENCY_MAP


class RNN_Module(nn.Module):
    def __init__(self,
                 in_features: int,
                 hidden_size: int,
                 num_layers: int,
                 cell_type: str,
                 dropout: float,
                 lagged_value: Optional[Union[List, np.ndarray]]=None):
        super().__init__()
        if cell_type == 'lstm':
            cell = nn.LSTM
        else:
            cell = nn.GRU
        self.lagged_value = lagged_value
        in_features = in_features if self.lagged_value is None else len(self.lagged_value) * in_features
        self.lstm = cell(input_size=in_features,
                         hidden_size=hidden_size,
                         num_layers=num_layers,
                         dropout=dropout,
                         bidirectional=False,
                         batch_first=True)

    def forward(self, x: torch.Tensor,
                hx: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> Tuple[torch.Tensor, ...]:
        if x.ndim == 2:
            x = x.unsqueeze(1)
        outputs, hidden_state, = self.lstm(x, hx)
        return outputs, hidden_state


class ForecastingRNNDecoder(BaseForecastingDecoder):
    """
    Standard searchable RNN decoder for time series data, only works when the encoder is
    """

    def __init__(self, **kwargs: Dict):
        super().__init__(**kwargs)
        # RNN is naturally auto-regressive. However, we will not consider it as a decoder for deep AR model
        self.auto_regressive = True
        self.rnn_kwargs = None
        self.lagged_value = [0, 1, 2, 3, 4, 5, 6, 7]

    @property
    def _required_fit_requirements(self) -> List[FitRequirement]:
        fit_requirement = super(ForecastingRNNDecoder, self)._required_fit_requirements
        fit_requirement.append(FitRequirement('rnn_kwargs', (Dict,), user_defined=False, dataset_property=False))
        return fit_requirement

    def _build_decoder(self,
                       input_shape: Tuple[int, ...],
                       n_prediction_heads: int,
                       dataset_properties: Dict) -> Tuple[nn.Module, int]:
        # RNN decoder only allows RNN encoder, these parameters need to exists.
        hidden_size = self.rnn_kwargs['hidden_size']
        num_layers = 2 * self.rnn_kwargs['num_layers'] if self.rnn_kwargs['bidirectional'] else self.rnn_kwargs[
            'num_layers']
        cell_type = self.rnn_kwargs['cell_type']
        dropout = self.rnn_kwargs['dropout']
        decoder = RNN_Module(in_features=dataset_properties['output_shape'][-1],
                             hidden_size=hidden_size,
                             num_layers=num_layers,
                             cell_type=cell_type,
                             dropout=dropout,
                             lagged_value=self.lagged_value
                             )
        return decoder, hidden_size

    @property
    def fitted_encoder(self):
        return ['RNNEncoder']

    def decoder_properties(self):
        decoder_properties = super().decoder_properties()
        decoder_properties.update({'has_hidden_states': True,
                                   'recurrent': True,
                                   'lagged_input': True,
                                   })
        return decoder_properties

    def fit(self, X: Dict[str, Any], y: Any = None) -> BaseEstimator:
        self.rnn_kwargs = X['rnn_kwargs']

        freq = X['dataset_properties'].get('freq', None)
        if 'lagged_value' in X['dataset_properties']:
            self.lagged_value = X['dataset_properties']['lagged_value']
        if freq is not None:
            try:
                freq = FREQUENCY_MAP[freq]
                lagged_values = get_lags_for_frequency(freq)
                self.lagged_value = [0] + lagged_values
            except Exception:
                warnings.warn(f'cannot find the proper lagged value for {freq}, we use the default lagged value')
                pass
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
        cs.add_hyperparameter(Constant('decoder_type', 'RNNDecoder'))  # this helps the encoder to recognize the decoder.
        return cs
