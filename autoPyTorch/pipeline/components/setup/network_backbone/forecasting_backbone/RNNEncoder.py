from typing import Any, Dict, Optional, Tuple, List, Union
import warnings
import numpy as np

import ConfigSpace as CS
from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import (
    CategoricalHyperparameter,
    UniformFloatHyperparameter,
    UniformIntegerHyperparameter
)

import torch
from torch import nn
from gluonts.time_feature.lag import get_lags_for_frequency

from autoPyTorch.pipeline.components.base_component import BaseEstimator
from autoPyTorch.pipeline.components.setup.network_backbone.forecasting_backbone.base_forecasting_encoder import (
    BaseForecastingEncoder, EncoderNetwork
)
from autoPyTorch.utils.common import HyperparameterSearchSpace, add_hyperparameter, get_hyperparameter
from autoPyTorch.utils.forecasting_time_features import FREQUENCY_MAP


class _RNN(EncoderNetwork):
    # we only consder GRU and LSTM here
    def __init__(self,
                 in_features: int,
                 config: Dict[str, Any],
                 lagged_value: Optional[Union[List, np.ndarray]] = None):
        super().__init__()
        self.config = config
        if config['cell_type'] == 'lstm':
            cell_type = nn.LSTM
        else:
            cell_type = nn.GRU
        self.lagged_value = lagged_value
        in_features = in_features if self.lagged_value is None else len(self.lagged_value) * in_features
        self.lstm = cell_type(input_size=in_features,
                              hidden_size=config["hidden_size"],
                              num_layers=config["num_layers"],
                              dropout=config.get("dropout", 0.0),
                              bidirectional=config["bidirectional"],
                              batch_first=True)

    def forward(self,
                x: torch.Tensor,
                output_seq: bool = False,
                hx: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> Tuple[torch.Tensor, ...]:
        B, T, _ = x.shape

        outputs, hidden_state = self.lstm(x, hx)

        if output_seq:
            return outputs, hidden_state
        else:
            if not self.config["bidirectional"]:
                return outputs[:, -1, :], hidden_state
            else:
                # concatenate last forward hidden state with first backward hidden state
                outputs_by_direction = outputs.view(B,
                                                    T,
                                                    2,
                                                    self.config["hidden_size"])
                out = torch.cat([
                    outputs_by_direction[:, -1, 0, :],
                    outputs_by_direction[:, 0, 1, :]
                ], dim=-1)
                return out, hidden_state


class RNNEncoder(BaseForecastingEncoder):
    """
    Standard searchable LSTM backbone for time series data
    """
    _fixed_seq_length = False

    def __init__(self, **kwargs: Dict):
        super().__init__(**kwargs)
        self.lagged_value = [1, 2, 3, 4, 5, 6, 7]

    def build_encoder(self, input_shape: Tuple[int, ...]) -> nn.Module:
        encoder = _RNN(in_features=input_shape[-1],
                       config=self.config,
                       lagged_value=self.lagged_value)
        return encoder

    @staticmethod
    def allowed_decoders():
        """
        decoder that is compatible with the encoder
        """
        return ['MLPDecoder', 'RNNDecoder']

    def encoder_properties(self):
        encoder_properties = super().encoder_properties()
        encoder_properties.update({'has_hidden_states': True,
                                   'lagged_input': True,
                                   })
        return encoder_properties

    def fit(self, X: Dict[str, Any], y: Any = None) -> BaseEstimator:
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
                # If
                pass
        return super().fit(X, y)

    def transform(self, X: Dict[str, Any]) -> Dict[str, Any]:
        rnn_kwargs = {'hidden_size': self.config['hidden_size'],
                      'num_layers': self.config['num_layers'],
                      'bidirectional': self.config['bidirectional'],
                      'cell_type': self.config['cell_type'],
                      'dropout': self.config.get('dropout', 0.0)}  # used for initialize
        X.update({'rnn_kwargs': rnn_kwargs})
        return super().transform(X)

    @staticmethod
    def get_properties(dataset_properties: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        return {
            'shortname': 'RNNBackbone',
            'name': 'RNNBackbone',
            'handles_tabular': False,
            'handles_image': False,
            'handles_time_series': True,
        }

    @staticmethod
    def get_hyperparameter_search_space(
            dataset_properties: Optional[Dict] = None,
            cell_type: HyperparameterSearchSpace = HyperparameterSearchSpace(hyperparameter="cell_type",
                                                                             value_range=['lstm', 'gru'],
                                                                             default_value='lstm'),
            num_layers: HyperparameterSearchSpace = HyperparameterSearchSpace(hyperparameter='num_layers',
                                                                              value_range=(1, 3),
                                                                              default_value=1),
            hidden_size: HyperparameterSearchSpace = HyperparameterSearchSpace(hyperparameter='hidden_size',
                                                                               value_range=(32, 512),
                                                                               default_value=256,
                                                                               log=True),
            use_dropout: HyperparameterSearchSpace = HyperparameterSearchSpace(hyperparameter='use_dropout',
                                                                               value_range=(True, False),
                                                                               default_value=False),
            dropout: HyperparameterSearchSpace = HyperparameterSearchSpace(hyperparameter='dropout',
                                                                           value_range=(0., 0.5),
                                                                           default_value=0.2),
            bidirectional: HyperparameterSearchSpace = HyperparameterSearchSpace(hyperparameter='bidirectional',
                                                                                 value_range=(True, False),
                                                                                 default_value=True),
            decoder_type: HyperparameterSearchSpace =
            HyperparameterSearchSpace(hyperparameter='decoder_type',
                                      value_range=('MLPDecoder', 'RNNDecoder'),
                                      default_value='MLPDecoder')
    ) -> ConfigurationSpace:
        """
        get hyperparameter search space

        """
        cs = CS.ConfigurationSpace()

        # TODO consider lstm layers with different hidden size
        # TODO bidirectional needs to be set as false for DeepAR model
        num_layers = get_hyperparameter(num_layers, UniformIntegerHyperparameter)
        use_dropout = get_hyperparameter(use_dropout, CategoricalHyperparameter)
        dropout = get_hyperparameter(dropout, UniformFloatHyperparameter)
        cs.add_hyperparameters([num_layers, use_dropout, dropout])

        # Add plain hyperparameters
        add_hyperparameter(cs, cell_type, CategoricalHyperparameter)
        add_hyperparameter(cs, hidden_size, UniformIntegerHyperparameter)
        add_hyperparameter(cs, bidirectional, CategoricalHyperparameter)
        add_hyperparameter(cs, decoder_type, CategoricalHyperparameter)

        cs.add_condition(CS.AndConjunction(CS.EqualsCondition(dropout, use_dropout, True),
                                           CS.GreaterThanCondition(dropout, num_layers, 1)))

        return cs
