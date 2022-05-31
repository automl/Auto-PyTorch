from typing import Any, Dict, List, Optional, Tuple, Union

import ConfigSpace as CS
from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import (CategoricalHyperparameter,
                                         UniformFloatHyperparameter,
                                         UniformIntegerHyperparameter)

import torch
from torch import nn

from autoPyTorch.datasets.base_dataset import BaseDatasetPropertiesType
from autoPyTorch.pipeline.components.base_component import BaseEstimator
from autoPyTorch.pipeline.components.setup.network_backbone.forecasting_backbone.forecasting_encoder.\
    base_forecasting_encoder import BaseForecastingEncoder
from autoPyTorch.pipeline.components.setup.network_backbone.forecasting_backbone.forecasting_encoder.components import (
    EncoderNetwork,
    EncoderProperties
)
from autoPyTorch.utils.common import (
    HyperparameterSearchSpace,
    add_hyperparameter,
    get_hyperparameter
)


class _RNN(EncoderNetwork):
    # we only consder GRU and LSTM here
    def __init__(self,
                 in_features: int,
                 config: Dict[str, Any],
                 lagged_value: Optional[List[int]] = None):
        super().__init__()
        if lagged_value is None:
            self.lagged_value = [0]
        else:
            self.lagged_value = lagged_value
        self.config = config
        if config['cell_type'] == 'lstm':
            cell_type = nn.LSTM
        else:
            cell_type = nn.GRU
        self.lstm = cell_type(input_size=in_features,
                              hidden_size=config["hidden_size"],
                              num_layers=config["num_layers"],
                              dropout=config.get("dropout", 0.0),
                              bidirectional=config["bidirectional"],
                              batch_first=True)
        self.cell_type = config['cell_type']

    def forward(self,
                x: torch.Tensor,
                output_seq: bool = False,
                hx: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> Tuple[torch.Tensor, ...]:
        B, T, _ = x.shape

        x, hidden_state = self.lstm(x, hx)

        if output_seq:
            return x, hidden_state
        else:
            return self.get_last_seq_value(x), hidden_state

    def get_last_seq_value(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        if not self.config["bidirectional"]:
            return x[:, -1:, ]
        else:
            x_by_direction = x.view(B,
                                    T,
                                    2,
                                    self.config["hidden_size"])
            x = torch.cat([
                x_by_direction[:, -1, [0], :],
                x_by_direction[:, 0, [1], :]
            ], dim=-1)
            return x


class RNNEncoder(BaseForecastingEncoder):
    """
    Standard searchable LSTM backbone for time series data
    """
    _fixed_seq_length = False

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self.lagged_value = [1, 2, 3, 4, 5, 6, 7]

    def build_encoder(self, input_shape: Tuple[int, ...]) -> nn.Module:
        in_features = input_shape[-1]
        encoder = _RNN(in_features=in_features,
                       config=self.config,
                       lagged_value=self.lagged_value,
                       )
        return encoder

    def n_encoder_output_feature(self) -> int:
        hidden_size: int = self.config['hidden_size']
        return 2 * hidden_size if self.config['bidirectional'] else hidden_size

    def n_hidden_states(self) -> int:
        if self.config['cell_type'] == 'lstm':
            return 2
        elif self.config['cell_type'] == 'gru':
            return 1
        else:
            raise NotImplementedError

    @staticmethod
    def allowed_decoders() -> List[str]:
        """
        decoder that is compatible with the encoder
        """
        return ['MLPDecoder', 'RNNDecoder']

    @staticmethod
    def encoder_properties() -> EncoderProperties:
        return EncoderProperties(has_hidden_states=True, lagged_input=True)

    def fit(self, X: Dict[str, Any], y: Any = None) -> BaseEstimator:
        if 'lagged_value' in X['dataset_properties']:
            self.lagged_value = X['dataset_properties']['lagged_value']
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
    def get_properties(
            dataset_properties: Optional[Dict[str, BaseDatasetPropertiesType]] = None) -> Dict[str, Union[str, bool]]:
        return {
            'shortname': 'RNNEncoder',
            'name': 'RNNEncoder',
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
                                                                               default_value=64,
                                                                               log=True),
            use_dropout: HyperparameterSearchSpace = HyperparameterSearchSpace(hyperparameter='use_dropout',
                                                                               value_range=(True, False),
                                                                               default_value=False),
            dropout: HyperparameterSearchSpace = HyperparameterSearchSpace(hyperparameter='dropout',
                                                                           value_range=(0., 0.5),
                                                                           default_value=0.1),
            bidirectional: HyperparameterSearchSpace = HyperparameterSearchSpace(hyperparameter='bidirectional',
                                                                                 value_range=(False,),
                                                                                 default_value=False),
            decoder_type: HyperparameterSearchSpace =
            HyperparameterSearchSpace(hyperparameter='decoder_type',
                                      value_range=('MLPDecoder', 'RNNDecoder'),
                                      default_value='MLPDecoder')
    ) -> ConfigurationSpace:
        """
        get hyperparameter search space, bidirectional is not casual so I do not allow it to be set as True,
        However, it might be further implemented to NLP tasks

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
