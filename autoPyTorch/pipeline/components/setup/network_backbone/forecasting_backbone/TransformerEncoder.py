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
from autoPyTorch.pipeline.components.setup.network_backbone.forecasting_backbone.transformer_util import \
    PositionalEncoding, build_transformer_layers


class _TransformerEncoder(EncoderNetwork):
    def __init__(self,
                 in_features: int,
                 d_model: int,
                 transformer_encoder_layers: [nn.Module],
                 use_positional_encoder: bool,
                 use_layer_norm_output: bool,
                 dropout_pe: float = 0.0,
                 layer_norm_eps_output: Optional[float] = None,
                 lagged_value: Optional[Union[List, np.ndarray]] = None):
        super().__init__()
        self.lagged_value = lagged_value
        in_features = in_features if self.lagged_value is None else len(self.lagged_value) * in_features

        self.input_layer = [nn.Linear(in_features, d_model, bias=False)]
        if use_positional_encoder:
            self.input_layer.append(PositionalEncoding(d_model, dropout_pe))
        self.input_layer = nn.Sequential(*self.input_layer)

        self.transformer_encoder_layers = nn.ModuleList(transformer_encoder_layers)

        self.use_layer_norm_output = use_layer_norm_output
        if use_layer_norm_output:
            self.norm_output = nn.LayerNorm(d_model, eps=layer_norm_eps_output)

    def forward(self,
                x: torch.Tensor,
                output_seq: bool = False,
                mask: Optional[torch.Tensor] = None,
                src_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.input_layer(x)

        for encoder_layer in self.transformer_encoder_layers:
            x = encoder_layer(x, mask, src_key_padding_mask)
        if self.use_layer_norm_output:
            x = self.norm_output(x)
        if output_seq:
            return x
        else:
            return x[:, -1, :]


class TransformerEncoder(BaseForecastingEncoder):
    """
    Standard searchable Transformer Encoder for time series data
    """
    _fixed_seq_length = False

    def __init__(self, **kwargs: Dict):
        super().__init__(**kwargs)
        self.lagged_value = [1, 2, 3, 4, 5, 6, 7]

    def build_encoder(self, input_shape: Tuple[int, ...]) -> nn.Module:
        d_model = 2 ** self.config['d_model_log']
        transformer_encoder_layers = []
        for layer_id in range(1, self.config['num_layers'] + 1):
            new_layer = build_transformer_layers(d_model=d_model, config=self.config,
                                                 layer_id=layer_id, layer_type='encoder')
            transformer_encoder_layers.append(new_layer)

        encoder = _TransformerEncoder(in_features=input_shape[-1],
                                      d_model=d_model,
                                      transformer_encoder_layers=transformer_encoder_layers,
                                      use_positional_encoder=self.config['use_positional_encoder'],
                                      use_layer_norm_output=self.config['use_layer_norm_output'],
                                      dropout_pe=self.config.get('dropout_positional_encoder', 0.0),
                                      layer_norm_eps_output=self.config.get('layer_norm_eps_output', None),
                                      lagged_value=self.lagged_value)
        return encoder

    @staticmethod
    def allowed_decoders():
        """
        decoder that is compatible with the encoder
        """
        return ['MLPDecoder', 'TransformerDecoder']

    def encoder_properties(self):
        encoder_properties = super().encoder_properties()
        encoder_properties.update({'lagged_input': True,
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
        transformer_encoder_kwargs = {'d_model_log': self.config['d_model_log']}  # used for initialize
        X.update({'transformer_encoder_kwargs': transformer_encoder_kwargs})
        return super().transform(X)

    @staticmethod
    def get_properties(dataset_properties: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        return {
            'shortname': 'TransformerEncoder',
            'name': 'TransformerEncoder',
            'handles_tabular': False,
            'handles_image': False,
            'handles_time_series': True,
        }

    @staticmethod
    def get_hyperparameter_search_space(
            dataset_properties: Optional[Dict] = None,
            num_layers: HyperparameterSearchSpace =
            HyperparameterSearchSpace(hyperparameter='num_layers',
                                      value_range=(1, 4),
                                      default_value=1),
            n_head_log: HyperparameterSearchSpace =
            HyperparameterSearchSpace(hyperparameter='n_head_log',
                                      value_range=(1, 4),
                                      default_value=3),
            d_model_log: HyperparameterSearchSpace =
            HyperparameterSearchSpace(hyperparameter='d_model_log',
                                      value_range=(4, 9),
                                      default_value=5),
            d_feed_forward_log: HyperparameterSearchSpace =
            HyperparameterSearchSpace(hyperparameter='d_feed_forward_log',
                                      value_range=(6, 12),
                                      default_value=7),
            layer_norm_eps: HyperparameterSearchSpace =
            HyperparameterSearchSpace(hyperparameter='layer_norm_eps',
                                      value_range=(1e-7, 1e-3),
                                      default_value=1e-5,
                                      log=True),
            use_positional_encoder: HyperparameterSearchSpace =
            HyperparameterSearchSpace(hyperparameter='use_positional_encoder',
                                      value_range=(True, False),
                                      default_value=True),
            use_layer_norm_output: HyperparameterSearchSpace =
            HyperparameterSearchSpace(hyperparameter='use_layer_norm_output',
                                      value_range=(True, False),
                                      default_value=True),
            activation: HyperparameterSearchSpace =
            HyperparameterSearchSpace(hyperparameter="activation",
                                      value_range=('relu', 'gelu'),
                                      default_value='relu',
                                      ),
            use_dropout: HyperparameterSearchSpace =
            HyperparameterSearchSpace(hyperparameter="use_dropout",
                                      value_range=(True, False),
                                      default_value=False,
                                      ),
            dropout: HyperparameterSearchSpace =
            HyperparameterSearchSpace(hyperparameter="dropout",
                                      value_range=(0, 0.8),
                                      default_value=0.5,
                                      ),
            decoder_type: HyperparameterSearchSpace =
            HyperparameterSearchSpace(hyperparameter='decoder_type',
                                      value_range=('MLPDecoder', 'TransformerDecoder'),
                                      default_value='TransformerDecoder')
    ) -> ConfigurationSpace:
        """
        get hyperparameter search space for Transformer, Given that d_model must be a multiple of n_head_log, we
        consider their log value (with base 2) as the hyperparameters

        """
        cs = CS.ConfigurationSpace()

        add_hyperparameter(cs, activation, CategoricalHyperparameter)
        add_hyperparameter(cs, d_model_log, UniformIntegerHyperparameter)

        min_transformer_layers, max_transformer_layers = num_layers.value_range

        num_layers = get_hyperparameter(num_layers, UniformIntegerHyperparameter)
        use_dropout = get_hyperparameter(use_dropout, CategoricalHyperparameter)

        # We can have dropout in the network for
        # better generalization
        use_positional_encoder = get_hyperparameter(use_positional_encoder, CategoricalHyperparameter)

        dropout_pe = HyperparameterSearchSpace(hyperparameter='dropout_positional_encoder',
                                               value_range=dropout.value_range,
                                               default_value=dropout.default_value,
                                               log=dropout.log)
        dropout_pe = get_hyperparameter(dropout_pe, UniformFloatHyperparameter)

        cs.add_hyperparameters([num_layers, use_dropout, use_positional_encoder, dropout_pe])
        cs.add_condition(CS.AndConjunction(
            CS.EqualsCondition(dropout_pe, use_dropout, True),
            CS.EqualsCondition(dropout_pe, use_positional_encoder, True)
        ))

        for i in range(1, int(max_transformer_layers) + 1):
            n_head_log_search_space = HyperparameterSearchSpace(hyperparameter='num_head_log_%d' % i,
                                                                value_range=n_head_log.value_range,
                                                                default_value=n_head_log.default_value,
                                                                log=n_head_log.log)
            d_feed_forward_log_search_space = HyperparameterSearchSpace(hyperparameter='d_feed_forward_log_%d' % i,
                                                                        value_range=d_feed_forward_log.value_range,
                                                                        default_value=d_feed_forward_log.default_value)

            layer_norm_eps_search_space = HyperparameterSearchSpace(hyperparameter='layer_norm_eps_%d' % i,
                                                                    value_range=layer_norm_eps.value_range,
                                                                    default_value=layer_norm_eps.default_value,
                                                                    log=layer_norm_eps.log)

            n_head_log_hp = get_hyperparameter(n_head_log_search_space, UniformIntegerHyperparameter)
            d_feed_forward_log_hp = get_hyperparameter(d_feed_forward_log_search_space, UniformIntegerHyperparameter)
            layer_norm_eps_hp = get_hyperparameter(layer_norm_eps_search_space, UniformFloatHyperparameter)

            layers_dims = [n_head_log_hp, d_feed_forward_log_hp, layer_norm_eps_hp]

            cs.add_hyperparameters(layers_dims)

            if i > int(min_transformer_layers):
                # The units of layer i should only exist
                # if there are at least i layers
                cs.add_conditions([
                    CS.GreaterThanCondition(hp_layer, num_layers, i - 1) for hp_layer in layers_dims
                ])
            dropout_search_space = HyperparameterSearchSpace(hyperparameter='dropout_%d' % i,
                                                             value_range=dropout.value_range,
                                                             default_value=dropout.default_value,
                                                             log=dropout.log)
            dropout_hp = get_hyperparameter(dropout_search_space, UniformFloatHyperparameter)
            cs.add_hyperparameter(dropout_hp)

            dropout_condition_1 = CS.EqualsCondition(dropout_hp, use_dropout, True)

            if i > int(min_transformer_layers):
                dropout_condition_2 = CS.GreaterThanCondition(dropout_hp, num_layers, i - 1)
                cs.add_condition(CS.AndConjunction(dropout_condition_1, dropout_condition_2))
            else:
                cs.add_condition(dropout_condition_1)

        use_layer_norm_output = get_hyperparameter(use_layer_norm_output, CategoricalHyperparameter)
        layer_norm_eps_output = HyperparameterSearchSpace(hyperparameter='layer_norm_eps_output',
                                                          value_range=layer_norm_eps.value_range,
                                                          default_value=layer_norm_eps.default_value,
                                                          log=layer_norm_eps.log)

        layer_norm_eps_output = get_hyperparameter(layer_norm_eps_output, UniformFloatHyperparameter)
        cs.add_hyperparameters([use_layer_norm_output, layer_norm_eps_output])
        cs.add_condition(CS.EqualsCondition(layer_norm_eps_output, use_layer_norm_output, True))

        add_hyperparameter(cs, decoder_type, CategoricalHyperparameter)

        return cs
