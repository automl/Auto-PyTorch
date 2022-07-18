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
from autoPyTorch.pipeline.components.setup.network_backbone.forecasting_backbone.components_util import (
    PositionalEncoding,
    build_transformer_layers
)
from autoPyTorch.pipeline.components.setup.network_backbone.forecasting_backbone.forecasting_encoder.\
    base_forecasting_encoder import BaseForecastingEncoder, EncoderProperties
from autoPyTorch.pipeline.components.setup.network_backbone.forecasting_backbone.forecasting_encoder.components import \
    EncoderNetwork
from autoPyTorch.utils.common import (
    HyperparameterSearchSpace,
    add_hyperparameter,
    get_hyperparameter
)


class _TransformerEncoder(EncoderNetwork):
    def __init__(self,
                 in_features: int,
                 d_model: int,
                 num_layers: int,
                 transformer_encoder_layers: nn.Module,
                 use_positional_encoder: bool,
                 use_layer_norm_output: bool,
                 dropout_pe: float = 0.0,
                 layer_norm_eps_output: Optional[float] = None,
                 lagged_value: Optional[List[int]] = None):
        super().__init__()
        if lagged_value is None:
            self.lagged_value = [0]
        else:
            self.lagged_value = lagged_value
        if in_features != d_model:
            input_layer = [nn.Linear(in_features, d_model, bias=False)]
        else:
            input_layer = []
        if use_positional_encoder:
            input_layer.append(PositionalEncoding(d_model, dropout_pe))
        self.input_layer = nn.Sequential(*input_layer)

        self.use_layer_norm_output = use_layer_norm_output
        if use_layer_norm_output:
            norm = nn.LayerNorm(d_model, eps=layer_norm_eps_output)
        else:
            norm = None
        self.transformer_encoder_layers = nn.TransformerEncoder(encoder_layer=transformer_encoder_layers,
                                                                num_layers=num_layers,
                                                                norm=norm)

    def forward(self,
                x: torch.Tensor,
                output_seq: bool = False,
                mask: Optional[torch.Tensor] = None,
                src_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.input_layer(x)
        x = self.transformer_encoder_layers(x)
        if output_seq:
            return x
        else:
            return self.get_last_seq_value(x)

    def get_last_seq_value(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, -1:]


class TransformerEncoder(BaseForecastingEncoder):
    """
    Standard searchable Transformer Encoder for time series data
    """
    _fixed_seq_length = False

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self.lagged_value = [1, 2, 3, 4, 5, 6, 7]

    def build_encoder(self, input_shape: Tuple[int, ...]) -> nn.Module:
        in_features = input_shape[-1]

        d_model = 2 ** self.config['d_model_log']
        transformer_encoder_layers = build_transformer_layers(d_model=d_model, config=self.config, layer_type='encoder')

        encoder = _TransformerEncoder(in_features=in_features,
                                      d_model=d_model,
                                      num_layers=self.config['num_layers'],
                                      transformer_encoder_layers=transformer_encoder_layers,
                                      use_positional_encoder=self.config['use_positional_encoder'],
                                      use_layer_norm_output=self.config['use_layer_norm_output'],
                                      dropout_pe=self.config.get('dropout_positional_encoder', 0.0),
                                      layer_norm_eps_output=self.config.get('layer_norm_eps_output', None),
                                      lagged_value=self.lagged_value)
        return encoder

    def n_encoder_output_feature(self) -> int:
        d_model: int = 2 ** self.config['d_model_log']
        return d_model

    @staticmethod
    def allowed_decoders() -> List[str]:
        """
        decoder that is compatible with the encoder
        """
        return ['MLPDecoder', 'TransformerDecoder']

    @staticmethod
    def encoder_properties() -> EncoderProperties:
        return EncoderProperties(lagged_input=True,
                                 is_casual=False)

    def fit(self, X: Dict[str, Any], y: Any = None) -> BaseEstimator:
        if 'lagged_value' in X['dataset_properties']:
            self.lagged_value = X['dataset_properties']['lagged_value']
        return super().fit(X, y)

    def transform(self, X: Dict[str, Any]) -> Dict[str, Any]:
        transformer_encoder_kwargs = {'d_model_log': self.config['d_model_log']}  # used for initialize
        X.update({'transformer_encoder_kwargs': transformer_encoder_kwargs})
        return super().transform(X)

    @staticmethod
    def get_properties(
            dataset_properties: Optional[Dict[str, BaseDatasetPropertiesType]] = None) -> Dict[str, Union[str, bool]]:
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
            norm_first: HyperparameterSearchSpace =
            HyperparameterSearchSpace(hyperparameter="norm_first",
                                      value_range=(True, False),
                                      default_value=True),
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
                                      default_value=0.1,
                                      ),
            decoder_type: HyperparameterSearchSpace =
            HyperparameterSearchSpace(hyperparameter='decoder_type',
                                      value_range=('MLPDecoder', 'TransformerDecoder'),
                                      default_value='MLPDecoder')
    ) -> ConfigurationSpace:
        """
        get hyperparameter search space for Transformer, Given that d_model must be a multiple of n_head_log, we
        consider their log value (with base 2) as the hyperparameters

        Args:
            num_layers (int):
                number of transformer layers
            n_head_log (int):
                log value (base 2, this should work for all the following hyperparameters with logs) of number of head
            d_model_log (int):
                log values of input of dimensions passed to feed forward network
            d_feed_forward_log (int):
                log values of feed forward network width
            norm_first (bool):
                if ``True``, layer norm is done prior to attention and feedforward operations, respectivaly.
                Otherwise, it's done after. Default: ``False`` (after).
            layer_norm_eps (float):
                eps for layer norm
            use_positional_encoder (bool):
                if positional encoder is applied
            use_layer_norm_output (bool):
                if layer norm output is applied
            activation (str):
                activation function type
            use_dropout (bool):
                if dropout is applied
            dropout (float):
                dropout rate
            decoder_type (str):
                type of decoder, could be MLPDecoder (DeepAR) or TransformerDecoder (seq2seq)

        Returns:
            ConfigurationSpace:
                configuration space
        """
        cs = CS.ConfigurationSpace()

        add_hyperparameter(cs, activation, CategoricalHyperparameter)
        add_hyperparameter(cs, d_model_log, UniformIntegerHyperparameter)
        add_hyperparameter(cs, norm_first, CategoricalHyperparameter)

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

        add_hyperparameter(cs, n_head_log, UniformIntegerHyperparameter)
        add_hyperparameter(cs, d_feed_forward_log, UniformIntegerHyperparameter)
        add_hyperparameter(cs, layer_norm_eps, UniformFloatHyperparameter)

        dropout = get_hyperparameter(dropout, UniformFloatHyperparameter)
        cs.add_hyperparameter(dropout)
        cs.add_condition(CS.EqualsCondition(dropout, use_dropout, True))

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
