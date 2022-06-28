from typing import Any, Dict, List, Optional, Tuple, Union

import ConfigSpace as CS
from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import (
    CategoricalHyperparameter,
    UniformFloatHyperparameter,
    UniformIntegerHyperparameter
)

import numpy as np

import torch
from torch import nn

from autoPyTorch.datasets.base_dataset import BaseDatasetPropertiesType
from autoPyTorch.pipeline.components.base_component import BaseEstimator
from autoPyTorch.pipeline.components.setup.network_backbone.forecasting_backbone.components_util import (
    PositionalEncoding, build_transformer_layers)
from autoPyTorch.pipeline.components.setup.network_backbone.forecasting_backbone.forecasting_decoder. \
    base_forecasting_decoder import BaseForecastingDecoder, DecoderProperties
from autoPyTorch.pipeline.components.setup.network_backbone.forecasting_backbone.forecasting_decoder.components import \
    DecoderNetwork
from autoPyTorch.utils.common import (
    FitRequirement,
    HyperparameterSearchSpace,
    add_hyperparameter,
    get_hyperparameter
)


class _TransformerDecoder(DecoderNetwork):
    def __init__(self,
                 in_features: int,
                 d_model: int,
                 num_layers: int,
                 transformer_decoder_layers: nn.Module,
                 use_positional_decoder: bool,
                 use_layer_norm_output: bool,
                 dropout_pd: float = 0.0,
                 layer_norm_eps_output: Optional[float] = None,
                 n_prediction_steps: int = 1,
                 lagged_value: Optional[Union[List, np.ndarray]] = None):
        super().__init__()
        self.lagged_value = lagged_value
        in_features = in_features

        # self.input_layer = [nn.Linear(in_features, d_model, bias=False)]
        self.input_layer = nn.Linear(in_features, d_model, bias=False)

        self.use_positional_decoder = use_positional_decoder
        if use_positional_decoder:
            self.pos_encoding = PositionalEncoding(d_model, dropout_pd)

        self.use_layer_norm_output = use_layer_norm_output

        if use_layer_norm_output:
            norm = nn.LayerNorm(d_model, eps=layer_norm_eps_output)
        else:
            norm = None
        self.transformer_decoder_layers = nn.TransformerDecoder(decoder_layer=transformer_decoder_layers,
                                                                num_layers=num_layers,
                                                                norm=norm)
        self.tgt_mask = nn.Transformer.generate_square_subsequent_mask(n_prediction_steps)

    def forward(self,
                x_future: torch.Tensor,
                encoder_output: torch.Tensor,
                pos_idx: Optional[Tuple[int]] = None) -> torch.Tensor:
        output = self.input_layer(x_future)
        if self.use_positional_decoder:
            output = self.pos_encoding(output, pos_idx)
        if self.training:
            output = self.transformer_decoder_layers(output, encoder_output,
                                                     tgt_mask=self.tgt_mask.to(encoder_output.device))
        else:
            output = self.transformer_decoder_layers(output, encoder_output)
        return output


class ForecastingTransformerDecoder(BaseForecastingDecoder):
    """
    Standard searchable Transformer decoder for time series data, only works when the encoder is a
    Transformer Encoder
    """
    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        # RNN is naturally auto-regressive. However, we will not consider it as a decoder for deep AR model
        self.transformer_encoder_kwargs: Optional[dict] = None
        self.lagged_value = [1, 2, 3, 4, 5, 6, 7]
        self.add_fit_requirements([FitRequirement('transformer_encoder_kwargs', (Dict,), user_defined=False,
                                                  dataset_property=False)])

    def _build_decoder(self,
                       encoder_output_shape: Tuple[int, ...],
                       future_variable_input: Tuple[int, ...],
                       n_prediction_heads: int,
                       dataset_properties: Dict) -> Tuple[nn.Module, int]:
        assert self.transformer_encoder_kwargs is not None
        d_model = 2 ** self.transformer_encoder_kwargs['d_model_log']
        transformer_decoder_layers = build_transformer_layers(d_model=d_model, config=self.config, layer_type='decoder')
        n_prediction_steps = dataset_properties['n_prediction_steps']

        decoder = _TransformerDecoder(in_features=future_variable_input[-1],
                                      d_model=d_model,
                                      num_layers=self.config['num_layers'],
                                      transformer_decoder_layers=transformer_decoder_layers,
                                      use_positional_decoder=self.config['use_positional_decoder'],
                                      use_layer_norm_output=self.config['use_layer_norm_output'],
                                      dropout_pd=self.config.get('dropout_positional_decoder', 0.0),
                                      layer_norm_eps_output=self.config.get('layer_norm_eps_output', None),
                                      n_prediction_steps=n_prediction_steps,
                                      lagged_value=self.lagged_value)

        return decoder, d_model

    @staticmethod
    def decoder_properties() -> DecoderProperties:
        return DecoderProperties(recurrent=True,
                                 lagged_input=True)

    def fit(self, X: Dict[str, Any], y: Any = None) -> BaseEstimator:
        self.transformer_encoder_kwargs = X['transformer_encoder_kwargs']
        if 'lagged_value' in X['dataset_properties']:
            self.lagged_value = X['dataset_properties']['lagged_value']
        return super().fit(X, y)

    @staticmethod
    def get_properties(dataset_properties: Optional[Dict[str, BaseDatasetPropertiesType]] = None
                       ) -> Dict[str, Union[str, bool]]:
        return {
            'shortname': 'TransformerDecoder',
            'name': 'TransformerDecoder',
            'handles_tabular': False,
            'handles_image': False,
            'handles_time_series': True,
        }

    @property
    def fitted_encoder(self) -> List[str]:
        return ['TransformerEncoder']

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
            use_positional_decoder: HyperparameterSearchSpace =
            HyperparameterSearchSpace(hyperparameter='use_positional_decoder',
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
                                      value_range=(0, 0.1),
                                      default_value=0.1,
                                      ),
    ) -> ConfigurationSpace:
        """
        get hyperparameter search space for Transformer, Given that d_model must be a multiple of n_head_log, we
        consider their log value (with base 2) as the hyperparameters

        Args:
            num_layers (int):
                number of transformer layers
            n_head_log (int):
                log value (base 2, this should work for all the following hyperparameters with logs) of number of head
            d_feed_forward_log (int):
                log values of feed forward network width
            norm_first (bool):
                if ``True``, layer norm is done prior to attention and feedforward operations, respectivaly.
                Otherwise, it's done after. Default: ``False`` (after).
            layer_norm_eps (float):
                eps for layer norm
            use_layer_norm_output (bool):
                if layer norm output is applied
            activation (str):
                activation function type
            use_dropout (bool):
                if dropout is applied
            dropout (float):
                dropout rate

        Returns:
            ConfigurationSpace:
                configuration space
        """
        cs = CS.ConfigurationSpace()

        add_hyperparameter(cs, activation, CategoricalHyperparameter)
        add_hyperparameter(cs, norm_first, CategoricalHyperparameter)

        min_transformer_layers, max_transformer_layers = num_layers.value_range

        num_layers = get_hyperparameter(num_layers, UniformIntegerHyperparameter)
        use_dropout = get_hyperparameter(use_dropout, CategoricalHyperparameter)

        # We can have dropout in the network for
        # better generalization
        use_positional_decoder = get_hyperparameter(use_positional_decoder, CategoricalHyperparameter)

        dropout_pd = HyperparameterSearchSpace(hyperparameter='dropout_positional_decoder',
                                               value_range=dropout.value_range,
                                               default_value=dropout.default_value,
                                               log=dropout.log)
        dropout_pd = get_hyperparameter(dropout_pd, UniformFloatHyperparameter)

        cs.add_hyperparameters([num_layers, use_dropout, use_positional_decoder, dropout_pd])
        cs.add_condition(CS.AndConjunction(
            CS.EqualsCondition(dropout_pd, use_dropout, True),
            CS.EqualsCondition(dropout_pd, use_positional_decoder, True)
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

        return cs
