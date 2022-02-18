from typing import Any, Dict, Optional, Tuple, List, Union

import torch
from torch import nn
import numpy as np

import ConfigSpace as CS
from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import (
    CategoricalHyperparameter,
    UniformIntegerHyperparameter,
    UniformFloatHyperparameter
)

from autoPyTorch.datasets.base_dataset import BaseDatasetPropertiesType
from autoPyTorch.pipeline.components.base_component import BaseEstimator
from autoPyTorch.utils.common import add_hyperparameter

from autoPyTorch.pipeline.components.setup.network_backbone.\
    forecasting_backbone.forecasting_decoder.base_forecasting_decoder import BaseForecastingDecoder, RecurrentDecoderNetwork

from autoPyTorch.pipeline.components.setup.network_backbone.forecasting_backbone.transformer_util import \
    PositionalEncoding, build_transformer_layers

from autoPyTorch.utils.common import HyperparameterSearchSpace, get_hyperparameter, FitRequirement


class _TransformerDecoder(RecurrentDecoderNetwork):
    def __init__(self,
                 in_features: int,
                 d_model: int,
                 num_layers: int,
                 transformer_decoder_layers: nn.Module,
                 use_positional_decoder: bool,
                 use_layer_norm_output: bool,
                 dropout_pd: float = 0.0,
                 layer_norm_eps_output: Optional[float] = None,
                 lagged_value: Optional[Union[List, np.ndarray]] = None):
        super().__init__()
        self.lagged_value = lagged_value
        in_features = in_features if self.lagged_value is None else len(self.lagged_value) * in_features

        self.input_layer = [nn.Linear(in_features, d_model, bias=False)]
        if use_positional_decoder:
            self.input_layer.append(PositionalEncoding(d_model, dropout_pd))
        self.input_layer = nn.Sequential(*self.input_layer)

        self.use_layer_norm_output = use_layer_norm_output

        if use_layer_norm_output:
            norm = nn.LayerNorm(d_model, eps=layer_norm_eps_output)
        else:
            norm = None
        self.transformer_decoder_layers = nn.TransformerDecoder(decoder_layer=transformer_decoder_layers,
                                                                num_layers=num_layers,
                                                                norm=norm)

    def forward(self, x_future: torch.Tensor, features_latent: torch.Tensor,
                tgt_mask: Optional[torch.Tensor] = None,
                memory_mask: Optional[torch.Tensor] = None,
                tgt_key_padding_mask: Optional[torch.Tensor] = None,
                memory_key_padding_mask: Optional[torch.Tensor] = None):
        output = self.input_layer(x_future)
        output = self.transformer_decoder_layers(output, features_latent, tgt_mask=tgt_mask,
                                                 memory_mask=memory_mask,
                                                 tgt_key_padding_mask=tgt_key_padding_mask,
                                                 memory_key_padding_mask=memory_key_padding_mask)
        return output


class ForecastingTransformerDecoder(BaseForecastingDecoder):
    def __init__(self, **kwargs: Dict):
        super().__init__(**kwargs)
        # RNN is naturally auto-regressive. However, we will not consider it as a decoder for deep AR model
        self.auto_regressive = True
        self.transformer_encoder_kwargs = None
        self.lagged_value = [0, 1, 2, 3, 4, 5, 6, 7]

    def _build_decoder(self,
                       input_shape: Tuple[int, ...],
                       n_prediction_heads: int,
                       dataset_properties: Dict) -> nn.Module:
        d_model = 2 ** self.transformer_encoder_kwargs['d_model_log']
        transformer_decoder_layers = build_transformer_layers(d_model=d_model, config=self.config, layer_type='decoder')

        decoder = _TransformerDecoder(in_features=dataset_properties['output_shape'][-1],
                                      d_model=d_model,
                                      num_layers=self.config['num_layers'],
                                      transformer_decoder_layers=transformer_decoder_layers,
                                      use_positional_decoder=self.config['use_positional_decoder'],
                                      use_layer_norm_output=self.config['use_layer_norm_output'],
                                      dropout_pd=self.config.get('dropout_positional_decoder', 0.0),
                                      layer_norm_eps_output=self.config.get('layer_norm_eps_output', None),
                                      lagged_value=self.lagged_value)

        return decoder, d_model

    @property
    def _required_fit_requirements(self) -> List[FitRequirement]:
        fit_requirement = super(ForecastingTransformerDecoder, self)._required_fit_requirements
        fit_requirement.append(FitRequirement('transformer_encoder_kwargs', (Dict,), user_defined=False,
                                              dataset_property=False))
        return fit_requirement

    def decoder_properties(self):
        decoder_properties = super().decoder_properties()
        decoder_properties.update({'recurrent': True,
                                   'lagged_input': True,
                                   'mask_on_future_target': True,
                                   })
        return decoder_properties

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
    def fitted_encoder(self):
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

        """
        cs = CS.ConfigurationSpace()

        add_hyperparameter(cs, activation, CategoricalHyperparameter)

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
