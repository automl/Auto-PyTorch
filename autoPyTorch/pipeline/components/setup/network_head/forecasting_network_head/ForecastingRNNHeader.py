from abc import ABC
from typing import Any, Dict, Optional, Tuple, Union

import ConfigSpace as CS
from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import (
    CategoricalHyperparameter,
    UniformFloatHyperparameter,
    Constant
)

import torch
from torch import nn

from autoPyTorch.datasets.base_dataset import BaseDatasetPropertiesType
from autoPyTorch.pipeline.components.base_component import BaseEstimator
from autoPyTorch.pipeline.components.setup.network_head.forecasting_network_head.forecasting_head import \
    ForecastingHead

from autoPyTorch.pipeline.components.setup.network_head.forecasting_network_head.distribution import ALL_DISTRIBUTIONS
from autoPyTorch.utils.common import HyperparameterSearchSpace, add_hyperparameter, get_hyperparameter


class _RNN_Decoder(nn.Module):
    def __init__(self,
                 in_features: int,
                 hidden_size: int,
                 num_layers: int,
                 cell_type: str,
                 config: Dict[str, Any]):
        super().__init__()
        self.config = config
        if cell_type == 'lstm':
            cell = nn.LSTM
        else:
            cell = nn.GRU
        self.lstm = cell(input_size=in_features,
                         hidden_size=hidden_size,
                         num_layers=num_layers,
                         dropout=config.get("dropout", 0.0),
                         bidirectional=False,
                         batch_first=True)

    def forward(self, x: torch.Tensor,
                hx: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> Tuple[torch.Tensor, ...]:
        outputs, hidden_state, = self.lstm(x, hx)
        return outputs, hidden_state


class ForecastingRNNHeader(ForecastingHead):
    """
    Standard searchable RNN decoder for time series data, only works when the encoder is
    """
    def __init__(self, **kwargs: Dict):
        super().__init__(**kwargs)
        # RNN is naturally auto-regressive. However, we will not consider it as a decoder for deep AR model
        self.auto_regressive = True

    def _build_head(self, input_shape: Tuple[int, ...], **arch_kwargs) -> nn.Module:
        head = _RNN_Decoder(in_features=input_shape[-1],
                            config=self.config,
                            **arch_kwargs)
        self.head = head
        return head

    @property
    def decoder_properties(self):
        decoder_properties = {'network_output_tuple': True,
                              'accept_additional_input': True,
                              'recurrent': True}
        return decoder_properties

    def fit(self, X: Dict[str, Any], y: Any = None) -> BaseEstimator:
        X['network_output_tuple'] = True
        return super().fit(X, y)

    @property
    def only_return_final_stage(self):
        return self.backbone.only_return_final_stage

    @only_return_final_stage.setter
    def only_return_final_stage(self, only_return_final_stage):
        self.backbone.only_return_final_stage = only_return_final_stage

    @staticmethod
    def get_properties(dataset_properties: Optional[Dict[str, BaseDatasetPropertiesType]] = None) -> Dict[str, Any]:
        return {
            'shortname': 'ForecastingRNNHead',
            'name': 'ForecastingRNNHead',
            'handles_tabular': False,
            'handles_image': False,
            'handles_time_series': True,
        }

    @staticmethod
    def get_hyperparameter_search_space(
            dataset_properties: Optional[Dict] = None,
            use_dropout: HyperparameterSearchSpace = HyperparameterSearchSpace(hyperparameter='use_dropout',
                                                                               value_range=(True, False),
                                                                               default_value=False),
            dropout: HyperparameterSearchSpace = HyperparameterSearchSpace(hyperparameter='dropout',
                                                                           value_range=(0., 0.5),
                                                                           default_value=0.2),
            dist_cls: HyperparameterSearchSpace = HyperparameterSearchSpace(hyperparameter="dist_cls",
                                                                            value_range=tuple(ALL_DISTRIBUTIONS.keys()),
                                                                            default_value=
                                                                            list(ALL_DISTRIBUTIONS.keys())[0]),
    ) -> ConfigurationSpace:
        cs = CS.ConfigurationSpace()

        use_dropout = get_hyperparameter(use_dropout, CategoricalHyperparameter)
        dropout = get_hyperparameter(dropout, UniformFloatHyperparameter)

        cs.add_hyperparameters([use_dropout, dropout])

        # Add plain hyperparameters
        # Hidden size is given by the encoder architecture
        cs.add_condition(CS.EqualsCondition(dropout, use_dropout, True))

        add_hyperparameter(cs, dist_cls, CategoricalHyperparameter)

        return cs

