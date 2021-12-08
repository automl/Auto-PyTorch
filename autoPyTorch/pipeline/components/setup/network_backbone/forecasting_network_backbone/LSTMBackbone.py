from typing import Any, Dict, Optional, Tuple

import ConfigSpace as CS
from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import (
    CategoricalHyperparameter,
    UniformFloatHyperparameter,
    UniformIntegerHyperparameter
)

import torch
from torch import nn

from autoPyTorch.pipeline.components.base_component import BaseEstimator
from autoPyTorch.pipeline.components.setup.network_backbone.forecasting_network_backbone.base_forecasting_backbone\
    import BaseForecastingNetworkBackbone
from autoPyTorch.utils.common import HyperparameterSearchSpace, add_hyperparameter, get_hyperparameter


class _LSTM(nn.Module):
    def __init__(self,
                 in_features: int,
                 config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self.only_return_final_stage = True
        self.lstm = nn.LSTM(input_size=in_features,
                            hidden_size=config["hidden_size"],
                            num_layers=config["num_layers"],
                            dropout=config.get("dropout", 0.0),
                            bidirectional=config["bidirectional"],
                            batch_first=True)

    def forward(self, x: torch.Tensor,
                hx: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> Tuple[torch.Tensor, ...]:
        B, T, _ = x.shape

        outputs, hidden_state, = self.lstm(x, hx)

        if self.only_return_final_stage:
            if not self.config["bidirectional"]:
                return outputs[:, -1, :],
            else:
                # concatenate last forward hidden state with first backward hidden state
                outputs_by_direction = outputs.view(B,
                                                    T,
                                                    2,
                                                    self.config["hidden_size"])
                out = torch.cat([
                    outputs_by_direction[:, -1, 0, :],
                    outputs_by_direction[:, 0, 1, :]
                ], dim=1)
                return out,
        else:
            return outputs, hidden_state


class LSTMBackbone(BaseForecastingNetworkBackbone):
    """
    Standard searchable LSTM backbone for time series data
    """
    _fixed_seq_length = False

    def __init__(self, **kwargs: Dict):
        super().__init__(**kwargs)

    def build_backbone(self, input_shape: Tuple[int, ...]) -> nn.Module:
        backbone = _LSTM(in_features=input_shape[-1],
                         config=self.config)
        self.backbone = backbone
        return backbone

    @property
    def backbone_properties(self):
        backbone_properties = {'network_output_tuple': True,
                               'accept_additional_input': True,
                               'hidden_states': True}
        return backbone_properties

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
    def get_properties(dataset_properties: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        return {
            'shortname': 'LSTMBackbone',
            'name': 'LSTMBackbone',
            'handles_tabular': False,
            'handles_image': False,
            'handles_time_series': True,
        }

    @staticmethod
    def get_hyperparameter_search_space(
            dataset_properties: Optional[Dict] = None,
            num_layers: HyperparameterSearchSpace = HyperparameterSearchSpace(hyperparameter='num_layers',
                                                                              value_range=(1, 3),
                                                                              default_value=1),
            hidden_size: HyperparameterSearchSpace = HyperparameterSearchSpace(hyperparameter='hidden_size',
                                                                               value_range=(64, 512),
                                                                               default_value=256),
            use_dropout: HyperparameterSearchSpace = HyperparameterSearchSpace(hyperparameter='use_dropout',
                                                                               value_range=(True, False),
                                                                               default_value=False),
            dropout: HyperparameterSearchSpace = HyperparameterSearchSpace(hyperparameter='dropout',
                                                                           value_range=(0., 0.5),
                                                                           default_value=0.2),
            bidirectional: HyperparameterSearchSpace = HyperparameterSearchSpace(hyperparameter='bidirectional',
                                                                                 value_range=(True, False),
                                                                                 default_value=True)
    ) -> ConfigurationSpace:
        cs = CS.ConfigurationSpace()

        # TODO consider lstm layers with different hidden size
        num_layers = get_hyperparameter(num_layers, UniformIntegerHyperparameter)
        use_dropout = get_hyperparameter(use_dropout, CategoricalHyperparameter)
        dropout = get_hyperparameter(dropout, UniformFloatHyperparameter)
        cs.add_hyperparameters([num_layers, use_dropout, dropout])

        # Add plain hyperparameters
        add_hyperparameter(cs, hidden_size, UniformIntegerHyperparameter)
        add_hyperparameter(cs, bidirectional, CategoricalHyperparameter)

        cs.add_condition(CS.AndConjunction(CS.EqualsCondition(dropout, use_dropout, True),
                                           CS.GreaterThanCondition(dropout, num_layers, 1)))

        return cs
