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

from autoPyTorch.pipeline.components.setup.network_backbone.base_network_backbone import NetworkBackboneComponent


class _LSTM(nn.Module):
    def __init__(self,
                 in_features: int,
                 config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self.lstm = nn.LSTM(input_size=in_features,
                            hidden_size=config["hidden_size"],
                            num_layers=config["num_layers"],
                            dropout=config.get("dropout", 0.0),
                            bidirectional=config["bidirectional"],
                            batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape

        hidden_states, (_, _) = self.lstm(x)
        if not self.config["bidirectional"]:
            return hidden_states[:, -1, :]
        else:
            # concatenate last forward hidden state with first backward hidden state
            hidden_states_by_direction = hidden_states.view(B,
                                                            T,
                                                            2,
                                                            self.config["hidden_size"])
            out = torch.cat([
                hidden_states_by_direction[:, -1, 0, :],
                hidden_states_by_direction[:, 0, 1, :]
            ], dim=1)
            return out


class LSTMBackbone(NetworkBackboneComponent):
    """
    Standard searchable LSTM backbone for time series data
    """

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)

    def build_backbone(self, input_shape: Tuple[int, ...]) -> nn.Module:
        backbone = _LSTM(in_features=input_shape[-1],
                         config=self.config)
        self.backbone = backbone
        return backbone

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
    def get_hyperparameter_search_space(dataset_properties: Optional[Dict] = None,
                                        num_layers: Tuple[Tuple, int] = ((1, 3), 1),
                                        hidden_size: Tuple[Tuple, int] = ((64, 512), 256),
                                        use_dropout: Tuple[Tuple, bool] = ((True, False), False),
                                        dropout: Tuple[Tuple, float] = ((0, 0.5), 0.2),
                                        bidirectional: Tuple[Tuple, bool] = ((True, False), True)
                                        ) -> ConfigurationSpace:
        cs = CS.ConfigurationSpace()

        min_num_layers, max_num_layers = num_layers[0]
        num_layers = UniformIntegerHyperparameter('num_layers',
                                                  lower=min_num_layers,
                                                  upper=max_num_layers,
                                                  default_value=num_layers[1])
        cs.add_hyperparameter(num_layers)

        min_hidden_size, max_hidden_size = hidden_size[0]
        hidden_size = UniformIntegerHyperparameter('hidden_size',
                                                   lower=min_hidden_size,
                                                   upper=max_hidden_size,
                                                   default_value=hidden_size[1])
        cs.add_hyperparameter(hidden_size)

        use_dropout = CategoricalHyperparameter('use_dropout',
                                                choices=use_dropout[0],
                                                default_value=use_dropout[1])

        min_dropout, max_dropout = dropout[0]
        dropout = UniformFloatHyperparameter('dropout',
                                             lower=min_dropout,
                                             upper=max_dropout,
                                             default_value=dropout[1])

        cs.add_hyperparameters([use_dropout, dropout])
        cs.add_condition(CS.AndConjunction(CS.EqualsCondition(dropout, use_dropout, True),
                                           CS.GreaterThanCondition(dropout, num_layers, 1)))

        bidirectional = CategoricalHyperparameter('bidirectional',
                                                  choices=bidirectional[0],
                                                  default_value=bidirectional[1])
        cs.add_hyperparameter(bidirectional)

        return cs
