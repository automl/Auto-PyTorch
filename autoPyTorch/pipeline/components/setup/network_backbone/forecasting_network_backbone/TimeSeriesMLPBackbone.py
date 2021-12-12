from typing import Any, Dict, List, Optional, Union, Tuple

import torch
from torch import nn

from ConfigSpace import ConfigurationSpace


from autoPyTorch.pipeline.components.setup.network_backbone.MLPBackbone import MLPBackbone
from autoPyTorch.pipeline.components.setup.network_backbone.forecasting_network_backbone.base_forecasting_backbone \
    import BaseForecastingNetworkBackbone, EncoderNetwork
from autoPyTorch.pipeline.components.base_component import BaseEstimator
from autoPyTorch.datasets.base_dataset import BaseDatasetPropertiesType
from autoPyTorch.pipeline.components.setup.network_backbone.utils import _activations
from autoPyTorch.utils.common import FitRequirement, HyperparameterSearchSpace


class _TimeSeriesMLP(EncoderNetwork):
    def __init__(self,
                 window_size: int,
                 module_layers: nn.Module,
                 ):
        super().__init__()
        self.window_size = window_size
        self.module_layers = module_layers

    def forward(self, x: torch.Tensor, output_seq: bool = False):
        """

        Args:
            x: torch.Tensor(B, L_in, N)
            output_seq (bool), if the MLP outputs a squence, in which case, the input will be rolled to fit the size of
            the network. For Instance if self.window_size = 3, and we obtain a squence with [1, 2, 3, 4, 5]
            the input of this mlp is rolled as :
            [[1, 2, 3]
            [2, 3, 4]
            [3, 4 ,5]]

        Returns:

        """
        if output_seq:
            x = x.unfold((1, self.window_size, 1)).transpose(-1, -2)
            # x.shape = [B, L_in - self.window + 1, self.window, N]
        else:
            if x.shape[1] > self.window_size:
                # we need to ensure that the input size fits the network shape
                x = x[:, -self.window_size:]  # x.shape = (B, self.window, N)
        x = x.flatten(-2)
        return self.module_layers(x)


class TimeSeriesMLPBackbone(BaseForecastingNetworkBackbone, MLPBackbone):
    _fixed_seq_length = True
    window_size = 1

    @property
    def encoder_properties(self):
        encoder_properties = {
            'has_hidden_states': False,
            'bijective_seq_output': False,
            'fixed_input_seq_length': True,
        }
        return encoder_properties

    @property
    def _required_fit_arguments(self) -> List[FitRequirement]:
        requirements_list = super()._required_fit_arguments
        requirements_list.append(FitRequirement('window_size', (int,), user_defined=False, dataset_property=False))
        return requirements_list

    def fit(self, X: Dict[str, Any], y: Any = None) -> BaseEstimator:
        self.window_size = X["window_size"]
        return super().fit(X, y)

    def build_backbone(self, input_shape: Tuple[int, ...]) -> nn.Module:
        in_features = input_shape[-1] * self.window_size
        return _TimeSeriesMLP(self.window_size, self._build_backbone(in_features))

    @staticmethod
    def get_properties(dataset_properties: Optional[Dict[str, BaseDatasetPropertiesType]] = None
                       ) -> Dict[str, Union[str, bool]]:
        return {
            'shortname': 'TSMLPBackbone',
            'name': 'TimeSeriesMLPBackbone',
            'handles_tabular': False,
            'handles_image': False,
            'handles_time_series': True,
        }

    @staticmethod
    def get_hyperparameter_search_space(
            dataset_properties: Optional[Dict[str, BaseDatasetPropertiesType]] = None,
            num_groups: HyperparameterSearchSpace = HyperparameterSearchSpace(hyperparameter="num_groups",
                                                                              value_range=(1, 15),
                                                                              default_value=5,
                                                                              ),
            activation: HyperparameterSearchSpace = HyperparameterSearchSpace(hyperparameter="activation",
                                                                              value_range=tuple(_activations.keys()),
                                                                              default_value=list(_activations.keys())[
                                                                                  0],
                                                                              ),
            use_dropout: HyperparameterSearchSpace = HyperparameterSearchSpace(hyperparameter="use_dropout",
                                                                               value_range=(True, False),
                                                                               default_value=False,
                                                                               ),
            num_units: HyperparameterSearchSpace = HyperparameterSearchSpace(hyperparameter="num_units",
                                                                             value_range=(16, 1024),
                                                                             default_value=256,
                                                                             log=True
                                                                             ),
            dropout: HyperparameterSearchSpace = HyperparameterSearchSpace(hyperparameter="dropout",
                                                                           value_range=(0, 0.8),
                                                                           default_value=0.5,
                                                                           ),
    ) -> ConfigurationSpace:
        return MLPBackbone.get_hyperparameter_search_space(dataset_properties=dataset_properties,
                                                           num_groups=num_groups,
                                                           activation=activation,
                                                           use_dropout=use_dropout,
                                                           num_units=num_units,
                                                           dropout=dropout)
