from typing import Any, Dict, List, Optional, Tuple, Union

from ConfigSpace import ConfigurationSpace
from ConfigSpace.hyperparameters import CategoricalHyperparameter

import torch
from torch import nn

from autoPyTorch.datasets.base_dataset import BaseDatasetPropertiesType
from autoPyTorch.pipeline.components.base_component import BaseEstimator
from autoPyTorch.pipeline.components.setup.network_backbone.MLPBackbone import MLPBackbone
from autoPyTorch.pipeline.components.setup.network_backbone.forecasting_backbone.forecasting_encoder.\
    base_forecasting_encoder import BaseForecastingEncoder, EncoderProperties
from autoPyTorch.pipeline.components.setup.network_backbone.forecasting_backbone.forecasting_encoder.components import \
    EncoderNetwork
from autoPyTorch.pipeline.components.setup.network_backbone.utils import _activations
from autoPyTorch.utils.common import (
    FitRequirement,
    HyperparameterSearchSpace,
    add_hyperparameter
)


class TimeSeriesMLP(EncoderNetwork):
    def __init__(self,
                 window_size: int,
                 network: Optional[nn.Module] = None
                 ):
        """
        Transform the input features (B, T, N) to fit the requirement of MLP
        Args:
            window_size (int): T
            fill_lower_resolution_seq: if sequence with lower resolution needs to be filled with 0
        (for multi-fidelity problems with resolution as fidelity)
        """
        super().__init__()
        self.window_size = window_size
        self.network = network

    def forward(self, x: torch.Tensor, output_seq: bool = False) -> torch.Tensor:
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
            x = x.unfold(1, self.window_size, 1).transpose(-1, -2)
            # x.shape = [B, L_in - self.window + 1, self.window, N]
        else:
            if x.shape[1] > self.window_size:
                # we need to ensure that the input size fits the network shape
                x = x[:, -self.window_size:]  # x.shape = (B, self.window, N)
        x = x.flatten(-2)
        return x if self.network is None else self.network(x)

    def get_last_seq_value(self, x: torch.Tensor) -> torch.Tensor:
        return x


class MLPEncoder(BaseForecastingEncoder, MLPBackbone):  # type: ignore[misc]
    _fixed_seq_length = True
    window_size = 1

    @staticmethod
    def encoder_properties() -> EncoderProperties:
        return EncoderProperties(bijective_seq_output=False, fixed_input_seq_length=True)

    @staticmethod
    def allowed_decoders() -> List[str]:
        """
        decoder that is compatible with the encoder
        """
        return ['MLPDecoder']

    @property
    def _required_fit_arguments(self) -> List[FitRequirement]:
        requirements_list = super()._required_fit_arguments
        requirements_list.append(FitRequirement('window_size', (int,), user_defined=False, dataset_property=False))
        return requirements_list

    def fit(self, X: Dict[str, Any], y: Any = None) -> BaseEstimator:
        self.window_size = X["window_size"]
        # when resolution is smaller
        return super().fit(X, y)

    def build_encoder(self, input_shape: Tuple[int, ...]) -> nn.Module:
        in_features = input_shape[-1]
        network = nn.Sequential(*self._build_backbone(in_features * self.window_size))
        return TimeSeriesMLP(window_size=self.window_size,
                             network=network)

    def n_encoder_output_feature(self) -> int:
        # This function should never be called!!
        num_out_features: int = self.config["num_units_%d" % (self.config['num_groups'])]
        return num_out_features

    def _add_layer(self, layers: List[nn.Module], in_features: int, out_features: int,
                   layer_id: int) -> None:
        """
        Dynamically add a layer given the in->out specification

        Args:
            layers (List[nn.Module]): The list where all modules are added
            in_features (int): input dimensionality of the new layer
            out_features (int): output dimensionality of the new layer

        """
        layers.append(nn.Linear(in_features, out_features))
        if self.config['normalization'] == 'BN':
            layers.append(nn.BatchNorm1d(out_features))
        elif self.config['normalization'] == 'LN':
            layers.append(nn.LayerNorm(out_features))
        layers.append(_activations[self.config["activation"]]())
        if self.config['use_dropout']:
            layers.append(nn.Dropout(self.config["dropout_%d" % layer_id]))

    @staticmethod
    def get_properties(dataset_properties: Optional[Dict[str, BaseDatasetPropertiesType]] = None
                       ) -> Dict[str, Union[str, bool]]:
        return {
            'shortname': 'TSMLPEncoder',
            'name': 'TimeSeriesMLPEncoder',
            'handles_tabular': False,
            'handles_image': False,
            'handles_time_series': True,
        }

    @staticmethod
    def get_hyperparameter_search_space(  # type: ignore[override]
            dataset_properties: Optional[Dict[str, BaseDatasetPropertiesType]] = None,
            num_groups: HyperparameterSearchSpace = HyperparameterSearchSpace(hyperparameter="num_groups",
                                                                              value_range=(1, 5),
                                                                              default_value=3,
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
                                                                             default_value=64,
                                                                             log=True
                                                                             ),
            normalization: HyperparameterSearchSpace = HyperparameterSearchSpace(hyperparameter='normalization',
                                                                                 value_range=('BN', 'LN', 'NoNorm'),
                                                                                 default_value='BN'),
            dropout: HyperparameterSearchSpace = HyperparameterSearchSpace(hyperparameter="dropout",
                                                                           value_range=(0, 0.8),
                                                                           default_value=0.1,
                                                                           ),
    ) -> ConfigurationSpace:
        cs = MLPBackbone.get_hyperparameter_search_space(dataset_properties=dataset_properties,  # type: ignore
                                                         num_groups=num_groups,
                                                         activation=activation,
                                                         use_dropout=use_dropout,
                                                         num_units=num_units,
                                                         dropout=dropout)
        add_hyperparameter(cs, normalization, CategoricalHyperparameter)
        return cs
