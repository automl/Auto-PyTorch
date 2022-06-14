from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

from ConfigSpace import ConfigurationSpace

import numpy as np

import torch
from torch import nn

from autoPyTorch.datasets.base_dataset import BaseDatasetPropertiesType
from autoPyTorch.pipeline.components.base_component import BaseEstimator
from autoPyTorch.pipeline.components.setup.network_backbone.forecasting_backbone.forecasting_decoder.components import \
    DecoderBlockInfo
from autoPyTorch.pipeline.components.setup.network_head.base_network_head import NetworkHeadComponent
from autoPyTorch.pipeline.components.setup.network_head.forecasting_network_head.NBEATS_head import build_NBEATS_network
from autoPyTorch.pipeline.components.setup.network_head.forecasting_network_head.distribution import (
    ALL_DISTRIBUTIONS,
    DisForecastingStrategy
)
from autoPyTorch.utils.common import FitRequirement


class QuantileHead(nn.Module):
    def __init__(self, head_components: List[nn.Module]):
        super().__init__()
        self.net = nn.ModuleList(head_components)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        return [net(x) for net in self.net]


class ForecastingHead(NetworkHeadComponent):
    """
    Base class for network heads used for forecasting.
     Holds the head module and the config which was used to create it.
    """
    _required_properties = ["name", "shortname", "handles_tabular", "handles_image", "handles_time_series"]

    def __init__(self,
                 random_state: Optional[np.random.RandomState] = None,
                 ):
        super(NetworkHeadComponent, self).__init__(random_state=random_state)

        self.add_fit_requirements(self._required_fit_requirements)
        self.head: Optional[nn.Module] = None
        self.output_shape: Optional[Tuple[int]] = None

    @property
    def _required_fit_requirements(self) -> List[FitRequirement]:
        return [
            FitRequirement('input_shape', (Iterable,), user_defined=True, dataset_property=True),
            FitRequirement('auto_regressive', (bool,), user_defined=False, dataset_property=False),
            FitRequirement('n_decoder_output_features', (int,), user_defined=False, dataset_property=False),
            FitRequirement('network_decoder', (Dict,), user_defined=False, dataset_property=False),
            FitRequirement('n_prediction_heads', (int,), user_defined=False, dataset_property=False),
            FitRequirement('output_shape', (Iterable,), user_defined=True, dataset_property=True),
            FitRequirement('net_output_type', (str,), user_defined=False, dataset_property=False),
            FitRequirement('n_prediction_steps', (int,), user_defined=False, dataset_property=True)

        ]

    def fit(self, X: Dict[str, Any], y: Any = None) -> BaseEstimator:
        """
        Builds the head component and assigns it to self.head

        Args:
            X (X: Dict[str, Any]):
                Dependencies needed by current component to perform fit
            y (Any):
                not used. To comply with sklearn API
        Returns:
            Self
        """
        self.check_requirements(X, y)

        output_shape = X['dataset_properties']['output_shape']

        net_output_type = X['net_output_type']

        if 'block_1' in X['network_decoder'] and X['network_decoder']['block_1'].decoder_properties.multi_blocks:
            # if the decoder is a stacked block, we directly build head inside the decoder
            if net_output_type != 'regression':
                raise ValueError("decoder with multi block structure only allow regression loss!")
            self.output_shape = (X['dataset_properties']['n_prediction_steps'], output_shape[-1])  # type: ignore
            return self

        num_quantiles = 0
        dist_cls = None
        if net_output_type == 'distribution':
            if 'dist_forecasting_strategy' not in X:
                raise ValueError('Distribution output type must contain dis_forecasting_strategy!')
            dist_forecasting_strategy = X['dist_forecasting_strategy']  # type: DisForecastingStrategy
            dist_cls = dist_forecasting_strategy.dist_cls
        elif net_output_type == 'quantile':
            if 'quantile_values' not in X:
                raise ValueError("For Quantile losses, quantiles must be given in X!")
            num_quantiles = len(X['quantile_values'])

        head_n_in_features: int = X["n_decoder_output_features"]
        n_prediction_heads = X["n_prediction_heads"]

        decoder_has_local_layer = X.get('mlp_has_local_layer', True)

        head_components = self.build_head(
            head_n_in_features=head_n_in_features,
            output_shape=output_shape,
            decoder_has_local_layer=decoder_has_local_layer,
            net_output_type=net_output_type,
            dist_cls=dist_cls,
            n_prediction_heads=n_prediction_heads,
            num_quantiles=num_quantiles,
        )
        self.head = head_components
        return self

    def transform(self, X: Dict[str, Any]) -> Dict[str, Any]:
        """
        Adds the network head into the fit dictionary 'X' and returns it.

        Args:
            X (Dict[str, Any]):
                'X' dictionary
        Returns:
            (Dict[str, Any]):
                the updated 'X' dictionary
        """
        if self.head is not None:
            X.update({'network_head': self.head})
        else:
            decoder = X['network_decoder']
            # NBEATS is a flat encoder, it only has one decoder
            first_decoder = decoder['block_1']
            assert self.output_shape is not None
            nbeats_decoder = build_NBEATS_network(first_decoder.decoder, self.output_shape)
            decoder['block_1'] = DecoderBlockInfo(decoder=nbeats_decoder,
                                                  decoder_properties=first_decoder.decoder_properties,
                                                  decoder_output_shape=first_decoder.decoder_output_shape,
                                                  decoder_input_shape=first_decoder.decoder_input_shape)
            X.update({'network_head': self.head,
                      'network_decoder': decoder})
        return X

    @staticmethod
    def get_properties(dataset_properties: Optional[Dict[str, BaseDatasetPropertiesType]] = None
                       ) -> Dict[str, Union[str, bool]]:
        """Get the properties of the underlying algorithm.

        Args:
            dataset_properties (Optional[Dict[str, Union[str, int]]):
                Describes the dataset to work on

        Returns:
            Dict[str, Any]:
                Properties of the algorithm
        """
        return {
            'shortname': 'ForecastingHead',
            'name': 'ForecastingHead',
            'handles_tabular': False,
            'handles_image': False,
            'handles_time_series': True,
        }

    def build_head(self,  # type: ignore[override]
                   head_n_in_features: int,
                   output_shape: Tuple[int, ...],
                   decoder_has_local_layer: bool = True,
                   net_output_type: str = "distribution",
                   dist_cls: Optional[str] = None,
                   n_prediction_heads: int = 1,
                   num_quantiles: int = 3,
                   ) -> nn.Module:
        """
        Builds the head module and returns it

        Args:
            head_n_in_features (int):
                shape of the input to the head (usually the shape of the backbone output)
            output_shape (Tuple[int, ...]):
                shape of the output of the head
            decoder_has_local_layer (bool):
                if the decoder has local layer
            net_output_type (str):
                network output type
            dist_cls (Optional[str]):
                output distribution, only works if required_net_out_put_type is 'distribution'
            n_prediction_heads (Dict):
                additional paramter for initializing architectures. How many heads to predict
            num_quantiles (int):
                number of quantile losses

        Returns:
            nn.Module:
                head module
        """
        if net_output_type == 'distribution':
            assert dist_cls is not None
            proj_layer_d = ALL_DISTRIBUTIONS[dist_cls](num_in_features=head_n_in_features,
                                                       output_shape=output_shape[1:],
                                                       n_prediction_heads=n_prediction_heads,
                                                       decoder_has_local_layer=decoder_has_local_layer
                                                       )
            return proj_layer_d
        elif net_output_type == 'regression':
            if decoder_has_local_layer:
                proj_layer_r = nn.Sequential(nn.Linear(head_n_in_features, np.product(output_shape[1:])))
            else:
                proj_layer_r = nn.Sequential(
                    nn.Linear(head_n_in_features, n_prediction_heads * np.product(output_shape[1:])),
                    nn.Unflatten(-1, (n_prediction_heads, *output_shape[1:])),
                )
            return proj_layer_r
        elif net_output_type == "quantile":
            if decoder_has_local_layer:
                proj_layer_quantiles = [
                    nn.Sequential(nn.Linear(head_n_in_features, np.product(output_shape[1:])))
                    for _ in range(num_quantiles)
                ]
            else:
                proj_layer_quantiles = [
                    nn.Sequential(
                        nn.Linear(head_n_in_features, n_prediction_heads * np.product(output_shape[1:])),
                        nn.Unflatten(-1, (n_prediction_heads, *output_shape[1:])),
                    ) for _ in range(num_quantiles)
                ]
            proj_layer_q = QuantileHead(proj_layer_quantiles)
            return proj_layer_q
        else:
            raise NotImplementedError(f"Unsupported network type "
                                      f"{net_output_type} (should be one of the following: "
                                      f"regression, distribution or quantiles)")

    @staticmethod
    def get_hyperparameter_search_space(
            dataset_properties: Optional[Dict[str, BaseDatasetPropertiesType]] = None,
    ) -> ConfigurationSpace:
        """Return the configuration space of network head.

        Returns:
            ConfigurationSpace:
                The configuration space of this algorithm.
        """
        cs = ConfigurationSpace()

        return cs
