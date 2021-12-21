from typing import Any, Dict, Iterable, Tuple, List, Optional

import numpy as np
import torch
from torch import nn
from ConfigSpace import ConfigurationSpace

from autoPyTorch.datasets.base_dataset import BaseDatasetPropertiesType
from autoPyTorch.pipeline.components.base_component import BaseEstimator
from autoPyTorch.pipeline.components.setup.network_head.base_network_head import NetworkHeadComponent
from autoPyTorch.utils.common import FitRequirement
from autoPyTorch.pipeline.components.setup.network_head.forecasting_network_head.distribution import \
    ALL_DISTRIBUTIONS
from autoPyTorch.pipeline.components.setup.network_head.forecasting_network_head.NBEATS_head import build_NBEATS_network


class ForecastingHead(NetworkHeadComponent):
    """
    Base class for network heads used for forecasting.
     Holds the head module and the config which was used to create it.
    """
    _required_properties = ["name", "shortname", "handles_tabular", "handles_image", "handles_time_series"]

    def __init__(self,
                 random_state: Optional[np.random.RandomState] = None):
        super(NetworkHeadComponent, self).__init__(random_state=random_state)

        self.add_fit_requirements(self._required_fit_requirements)
        self.head: Optional[nn.Module] = None
        self.required_net_out_put_type: Optional[str] = None
        self.output_shape = None

    @property
    def _required_fit_requirements(self) -> List[FitRequirement]:
        return [
            FitRequirement('input_shape', (Iterable,), user_defined=True, dataset_property=True),
            FitRequirement('task_type', (str,), user_defined=True, dataset_property=True),
            FitRequirement('auto_regressive', (bool,), user_defined=False, dataset_property=False),
            FitRequirement('decoder_properties', (Dict,), user_defined=False, dataset_property=False),
            FitRequirement('n_decoder_output_features', (int, ), user_defined=False, dataset_property=False),
            FitRequirement('n_prediction_heads', (int,), user_defined=False, dataset_property=False),
            FitRequirement('output_shape', (Iterable, int), user_defined=True, dataset_property=True),
        ]

    def fit(self, X: Dict[str, Any], y: Any = None) -> BaseEstimator:
        """
        Builds the head component and assigns it to self.head

        Args:
            X (X: Dict[str, Any]): Dependencies needed by current component to perform fit
            y (Any): not used. To comply with sklearn API
        Returns:
            Self
        """
        self.check_requirements(X, y)
        output_shape = X['dataset_properties']['output_shape']

        self.required_net_out_put_type = X['required_net_out_put_type']

        if X['decoder_properties']['multi_blocks']:
            # if the decoder is a stacked block, we directly build head inside the decoder
            if X.get('network_decoder', None) is None:
                raise ValueError("when decoder has multi_blocks, it must be specified!")
            if self.required_net_out_put_type != 'regression':
                raise ValueError("decoder with multi block structure only allow regression loss!")
            self.output_shape = output_shape
            return self

        if self.required_net_out_put_type == 'distribution':
            if 'dist_cls' not in X:
                raise ValueError('Distribution output type must contain dist_cls!!')

        dist_cls = X.get('dist_cls', None)

        auto_regressive = X.get('auto_regressive', False)

        head_input_shape = X["n_decoder_output_features"]
        n_prediction_heads = X["n_prediction_heads"]

        self.head = self.build_head(
            input_shape=head_input_shape,
            output_shape=output_shape,
            auto_regressive=auto_regressive,
            dist_cls=dist_cls,
            n_prediction_heads=n_prediction_heads,
        )
        return self

    def transform(self, X: Dict[str, Any]) -> Dict[str, Any]:
        """
        Adds the network head into the fit dictionary 'X' and returns it.

        Args:
            X (Dict[str, Any]): 'X' dictionary
        Returns:
            (Dict[str, Any]): the updated 'X' dictionary
        """
        if self.head is not None:
            X.update({'network_head': self.head})
        else:
            decoder = X['network_decoder']
            decoder = build_NBEATS_network(decoder, self.output_shape)
            X.update({'network_head': self.head,
                      'network_decoder': decoder})
        return X


    def build_head(self,
                   input_shape: Tuple[int, ...],
                   output_shape: Tuple[int, ...],
                   auto_regressive: bool = False,
                   dist_cls: Optional[str] = None,
                   n_prediction_heads: int = 1) -> nn.Module:
        """
        Builds the head module and returns it

        Args:
            input_shape (Tuple[int, ...]): shape of the input to the head (usually the shape of the backbone output)
            output_shape (Tuple[int, ...]): shape of the output of the head
            auto_regressive (bool): if the network is auto-regressive
            dist_cls (Optional[str]): output distribution, only works if required_net_out_put_type is 'distribution'
            n_prediction_heads (Dict): additional paramter for initializing architectures. How many heads to predict

        Returns:
            nn.Module: head module
        """
        head_layer = self.build_proj_layer(
            input_shape=input_shape,
            output_shape=output_shape,
            auto_regressive=auto_regressive,
            net_out_put_type=self.required_net_out_put_type,
            dist_cls=dist_cls,
            n_prediction_heads=n_prediction_heads
        )
        return head_layer

    @staticmethod
    def build_proj_layer(input_shape: Tuple[int, ...],
                         output_shape: Tuple[int, ...],
                         n_prediction_heads: int,
                         auto_regressive: bool,
                         net_out_put_type: str,
                         dist_cls: Optional[str] = None) -> torch.nn.Module:
        """
        a final layer that project the head output to the final distribution
        Args:
            input_shape (int): input shape to build the header,
            is used to initialize size of the linear layer
            output_shape (Tuple[int, ..]): deserved output shape
            n_prediction_heads: int, how many steps the head want to predict
            auto_regressive (bool): if the network is auto-regressive
            net_out_put_type (str), type of the loss, it determines the output of the network
            dist_cls (str), distribution class, only activate if output is a distribution

        Returns:
            proj_layer: nn.Module
            projection layer that maps the features to the final output
            required_padding_value: float,
            which values need to be padded when loadding the data

        """
        if net_out_put_type == 'distribution':
            if dist_cls not in ALL_DISTRIBUTIONS.keys():
                raise ValueError(f'Unsupported distribution class type: {dist_cls}')
            proj_layer = ALL_DISTRIBUTIONS[dist_cls](num_in_features=input_shape,
                                                     output_shape=output_shape[1:],
                                                     n_prediction_heads=n_prediction_heads,
                                                     auto_regressive=auto_regressive)
            return proj_layer
        elif net_out_put_type == 'regression':
            if auto_regressive:
                proj_layer = nn.Sequential(nn.Linear(input_shape, np.product(output_shape[1:])))
            else:
                proj_layer = nn.Sequential(nn.Unflatten(-1, (n_prediction_heads, input_shape)),
                                       nn.Linear(input_shape, np.product(output_shape[1:])),
                                       # nn.Unflatten(-1, tuple(output_shape)),
                                       )
            return proj_layer
        else:
            raise ValueError(f"Unsupported network type "
                             f"{net_out_put_type} (should be regression or distribution)")

    @staticmethod
    def get_hyperparameter_search_space(
            dataset_properties: Optional[Dict[str, BaseDatasetPropertiesType]] = None
    ) -> ConfigurationSpace:
        """Return the configuration space of this classification algorithm.

        Args:
            dataset_properties (Optional[Dict[str, Union[str, int]]):
                Describes the dataset to work on

        Returns:
            ConfigurationSpace:
                The configuration space of this algorithm.
        """
        cs = ConfigurationSpace()
        return cs
