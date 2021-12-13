from abc import abstractmethod, ABC
from typing import Any, Dict, Iterable, Tuple, List, Optional

import numpy as np
import torch
from torch import nn

from autoPyTorch.pipeline.components.base_component import BaseEstimator
from autoPyTorch.pipeline.components.setup.network_head.base_network_head import NetworkHeadComponent
from autoPyTorch.pipeline.components.setup.network_backbone.utils import get_output_shape
from autoPyTorch.utils.common import FitRequirement
from autoPyTorch.pipeline.components.setup.network_head.forecasting_network_head.distribution import \
    ALL_DISTRIBUTIONS, ProjectionLayer


class ForecastingHead(NetworkHeadComponent):
    """
    Base class for network heads used for forecasting.
     Holds the head module and the config which was used to create it.
    """
    _required_properties = ["name", "shortname", "handles_tabular", "handles_image", "handles_time_series",
                            "n_prediction_steps"]

    def __init__(self,
                 **kwargs: Any):
        super().__init__()
        self.add_fit_requirements(self._required_fit_requirements)
        self.head: Optional[nn.Module] = None
        self.required_net_out_put_type: Optional[str] = None
        self.auto_regressive = kwargs.get('auto_regressive', False)

        self.config = kwargs


    @property
    def _required_fit_requirements(self) -> List[FitRequirement]:
        return [
            FitRequirement('input_shape', (Iterable,), user_defined=True, dataset_property=True),
            FitRequirement('task_type', (str,), user_defined=True, dataset_property=True),
            FitRequirement('encoder_properties', (str,), user_defined=False, dataset_property=False),
            FitRequirement('n_prediction_steps', (int,), user_defined=True, dataset_property=True),
            FitRequirement('output_shape', (Iterable, int), user_defined=True, dataset_property=True),
            FitRequirement('window_size', (int,), user_defined=False, dataset_property=False),
            FitRequirement('loss_type', (str,), user_defined=False, dataset_property=False)
        ]

    @property
    def decoder_properties(self):
        decoder_property = {'additional_output': False,
                            'additional_input': False,
                            'fixed_input_seq_length': False,
                            'recurrent': False,
                            }
        return decoder_property

    def fit(self, X: Dict[str, Any], y: Any = None) -> BaseEstimator:
        """
        Builds the head component and assigns it to self.head

        Args:
            X (X: Dict[str, Any]): Dependencies needed by current component to perform fit
            y (Any): not used. To comply with sklearn API
        Returns:
            Self
        """
        input_shape = X['dataset_properties']['input_shape']
        output_shape = X['dataset_properties']['output_shape']

        self.required_net_out_put_type = X['required_net_out_put_type']

        if self.required_net_out_put_type == 'distribution':
            if 'dist_cls' not in X:
                raise ValueError('Distribution output type must contain dist_cls!!')

        dist_cls = X.get('dist_cls', None)



        auto_regressive = self.auto_regressive

        auto_regressive = False # TODO implement auto_regressive mdoels!!



        X.update({"auto_regressive": auto_regressive})
        encoder_properties = X['encoder_properties']

        # for information about encoder_properties, please check
        # autoPyTorch.pipeline.components.setup.network_backbone.forecasting_network_backbone.base_forecasting_backbone
        # TODO create a separate module so that users could know what is contained in encoder_properties

        # TODO consider Auto-regressive model on vanilla network head
        if auto_regressive:
            n_prediction_heads = 1
        else:
            n_prediction_heads = output_shape[0]
        # output shape now doe not contain information about n_prediction_steps

        fixed_input_seq_length = encoder_properties.get("fixed_input_seq_length", False)
        has_hidden_states = encoder_properties.get("has_hidden_states", False)

        if fixed_input_seq_length:
            input_shape = (X["window_size"], input_shape[-1])

        arch_kwargs = {'n_prediction_heads': n_prediction_heads}

        self.head = self.build_head(
            input_shape=get_output_shape(X['network_backbone'], input_shape=input_shape,
                                         has_hidden_states=has_hidden_states),
            output_shape=output_shape,
            auto_regressive=auto_regressive,
            dist_cls=dist_cls,
            **arch_kwargs,
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
        X.update({'decoder_properties': self.decoder_properties})
        return super().transform(X)

    def build_head(self,
                   input_shape: Tuple[int, ...],
                   output_shape: Tuple[int, ...],
                   auto_regressive: bool = False,
                   dist_cls: Optional[str] = None,
                   **arch_kwargs: Dict) -> nn.Module:
        """
        Builds the head module and returns it

        Args:
            input_shape (Tuple[int, ...]): shape of the input to the head (usually the shape of the backbone output)
            output_shape (Tuple[int, ...]): shape of the output of the head
            auto_regressive (bool): if the network is auto-regressive
            dist_cls (Optional[str]): output distribution, only works if required_net_out_put_type is 'distribution'
            arch_kwargs (Dict): additional paramter for initializing architectures.

        Returns:
            nn.Module: head module
        """
        base_header_layer, num_head_base_output_features = self._build_head(input_shape, **arch_kwargs)
        proj_layer = []

        output_layer = self.build_proj_layer(
            num_head_base_output_features=num_head_base_output_features,
            output_shape=output_shape,
            auto_regressive=auto_regressive,
            net_out_put_type=self.required_net_out_put_type,
            dist_cls=dist_cls,
            n_prediction_heads=arch_kwargs['n_prediction_heads']
        )
        proj_layer.append(output_layer)
        return nn.Sequential(*base_header_layer, *proj_layer)

    @abstractmethod
    def _build_head(self, input_shape: Tuple[int, ...], **arch_kwargs) -> Tuple[List[nn.Module], int]:
        """
        Builds the head module and returns it

        Args:
            input_shape (Tuple[int, ...]): shape of the input to the head (usually the shape of the backbone output)
            output_shape (Tuple[int, ...]): shape of the output of the head
            n_prediction_steps (int): how many steps need to be predicted in advance

        Returns:
            nn.Module: head module
        """
        raise NotImplementedError()

    @staticmethod
    def build_proj_layer(num_head_base_output_features: int,
                         output_shape: Tuple[int, ...],
                         n_prediction_heads: int,
                         auto_regressive: bool,
                         net_out_put_type: str,
                         dist_cls: Optional[str] = None) -> torch.nn.Module:
        """
        a final layer that project the head output to the final distribution
        Args:
            num_head_base_output_features (int): output feature of head base,
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
            proj_layer = ALL_DISTRIBUTIONS[dist_cls](num_in_features=num_head_base_output_features,
                                                     output_shape=output_shape[1:],
                                                     n_prediction_heads=n_prediction_heads,
                                                     auto_regressive=auto_regressive)
            return proj_layer
        elif net_out_put_type == 'regression':
            proj_layer = nn.Sequential(nn.Unflatten(-1, (n_prediction_heads, num_head_base_output_features)),
                                       nn.Linear(num_head_base_output_features, np.product(output_shape[1:])),
                                       # nn.Unflatten(-1, tuple(output_shape)),
                                       )
            return proj_layer
        else:
            raise ValueError(f"Unsupported network type "
                             f"{net_out_put_type} (should be regression or distribution)")
