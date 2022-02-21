from abc import abstractmethod, ABC
from typing import Any, Dict, Iterable, Tuple, List, Optional

import torch
from torch import nn

from autoPyTorch.pipeline.components.setup.network_backbone.utils import get_output_shape
from autoPyTorch.utils.common import FitRequirement
from autoPyTorch.pipeline.components.base_component import BaseEstimator, autoPyTorchComponent


class RecurrentDecoderNetwork(nn.Module):
    def forward(self, x_future: torch.Tensor, features_latent: torch.Tensor):
        """
        Base forecasting Decoder Network, its output needs to be a 3-d Tensor:


        Args:
            x_future torch.Tensor(B, L_future, N_out), the future features
            features_latent: torch.Tensor(B, L_encoder, N), output of the encoder network, or the hidden states
        Returns:
            net_output: torch.Tensor with shape either (B, L_future, N)

        """
        raise NotImplementedError


class BaseForecastingDecoder(autoPyTorchComponent):
    """
    Base class for network heads used for forecasting.
     Holds the head module and the config which was used to create it.
    """
    _required_properties = ["name", "shortname", "handles_tabular", "handles_image", "handles_time_series"]

    def __init__(self,
                 **kwargs: Any):
        super().__init__()
        self.add_fit_requirements(self._required_fit_requirements)

        self.config = kwargs
        self.decoder: Optional[nn.Module] = None
        self.n_decoder_output_features = None
        self.n_prediction_heads = 1

    @property
    def _required_fit_requirements(self) -> List[FitRequirement]:
        return [
            FitRequirement('task_type', (str,), user_defined=True, dataset_property=True),
            FitRequirement('network_encoder', (nn.Module,), user_defined=False, dataset_property=False),
            FitRequirement('encoder_properties', (Dict,), user_defined=False, dataset_property=False),
            FitRequirement('window_size', (int,), user_defined=False, dataset_property=False),
        ]

    @property
    def fitted_encoder(self):
        return []

    def decoder_properties(self):
        decoder_properties = {'has_hidden_states': False,
                              'has_local_layer': True,
                              'recurrent': False,
                              'lagged_input': False,
                              'multi_blocks': False,
                              'mask_on_future_target': False,
                              }
        return decoder_properties

    def fit(self, X: Dict[str, Any], y: Any = None) -> BaseEstimator:
        """
        Builds the head component and assigns it to self.decoder

        Args:
            X (X: Dict[str, Any]): Dependencies needed by current component to perform fit
            y (Any): not used. To comply with sklearn API
        Returns:
            Self
        """
        self.check_requirements(X, y)
        input_shape = X['dataset_properties']['input_shape']
        output_shape = X['dataset_properties']['output_shape']

        auto_regressive = self.auto_regressive

        X.update({"auto_regressive": auto_regressive})

        if auto_regressive:
            self.n_prediction_heads = 1
        else:
            self.n_prediction_heads = output_shape[0]

        encoder_properties = X['encoder_properties']
        has_hidden_states = encoder_properties.get("has_hidden_states", False)

        self.decoder, self.n_decoder_output_features = self.build_decoder(
            input_shape=get_output_shape(X['network_encoder'], input_shape=input_shape,
                                         has_hidden_states=has_hidden_states),
            n_prediction_heads=self.n_prediction_heads,
            dataset_properties=X['dataset_properties']
        )

        X['n_decoder_output_features'] = self.n_decoder_output_features
        return self

    def transform(self, X: Dict[str, Any]) -> Dict[str, Any]:
        """
        Adds the network head into the fit dictionary 'X' and returns it.

        Args:
            X (Dict[str, Any]): 'X' dictionary
        Returns:
            (Dict[str, Any]): the updated 'X' dictionary
        """
        X.update({'decoder_properties': self.decoder_properties(),
                  'network_decoder': self.decoder,
                  'n_prediction_heads': self.n_prediction_heads,
                  'n_decoder_output_features': self.n_decoder_output_features})

        return X

    def build_decoder(self,
                      input_shape: Tuple[int, ...],
                      n_prediction_heads: int,
                      dataset_properties: Dict) -> Tuple[nn.Module, int]:
        """
        Builds the head module and returns it

        Args:
            input_shape (Tuple[int, ...]): shape of the input to the head (usually the shape of the backbone output)
            n_prediction_heads (int): how many prediction heads the network has, used for final forecasting heads
            dataset_properties (Dict): dataset properties
        Returns:
            nn.Module: head module
        """
        decoder, n_decoder_features = self._build_decoder(input_shape, n_prediction_heads, dataset_properties)
        return decoder, n_decoder_features

    @abstractmethod
    def _build_decoder(self, input_shape: Tuple[int, ...], n_prediction_heads: int,
                       dataset_properties:Dict) -> Tuple[nn.Module, int]:
        """
        Builds the head module and returns it

        Args:
            input_shape (Tuple[int, ...]): shape of the input to the head (usually the shape of the backbone output)
            n_prediction_heads (int): how many prediction heads will be generated after the encoder

        Returns:
            decoder (nn.Module): decoder module
            n_decoder_features (int): output of decoder features, used for initialize network head.
        """
        raise NotImplementedError()

    @classmethod
    def get_name(cls) -> str:
        """
        Get the name of the head

        Args:
            None

        Returns:
            str: Name of the head
        """
        return str(cls.get_properties()["shortname"])

