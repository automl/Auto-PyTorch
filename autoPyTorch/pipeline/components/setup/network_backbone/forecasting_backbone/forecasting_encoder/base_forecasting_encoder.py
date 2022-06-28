from abc import abstractmethod
from collections import OrderedDict
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

from torch import nn

import torchvision

from autoPyTorch.pipeline.components.base_component import BaseEstimator, autoPyTorchComponent
from autoPyTorch.pipeline.components.setup.network_backbone.forecasting_backbone.components_util import NetworkStructure
from autoPyTorch.pipeline.components.setup.network_backbone.forecasting_backbone.forecasting_encoder.components import (
    EncoderBlockInfo,
    EncoderProperties
)
from autoPyTorch.pipeline.components.setup.network_backbone.utils import get_output_shape
from autoPyTorch.utils.common import FitRequirement


class BaseForecastingEncoder(autoPyTorchComponent):
    """
    Base class for network backbones. Holds the encoder module and the config which was used to create it.
    """
    _required_properties = ["name", "shortname", "handles_tabular", "handles_image", "handles_time_series"]

    def __init__(self,
                 block_number: int = 1,
                 **kwargs: Any):
        autoPyTorchComponent.__init__(self)
        self.add_fit_requirements(
            self._required_fit_arguments
        )
        self.encoder: nn.Module = None
        self.config = kwargs
        self.input_shape: Optional[Tuple[int, ...]] = None
        self.block_number = block_number
        self.encoder_output_shape: Optional[Tuple[int, ...]] = None

    @property
    def _required_fit_arguments(self) -> List[FitRequirement]:
        return [
            FitRequirement('is_small_preprocess', (bool,), user_defined=True, dataset_property=True),
            FitRequirement('uni_variant', (bool,), user_defined=False, dataset_property=True),
            FitRequirement('input_shape', (Iterable,), user_defined=True, dataset_property=True),
            FitRequirement('output_shape', (Iterable,), user_defined=True, dataset_property=True),
            FitRequirement('network_structure', (NetworkStructure,), user_defined=False, dataset_property=False),
            FitRequirement('transform_time_features', (bool,), user_defined=False, dataset_property=False),
            FitRequirement('time_feature_transform', (Iterable,), user_defined=False, dataset_property=True),
            FitRequirement('network_embedding', (nn.Module, ), user_defined=False, dataset_property=False),
            FitRequirement('window_size', (int,), user_defined=False, dataset_property=False)
        ]

    def fit(self, X: Dict[str, Any], y: Any = None) -> BaseEstimator:
        self.check_requirements(X, y)

        input_shape = (*X["dataset_properties"]['input_shape'][:-1], 0)
        output_shape = X["dataset_properties"]['output_shape']

        if self.block_number == 1:
            if not X["dataset_properties"]["uni_variant"]:
                X_train = X.get('X_train', None)
                if X_train is None:
                    raise ValueError('Non uni_variant dataset must contain X_train!')

                if X["dataset_properties"]["is_small_preprocess"]:
                    input_shape = X_train.shape[1:]
                else:
                    # get input shape by transforming first two elements of the training set
                    transforms = torchvision.transforms.Compose(X['preprocess_transforms'])
                    X_train = X_train.values[:1, np.newaxis, ...]
                    X_train = transforms(X_train)
                    input_shape = np.concatenate(X_train).shape[1:]

            if X['transform_time_features']:
                n_time_feature_transform = len(X['dataset_properties']['time_feature_transform'])
            else:
                n_time_feature_transform = 0

            input_shape = (*input_shape[:-1], input_shape[-1] + n_time_feature_transform)

            if 'network_embedding' in X.keys():
                input_shape = get_output_shape(X['network_embedding'], input_shape=input_shape)

            variable_selection = X['network_structure'].variable_selection
            if variable_selection:
                in_features = self.n_encoder_output_feature()
            elif self.encoder_properties().lagged_input and hasattr(self, 'lagged_value'):
                in_features = len(self.lagged_value) * output_shape[-1] + input_shape[-1]
            else:
                in_features = output_shape[-1] + input_shape[-1]

            input_shape = (X['window_size'], in_features)
        else:
            if 'network_encoder' not in X or f'block_{self.block_number -1}' not in X['network_encoder']:
                raise ValueError('Lower block layers must be fitted and transformed first!')
            network_block_info = X['network_encoder'][f'block_{self.block_number -1}']
            input_shape = network_block_info.encoder_output_shape

        self.encoder = self.build_encoder(
            input_shape=input_shape,
        )

        self.input_shape = input_shape

        has_hidden_states = self.encoder_properties().has_hidden_states
        self.encoder_output_shape = get_output_shape(self.encoder, input_shape, has_hidden_states)
        if self.n_encoder_output_feature() != self.encoder_output_shape[-1]:
            raise ValueError(f'n_encoder_output_feature ({ self.n_encoder_output_feature()}) '
                             f'must equal to the output dimension f({self.encoder_output_shape})')
        return self

    @staticmethod
    def allowed_decoders() -> List[str]:
        raise NotImplementedError

    @abstractmethod
    def n_encoder_output_feature(self) -> int:
        # We need this to compute the output of the variable selection network
        raise NotImplementedError

    def n_hidden_states(self) -> int:
        return 0

    def transform(self, X: Dict[str, Any]) -> Dict[str, Any]:
        X['dataset_properties'].update({'input_shape': self.input_shape})
        network_encoder = X.get('network_encoder', OrderedDict())
        assert self.input_shape is not None
        assert self.encoder_output_shape is not None
        network_encoder[f'block_{self.block_number}'] = EncoderBlockInfo(encoder=self.encoder,
                                                                         encoder_properties=self.encoder_properties(),
                                                                         encoder_input_shape=self.input_shape,
                                                                         encoder_output_shape=self.encoder_output_shape,
                                                                         n_hidden_states=self.n_hidden_states())

        X.update({'network_encoder': network_encoder})
        return X

    @abstractmethod
    def build_encoder(self,
                      input_shape: Tuple[int, ...]) -> nn.Module:
        """
        Builds the backbone module and returns it

        Args:
            input_shape (Tuple[int, ...]):
                input feature shape

        Returns:
            nn.Module: backbone module
        """
        pass

    @staticmethod
    def encoder_properties() -> EncoderProperties:
        """
        Encoder properties, this determines how the data flows over the forecasting networks

        """
        encoder_properties = EncoderProperties()
        return encoder_properties
