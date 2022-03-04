import numpy as np
from collections import OrderedDict

import pandas as pd
from scipy.sparse import csr_matrix

import torch
import torchvision
from ConfigSpace import ConfigurationSpace
from autoPyTorch.utils.common import FitRequirement
from torch import nn
from abc import abstractmethod
from typing import Any, Dict, Iterable, Optional, Tuple, List, Union, NamedTuple

from autoPyTorch.pipeline.components.base_component import BaseEstimator
from autoPyTorch.datasets.base_dataset import BaseDatasetPropertiesType
from autoPyTorch.pipeline.components.setup.network_backbone.utils import get_output_shape
from autoPyTorch.pipeline.components.base_component import autoPyTorchComponent
from autoPyTorch.pipeline.components.setup.network_backbone.forecasting_backbone.forecasting_encoder.components import (
    EncoderProperties, EncoderBlockInfo, EncoderNetwork
)


class BaseForecastingEncoder(autoPyTorchComponent):
    """
    Base class for network backbones. Holds the backbone module and the config which was used to create it.
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
            FitRequirement('X_train', (np.ndarray, pd.DataFrame, csr_matrix), user_defined=True,
                           dataset_property=False),
            FitRequirement('y_train', (np.ndarray, pd.DataFrame, csr_matrix), user_defined=True,
                           dataset_property=False),
            FitRequirement('uni_variant', (bool,), user_defined=False, dataset_property=True),
            FitRequirement('input_shape', (Iterable,), user_defined=True, dataset_property=True),
            FitRequirement('output_shape', (Iterable,), user_defined=True, dataset_property=True),
            FitRequirement('static_features_shape', (int,), user_defined=True, dataset_property=True),
        ]

    def fit(self, X: Dict[str, Any], y: Any = None) -> BaseEstimator:
        self.check_requirements(X, y)
        X_train = X['X_train']
        y_train = X['y_train']

        input_shape = X["dataset_properties"]['input_shape']
        output_shape = X["dataset_properties"]['output_shape']
        static_features_shape = X["dataset_properties"]["static_features_shape"]

        if self.block_number == 1:
            if not X["dataset_properties"]["uni_variant"]:
                if not X["dataset_properties"]["is_small_preprocess"]:
                    # get input shape by transforming first two elements of the training set
                    transforms = torchvision.transforms.Compose(X['preprocess_transforms'])
                    X_train = X_train[:1, np.newaxis, ...]
                    X_train = transforms(X_train)
                    input_shape = np.concatenate(X_train).shape[1:]

            if 'network_embedding' in X.keys():
                input_shape = get_output_shape(X['network_embedding'], input_shape=input_shape)

            in_features = input_shape[-1]

            variable_selection = X.get("variable_selection", False)
            if variable_selection:
                in_features = self.n_encoder_output_feature()
            elif self.encoder_properties().lagged_input and hasattr(self, 'lagged_value'):
                in_features = len(self.lagged_value) * output_shape[-1] + input_shape[-1] + static_features_shape
            else:
                in_features = output_shape[-1] + input_shape[-1] + static_features_shape

            input_shape = (X['window_size'], in_features)
        else:
            input_shape = X['encoder_output_shape']

        self.encoder = self.build_encoder(
            input_shape=input_shape,
        )

        self.input_shape = input_shape

        has_hidden_states = self.encoder_properties().has_hidden_states
        self.encoder_output_shape = get_output_shape(self.encoder, input_shape, has_hidden_states)
        return self

    @staticmethod
    def allowed_decoders():
        raise NotImplementedError

    @abstractmethod
    def n_encoder_output_feature(self) -> int:
        raise NotImplementedError

    def n_hidden_states(self) -> int:
        return 0

    def transform(self, X: Dict[str, Any]) -> Dict[str, Any]:
        X['dataset_properties'].update({'input_shape': self.input_shape})
        network_encoder = X.get('network_encoder', OrderedDict())
        network_encoder[f'block_{self.block_number}'] = EncoderBlockInfo(encoder=self.encoder,
                                                                         encoder_properties=self.encoder_properties(),
                                                                         encoder_input_shape=self.input_shape,
                                                                         encoder_output_shape=self.encoder_output_shape,
                                                                         n_hidden_states=self.n_hidden_states())

        X.update({f'network_encoder': network_encoder})
        return X

    @abstractmethod
    def build_encoder(self,
                      input_shape: Tuple[int, ...]) -> nn.Module:
        """
        Builds the backbone module and returns it

        Args:
            targets_shape (Tuple[int, ...]): shape of target
            input_shape (Tuple[int, ...]): input feature shape
            static_feature_shape (int): static feature shape.

        Returns:
            nn.Module: backbone module
        """
        raise NotImplementedError()

    @staticmethod
    def encoder_properties() -> EncoderProperties:
        """
        Encoder properties, this determines how the data flows over the forecasting networks

        has_hidden_states, it determines if the network contains hidden states and thus return or accept the hidden
        states
        bijective_seq_output, determines if the network returns a sequence with the same sequence length as the input
        sequence when output_seq is set True
        fix_input_shape if the input shape is fixed, this is useful for building network head
        lagged_input, if lagged input values are applied, this technique is implemented in DeepAR and Transformer
        implemented in gluonTS:
        https://github.com/awslabs/gluon-ts/blob/master/src/gluonts/torch/model/deepar/module.py
        """
        encoder_properties = EncoderProperties()
        return encoder_properties
