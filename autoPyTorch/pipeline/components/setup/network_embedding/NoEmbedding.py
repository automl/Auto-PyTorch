from typing import Dict, Optional, Union

from ConfigSpace.configuration_space import ConfigurationSpace

import numpy as np

import torch
from torch import nn

from autoPyTorch.datasets.base_dataset import BaseDatasetPropertiesType
from autoPyTorch.pipeline.components.setup.network_embedding.base_network_embedding import NetworkEmbeddingComponent


class _NoEmbedding(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class NoEmbedding(NetworkEmbeddingComponent):
    """
    Class to learn an embedding for categorical hyperparameters.
    """

    def __init__(self, random_state: Optional[Union[np.random.RandomState, int]] = None):
        super().__init__(random_state=random_state)

    def build_embedding(self, num_input_features: np.ndarray, num_numerical_features: int) -> nn.Module:
        return _NoEmbedding()

    @staticmethod
    def get_hyperparameter_search_space(
        dataset_properties: Optional[Dict[str, BaseDatasetPropertiesType]] = None,
    ) -> ConfigurationSpace:
        cs = ConfigurationSpace()
        return cs

    @staticmethod
    def get_properties(dataset_properties: Optional[Dict[str, BaseDatasetPropertiesType]] = None
                       ) -> Dict[str, Union[str, bool]]:
        return {
            'shortname': 'no embedding',
            'name': 'NoEmbedding',
            'handles_tabular': True,
            'handles_image': False,
            'handles_time_series': False,
        }
