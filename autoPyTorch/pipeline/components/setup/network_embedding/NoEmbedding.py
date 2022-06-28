from typing import Dict, List, Optional, Tuple, Union

from ConfigSpace.configuration_space import ConfigurationSpace

import numpy as np

import torch
from torch import nn

from autoPyTorch.datasets.base_dataset import BaseDatasetPropertiesType
from autoPyTorch.pipeline.components.setup.network_embedding.base_network_embedding import NetworkEmbeddingComponent


class _NoEmbedding(nn.Module):
    def get_partial_models(self, subset_features: List[int]) -> "_NoEmbedding":
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class NoEmbedding(NetworkEmbeddingComponent):
    """
    Class to learn an embedding for categorical hyperparameters.
    """

    def __init__(self, random_state: Optional[Union[np.random.RandomState, int]] = None):
        super().__init__(random_state=random_state)

    def build_embedding(self,
                        num_input_features: np.ndarray,
                        num_numerical_features: int) -> Tuple[nn.Module, Optional[List[int]]]:
        return _NoEmbedding(), None

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
            'handles_time_series': True,
        }
