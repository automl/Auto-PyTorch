from typing import Any, Dict, Optional, Union

from ConfigSpace.configuration_space import ConfigurationSpace

import numpy as np

from torch import nn

from autoPyTorch.pipeline.components.setup.network_embedding.base_network_embedding import NetworkEmbeddingComponent


class _NoEmbedding(nn.Module):
    def __init__(self, num_input_features, num_numerical_features):
        super().__init__()
        self.n_feats = num_input_features
        self.num_numerical = num_numerical_features

    def forward(self, x):
        return x


class NoEmbedding(NetworkEmbeddingComponent):
    """
    Class to learn an embedding for categorical hyperparameters.
    """

    def __init__(self, random_state: Optional[Union[np.random.RandomState, int]] = None):
        super().__init__(random_state=random_state)

    def build_embedding(self, num_input_features, num_numerical_features) -> nn.Module:
        return _NoEmbedding(num_input_features=num_input_features,
                            num_numerical_features=num_numerical_features)

    @staticmethod
    def get_hyperparameter_search_space(
        dataset_properties: Optional[Dict[str, str]] = None,
    ) -> ConfigurationSpace:
        cs = ConfigurationSpace()
        return cs

    @staticmethod
    def get_properties(dataset_properties: Optional[Dict[str, Any]] = None) -> Dict[str, Union[str, bool]]:
        return {
            'shortname': 'no embedding',
            'name': 'NoEmbedding',
            'handles_tabular': True,
            'handles_image': False,
            'handles_time_series': False,
        }