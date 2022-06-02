from math import ceil
from typing import Any, Dict, List, Optional, Union

from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import (
    CategoricalHyperparameter,
    UniformFloatHyperparameter,
    UniformIntegerHyperparameter,
)

import numpy as np

import torch
from torch import nn

from autoPyTorch.datasets.base_dataset import BaseDatasetPropertiesType
from autoPyTorch.pipeline.components.setup.network_embedding.base_network_embedding import NetworkEmbeddingComponent
from autoPyTorch.utils.common import HyperparameterSearchSpace, add_hyperparameter


def get_num_output_dimensions(config, num_categs_per_feature):
    """ Returns list of embedding sizes for each categorical variable.
        Selects this adaptively based on training_datset.
        Note: Assumes there is at least one embed feature.
    """
    max_embedding_dim = config['max_embedding_dim']
    embed_exponent = config['embed_exponent']
    size_factor = config['embedding_size_factor']
    num_output_dimensions = [int(size_factor*max(
                                                 2,
                                                 min(max_embedding_dim,
                                                     1.6 * num_categories**embed_exponent)))
                             if num_categories > 0 else 1 for num_categories in num_categs_per_feature]
    return num_output_dimensions


class _LearnedEntityEmbedding(nn.Module):
    """ Learned entity embedding module for categorical features"""

    def __init__(self, config: Dict[str, Any], num_categories_per_col: np.ndarray, num_features_excl_embed: int):
        """
        Args:
            config (Dict[str, Any]): The configuration sampled by the hyperparameter optimizer
            num_input_features (np.ndarray): column wise information of number of output columns after transformation
                for each categorical column and 0 for numerical columns
            num_features_excl_embed (int): number of features in X excluding the features that need to be embedded
        """
        super().__init__()
        self.config = config
        # list of number of categories of categorical data
        # or 0 for numerical data
        self.num_categories_per_col = num_categories_per_col
        self.embed_features = self.num_categories_per_col > 0

        self.num_embed_features = self.num_categories_per_col[self.embed_features]

        self.num_output_dimensions = get_num_output_dimensions(config, self.num_categories_per_col)

        self.num_out_feats = num_features_excl_embed + sum(self.num_output_dimensions)

        self.ee_layers = self._create_ee_layers()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # pass the columns of each categorical feature through entity embedding layer
        # before passing it through the model
        concat_seq = []

        x_pointer = 0
        layer_pointer = 0
        for x_pointer, embed in enumerate(self.embed_features):
            current_feature_slice = x[:, x_pointer]
            if not embed:
                x_pointer += 1
                concat_seq.append(current_feature_slice.view(-1, 1))
                continue
            current_feature_slice = current_feature_slice.to(torch.int)
            concat_seq.append(self.ee_layers[layer_pointer](current_feature_slice))
            layer_pointer += 1

        return torch.cat(concat_seq, dim=1)

    def _create_ee_layers(self) -> nn.ModuleList:
        # entity embeding layers are Linear Layers
        layers = nn.ModuleList()
        for num_cat, embed, num_out in zip(self.num_categories_per_col,
                                           self.embed_features,
                                           self.num_output_dimensions):
            if not embed:
                continue
            layers.append(nn.Embedding(num_cat, num_out))
        return layers


class LearnedEntityEmbedding(NetworkEmbeddingComponent):
    """
    Class to learn an embedding for categorical hyperparameters.
    """

    def __init__(self, random_state: Optional[np.random.RandomState] = None, **kwargs: Any):
        super().__init__(random_state=random_state)
        self.config = kwargs

    def build_embedding(self, num_categories_per_col: np.ndarray, num_features_excl_embed: int) -> nn.Module:
        return _LearnedEntityEmbedding(config=self.config,
                                       num_categories_per_col=num_categories_per_col,
                                       num_features_excl_embed=num_features_excl_embed)

    @staticmethod
    def get_hyperparameter_search_space(
        dataset_properties: Optional[Dict[str, BaseDatasetPropertiesType]] = None,
        embed_exponent: HyperparameterSearchSpace = HyperparameterSearchSpace(hyperparameter="embed_exponent",
                                                                                   value_range=(0.56,),
                                                                                   default_value=0.56),
        max_embedding_dim: HyperparameterSearchSpace = HyperparameterSearchSpace(hyperparameter="max_embedding_dim",
                                                                                   value_range=(100,),
                                                                                   default_value=100),
        embedding_size_factor: HyperparameterSearchSpace = HyperparameterSearchSpace(hyperparameter="embedding_size_factor",
                                                                                     value_range=(1.0, 0.5, 1.5, 0.7, 0.6, 0.8, 0.9, 1.1, 1.2, 1.3, 1.4),
                                                                                     default_value=1,
                                                                                     ),
    ) -> ConfigurationSpace:
        cs = ConfigurationSpace()
        if dataset_properties is not None:
            if len(dataset_properties['categorical_columns']) > 0:
                add_hyperparameter(cs, embed_exponent, UniformFloatHyperparameter)
                add_hyperparameter(cs, max_embedding_dim, UniformIntegerHyperparameter)
                add_hyperparameter(cs, embedding_size_factor, CategoricalHyperparameter)

        return cs

    @staticmethod
    def get_properties(dataset_properties: Optional[Dict[str, BaseDatasetPropertiesType]] = None
                       ) -> Dict[str, Union[str, bool]]:
        return {
            'shortname': 'embedding',
            'name': 'LearnedEntityEmbedding',
            'handles_tabular': True,
            'handles_image': False,
            'handles_time_series': False,
        }
