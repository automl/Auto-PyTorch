from typing import Any, Dict, Optional, Union

from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import (
    UniformIntegerHyperparameter
)

import numpy as np

from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomTreesEmbedding as SklearnRandomTreesEmbedding

from autoPyTorch.datasets.base_dataset import BaseDatasetPropertiesType
from autoPyTorch.pipeline.components.preprocessing.tabular_preprocessing.feature_preprocessing. \
    base_feature_preprocessor import autoPyTorchFeaturePreprocessingComponent
from autoPyTorch.pipeline.components.preprocessing.tabular_preprocessing.feature_preprocessing. \
    utils import NoneType_
from autoPyTorch.utils.common import HyperparameterSearchSpace, add_hyperparameter, check_none


class RandomTreesEmbedding(autoPyTorchFeaturePreprocessingComponent):
    def __init__(self, n_estimators: int = 10,
                 max_depth: Union[int, NoneType_] = 5, min_samples_split: int = 2,
                 min_samples_leaf: int = 1,
                 max_leaf_nodes: Union[int, NoneType_] = "none",
                 sparse_output: bool = False,
                 random_state: Optional[np.random.RandomState] = None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_leaf_nodes = max_leaf_nodes
        self.sparse_output = sparse_output

        super().__init__(random_state=random_state)

    def get_components_kwargs(self) -> Dict[str, Any]:
        """
        returns keyword arguments required by the feature preprocessor

        Returns:
            Dict[str, Any]: kwargs
        """
        return dict(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            max_leaf_nodes=self.max_leaf_nodes,
            sparse_output=self.sparse_output,
            random_state=self.random_state
        )

    def fit(self, X: Dict[str, Any], y: Any = None) -> BaseEstimator:
        if check_none(self.max_leaf_nodes):
            self.max_leaf_nodes = None
        if check_none(self.max_depth):
            self.max_depth = None

        self.preprocessor['numerical'] = SklearnRandomTreesEmbedding(**self.get_components_kwargs())
        return self

    @staticmethod
    def get_properties(dataset_properties: Optional[Dict[str, BaseDatasetPropertiesType]] = None) -> Dict[str, Any]:
        return {'shortname': 'RandomTreesEmbedding',
                'name': 'Random Trees Embedding',
                'handles_sparse': True,
                'handles_classification': True,
                'handles_regression': True
                }

    @staticmethod
    def get_hyperparameter_search_space(
        dataset_properties: Optional[Dict[str, BaseDatasetPropertiesType]] = None,
        n_estimators: HyperparameterSearchSpace = HyperparameterSearchSpace(hyperparameter='n_estimators',
                                                                            value_range=(10, 100),
                                                                            default_value=10,
                                                                            ),
        max_depth: HyperparameterSearchSpace = HyperparameterSearchSpace(hyperparameter='max_depth',
                                                                         value_range=(2, 10),
                                                                         default_value=5,
                                                                         ),
        min_samples_split: HyperparameterSearchSpace = HyperparameterSearchSpace(hyperparameter='min_samples_split',
                                                                                 value_range=(2, 20),
                                                                                 default_value=2,
                                                                                 ),
        min_samples_leaf: HyperparameterSearchSpace = HyperparameterSearchSpace(hyperparameter='min_samples_leaf',
                                                                                value_range=(1, 20),
                                                                                default_value=1,
                                                                                ),
        max_leaf_nodes: HyperparameterSearchSpace = HyperparameterSearchSpace(hyperparameter='max_leaf_nodes',
                                                                              value_range=("none",),
                                                                              default_value="none",
                                                                              ),
    ) -> ConfigurationSpace:

        cs = ConfigurationSpace()
        add_hyperparameter(cs, n_estimators, UniformIntegerHyperparameter)
        add_hyperparameter(cs, max_depth, UniformIntegerHyperparameter)
        add_hyperparameter(cs, min_samples_split, UniformIntegerHyperparameter)
        add_hyperparameter(cs, min_samples_leaf, UniformIntegerHyperparameter)
        add_hyperparameter(cs, max_leaf_nodes, UniformIntegerHyperparameter)

        return cs
