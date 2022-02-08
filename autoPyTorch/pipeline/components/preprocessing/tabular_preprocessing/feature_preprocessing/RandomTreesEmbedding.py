from typing import Any, Dict, Optional, Union

from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import (
    CategoricalHyperparameter,
    UniformFloatHyperparameter,
    UniformIntegerHyperparameter
)

import numpy as np

from sklearn.ensemble import RandomTreesEmbedding as SklearnRandomTreesEmbedding
from sklearn.base import BaseEstimator

from autoPyTorch.datasets.base_dataset import BaseDatasetPropertiesType
from autoPyTorch.pipeline.components.preprocessing.tabular_preprocessing.feature_preprocessing. \
    base_feature_preprocessor import autoPyTorchFeaturePreprocessingComponent
from autoPyTorch.utils.common import HyperparameterSearchSpace, add_hyperparameter


class RandomTreesEmbedding(autoPyTorchFeaturePreprocessingComponent):
    def __init__(self, bootstrap: bool = True, n_estimators: int = 10,
                 max_depth: Optional[Union[str, int]] = 5, min_samples_split: int = 2,
                 min_samples_leaf: int = 1, min_weight_fraction_leaf: bool = True,
                 max_leaf_nodes: Optional[Union[str, int]] = "none",
                 sparse_output: bool = False,
                 random_state: Optional[np.random.RandomState] = None):
        self.bootstrap = bootstrap
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_leaf_nodes = max_leaf_nodes
        self.sparse_output = sparse_output

        super().__init__(random_state=random_state)

    def fit(self, X: Dict[str, Any], y: Any = None) -> BaseEstimator:
        if self.max_leaf_nodes == "none":
            self.max_leaf_nodes = None
        if self.max_depth == "none":
            self.max_depth = None

        self.preprocessor['numerical'] = RandomTreesEmbedding(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            max_leaf_nodes=self.max_leaf_nodes,
            sparse_output=self.sparse_output,
            random_state=self.random_state
        )
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
        bootstrap: HyperparameterSearchSpace = HyperparameterSearchSpace(hyperparameter='bootstrap',
                                                                         value_range=(True, False),
                                                                         default_value=True,
                                                                         ),
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
        min_weight_fraction_leaf: HyperparameterSearchSpace = HyperparameterSearchSpace(
            hyperparameter='min_weight_fraction_leaf',
            value_range=(1.0,),
            default_value=1.0),
        max_leaf_nodes: HyperparameterSearchSpace = HyperparameterSearchSpace(hyperparameter='max_leaf_nodes',
                                                                              value_range=("none"),
                                                                              default_value="none",
                                                                              ),
    ) -> ConfigurationSpace:

        cs = ConfigurationSpace()
        add_hyperparameter(cs, bootstrap, CategoricalHyperparameter)
        add_hyperparameter(cs, n_estimators, UniformIntegerHyperparameter)
        add_hyperparameter(cs, max_depth, UniformIntegerHyperparameter)
        add_hyperparameter(cs, min_samples_split, UniformIntegerHyperparameter)
        add_hyperparameter(cs, min_samples_leaf, UniformIntegerHyperparameter)
        add_hyperparameter(cs, min_weight_fraction_leaf, UniformFloatHyperparameter)
        add_hyperparameter(cs, max_leaf_nodes, UniformIntegerHyperparameter)

        return cs
