from math import ceil, floor
from typing import Any, Callable, Dict, List, Optional, Union

from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.forbidden import ForbiddenInClause, \
    ForbiddenAndConjunction, ForbiddenEqualsClause
from ConfigSpace.hyperparameters import (
    CategoricalHyperparameter,
    UniformFloatHyperparameter,
    UniformIntegerHyperparameter,
)

import numpy as np

from sklearn.cluster import FeatureAgglomeration
from sklearn.base import BaseEstimator

from autoPyTorch.datasets.base_dataset import BaseDatasetPropertiesType
from autoPyTorch.pipeline.components.preprocessing.tabular_preprocessing.feature_preprocessing. \
    base_feature_preprocessor import autoPyTorchFeaturePreprocessingComponent
from autoPyTorch.utils.common import HyperparameterSearchSpace, add_hyperparameter, get_hyperparameter


class FeatureAgglomeration(autoPyTorchFeaturePreprocessingComponent):
    def __init__(self, n_clusters: int = 25,
                 affinity: str = 'euclidean', linkage: str = 'ward',
                 pooling_func: str = "max",
                 random_state: Optional[np.random.RandomState] = None
                 ):
        self.n_clusters = n_clusters
        self.affinity = affinity
        self.linkage = linkage
        self.pooling_func: Union[str, Callable] = pooling_func
        self.pooling_func_mapping: Dict[str, Callable] = dict(mean=np.mean,
                                                              median=np.median,
                                                              max=np.max)

        super().__init__(random_state=random_state)

    def fit(self, X: Dict[str, Any], y: Any = None) -> BaseEstimator:
        if not callable(self.pooling_func):
            self.pooling_func = self.pooling_func_mapping[self.pooling_func]

        self.preprocessor['numerical'] = FeatureAgglomeration(
            n_clusters=self.n_clusters, affinity=self.affinity,
            linkage=self.linkage, pooling_func=self.pooling_func)

        return self

    @staticmethod
    def get_hyperparameter_search_space(
        dataset_properties: Optional[Dict[str, BaseDatasetPropertiesType]] = None,
        n_components: HyperparameterSearchSpace = HyperparameterSearchSpace(hyperparameter='n_components',
                                                                            value_range=(0.5, 0.9),
                                                                            default_value=0.5,
                                                                            ),
        affinity: HyperparameterSearchSpace = HyperparameterSearchSpace(hyperparameter='affinity',
                                                                        value_range=("euclidean", "manhattan", "cosine"),
                                                                        default_value="euclidean",
                                                                        ),
        linkage: HyperparameterSearchSpace = HyperparameterSearchSpace(hyperparameter='linkage',
                                                                       value_range=("ward", "complete", "average"),
                                                                       default_value="ward",
                                                                       ),
        pooling_func: HyperparameterSearchSpace = HyperparameterSearchSpace(hyperparameter='pooling_func',
                                                                            value_range=("mean", "median", "max"),
                                                                            default_value="max",
                                                                            ),
    ) -> ConfigurationSpace:
        if dataset_properties is not None:
            n_features = len(dataset_properties['numerical_columns']) if isinstance(
                dataset_properties['numerical_columns'], List) else 0
            if n_features == 1:
                log = False
            else:
                log = n_components.log
            n_components = HyperparameterSearchSpace(hyperparameter='n_components',
                                                     value_range=(
                                                         floor(float(n_components.value_range[0]) * n_features),
                                                         ceil(float(n_components.value_range[1]) * n_features)),
                                                     default_value=ceil(float(n_components.default_value) * n_features),
                                                     log=log)
        else:
            n_components = HyperparameterSearchSpace(hyperparameter='n_components',
                                                     value_range=(10, 2000),
                                                     default_value=100,
                                                     log=n_components.log)
        cs = ConfigurationSpace()

        add_hyperparameter(cs, n_components, UniformIntegerHyperparameter)
        affinity_hp = get_hyperparameter(affinity, CategoricalHyperparameter)
        linkage_hp = get_hyperparameter(linkage, CategoricalHyperparameter)
        add_hyperparameter(cs, pooling_func, CategoricalHyperparameter)
        cs.add_hyperparameters([affinity_hp, linkage_hp])

        affinity_choices = []
        if "manhattan" in affinity_hp.choices:
            affinity_choices.append("manhattan")
        if "cosine" in affinity_hp.choices:
            affinity_choices.append("cosine")
        
        if "ward" in linkage_hp.choices and len(affinity_choices) > 0:
            forbidden_condition = ForbiddenAndConjunction(
                ForbiddenInClause(affinity_hp, affinity_choices),
                ForbiddenEqualsClause(linkage_hp, "ward")
            )
            cs.add_forbidden_clause(forbidden_condition)

        return cs


    @staticmethod
    def get_properties(dataset_properties: Optional[Dict[str, BaseDatasetPropertiesType]] = None) -> Dict[str, Any]:
        return {'shortname': 'PCA',
                'name': 'Principal Component Analysis',
                'handles_sparse': False,
                'handles_classification': True,
                'handles_regression': True
                }
