from typing import Any, Callable, Dict, Optional

from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.forbidden import (
    ForbiddenAndConjunction,
    ForbiddenEqualsClause,
    ForbiddenInClause
)
from ConfigSpace.hyperparameters import (
    CategoricalHyperparameter,
    UniformIntegerHyperparameter,
)

import numpy as np

from sklearn.base import BaseEstimator
from sklearn.cluster import FeatureAgglomeration as SklearnFeatureAgglomeration

from autoPyTorch.datasets.base_dataset import BaseDatasetPropertiesType
from autoPyTorch.pipeline.components.preprocessing.tabular_preprocessing.feature_preprocessing. \
    base_feature_preprocessor import autoPyTorchFeaturePreprocessingComponent
from autoPyTorch.pipeline.components.preprocessing.tabular_preprocessing.feature_preprocessing. \
    utils import percentage_value_range_to_integer_range
from autoPyTorch.utils.common import HyperparameterSearchSpace, add_hyperparameter, get_hyperparameter


class FeatureAgglomeration(autoPyTorchFeaturePreprocessingComponent):
    """
    Recursively merge pair of clusters of features constructed
    using agglomerative clustering.

    Args:
            n_clusters (int):
                The number of clusters to find. Defaults to 25.
                Note:
                    This number needs to be less than the total number of
                    features. To keep the hyperparameter search space general
                    to different datasets, autoPyTorch defines its value
                    range as the percentage of the number of features (in float).
                    This is then used to construct the range of n_clusters using
                    n_clusters = percentage of features * number of features.
            affinity (str):
                Metric used to compute the linkage. If linkage is “ward”, only
                “euclidean” is accepted. Defaults to 'euclidean'.
            linkage (str):
                Which linkage criterion to use. The linkage criterion determines
                which distance to use between sets of features. Defaults to 'ward'.
            pooling_func (str):
                Combines the values of agglomerated features into a single value,
                autoPyTorch uses (max, min and median) functions from numpy. Defaults to "max".
    """
    def __init__(self, n_clusters: int = 25,
                 affinity: str = 'euclidean', linkage: str = 'ward',
                 pooling_func: str = "max",
                 random_state: Optional[np.random.RandomState] = None
                 ):
        self.n_clusters = n_clusters
        self.affinity = affinity
        self.linkage = linkage
        self.pooling_func: Callable = getattr(np, pooling_func)

        super().__init__(random_state=random_state)

    def fit(self, X: Dict[str, Any], y: Any = None) -> BaseEstimator:

        self.check_requirements(X, y)

        self.preprocessor['numerical'] = SklearnFeatureAgglomeration(
            n_clusters=self.n_clusters, affinity=self.affinity,
            linkage=self.linkage, pooling_func=self.pooling_func)

        return self

    @staticmethod
    def get_hyperparameter_search_space(
        dataset_properties: Optional[Dict[str, BaseDatasetPropertiesType]] = None,
        n_clusters: HyperparameterSearchSpace = HyperparameterSearchSpace(hyperparameter='n_clusters',
                                                                          value_range=(0.5, 0.9),
                                                                          default_value=0.5,
                                                                          ),
        affinity: HyperparameterSearchSpace = HyperparameterSearchSpace(hyperparameter='affinity',
                                                                        value_range=("euclidean",
                                                                                     "manhattan",
                                                                                     "cosine"),
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
        n_clusters = percentage_value_range_to_integer_range(
            hyperparameter_search_space=n_clusters,
            default_value_range=(2, 400),
            default_value=25,
            dataset_properties=dataset_properties,
        )
        cs = ConfigurationSpace()

        add_hyperparameter(cs, n_clusters, UniformIntegerHyperparameter)
        affinity_hp = get_hyperparameter(affinity, CategoricalHyperparameter)
        linkage_hp = get_hyperparameter(linkage, CategoricalHyperparameter)
        add_hyperparameter(cs, pooling_func, CategoricalHyperparameter)
        cs.add_hyperparameters([affinity_hp, linkage_hp])

        # If linkage is “ward”, only “euclidean” is accepted.
        non_euclidian_affinity = [choice for choice in ["manhattan", "cosine"] if choice in affinity_hp.choices]

        if "ward" in linkage_hp.choices and len(non_euclidian_affinity) > 0:
            forbidden_condition = ForbiddenAndConjunction(
                ForbiddenInClause(affinity_hp, non_euclidian_affinity),
                ForbiddenEqualsClause(linkage_hp, "ward")
            )
            cs.add_forbidden_clause(forbidden_condition)

        return cs

    @staticmethod
    def get_properties(dataset_properties: Optional[Dict[str, BaseDatasetPropertiesType]] = None) -> Dict[str, Any]:
        return {'shortname': 'FeatureAgglomeration',
                'name': 'Feature Agglomeration',
                'handles_sparse': False,
                'handles_classification': True,
                'handles_regression': True
                }
