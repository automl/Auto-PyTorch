from typing import Any, Dict, List, Optional, Union

from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import (
    CategoricalHyperparameter,
    UniformFloatHyperparameter,
    UniformIntegerHyperparameter,
)

import numpy as np

from sklearn.base import BaseEstimator
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.feature_selection import SelectFromModel

from autoPyTorch.datasets.base_dataset import BaseDatasetPropertiesType
from autoPyTorch.pipeline.components.preprocessing.tabular_preprocessing.feature_preprocessing. \
    base_feature_preprocessor import autoPyTorchFeaturePreprocessingComponent
from autoPyTorch.pipeline.components.preprocessing.tabular_preprocessing.feature_preprocessing. \
    utils import NoneType_
from autoPyTorch.utils.common import FitRequirement, HyperparameterSearchSpace, add_hyperparameter, check_none


CRITERION_CHOICES = ('mse', 'friedman_mse', 'mae')


class ExtraTreesPreprocessorRegression(autoPyTorchFeaturePreprocessingComponent):
    """
    Selects features based on importance weights using extra trees
    """
    def __init__(self, bootstrap: bool = True, n_estimators: int = 10,
                 criterion: str = "mse", max_features: float = 1,
                 max_depth: Union[int, NoneType_] = 5, min_samples_split: int = 2,
                 min_samples_leaf: int = 1, min_weight_fraction_leaf: float = 0,
                 max_leaf_nodes: Union[int, NoneType_] = "none",
                 oob_score: bool = False, verbose: int = 0,
                 random_state: Optional[np.random.RandomState] = None):
        self.bootstrap = bootstrap
        self.n_estimators = n_estimators
        if criterion not in CRITERION_CHOICES:
            raise ValueError(f"`criterion` of {self.__class__.__name__} "
                             f"must be in {CRITERION_CHOICES}, but got: {criterion}")
        self.criterion = criterion
        self.max_features = max_features
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_leaf_nodes = max_leaf_nodes
        self.oob_score = oob_score
        self.verbose = verbose

        super().__init__(random_state=random_state)

        self.add_fit_requirements([
            FitRequirement('numerical_columns', (List,), user_defined=True, dataset_property=True)])

    def get_components_kwargs(self) -> Dict[str, Any]:
        """
        returns keyword arguments required by the feature preprocessor

        Returns:
            Dict[str, Any]: kwargs
        """
        return dict(
            bootstrap=self.bootstrap,
            n_estimators=self.n_estimators,
            criterion=self.criterion,
            max_features=self.max_features,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            min_weight_fraction_leaf=self.min_weight_fraction_leaf,
            max_leaf_nodes=self.max_leaf_nodes,
            oob_score=self.oob_score,
            verbose=self.verbose,
            random_state=self.random_state,
        )

    def fit(self, X: Dict[str, Any], y: Any = None) -> BaseEstimator:

        self.check_requirements(X, y)

        if check_none(self.max_leaf_nodes):
            self.max_leaf_nodes = None
        elif isinstance(self.max_leaf_nodes, int):
            self.max_leaf_nodes = int(self.max_leaf_nodes)
        else:
            raise ValueError(f"Expected `max_leaf_nodes` to be either "
                             f"in ('None', 'none', None) or an integer, got {self.max_leaf_nodes}")

        if check_none(self.max_depth):
            self.max_depth = None
        elif isinstance(self.max_depth, int):
            self.max_depth = int(self.max_depth)
        else:
            raise ValueError(f"Expected `max_depth` to be either "
                             f"in ('None', 'none', None) or an integer, got {self.max_depth}")

        num_features = len(X['dataset_properties']['numerical_columns'])
        max_features = int(
            float(self.max_features) * (np.log(num_features) + 1))
        # Use at most half of the features
        max_features = max(1, min(int(num_features / 2), max_features))

        # TODO: add class_weights
        estimator = ExtraTreesRegressor(**self.get_components_kwargs())

        self.preprocessor['numerical'] = SelectFromModel(estimator=estimator,
                                                         threshold='mean',
                                                         prefit=False)
        return self

    @staticmethod
    def get_hyperparameter_search_space(
        dataset_properties: Optional[Dict[str, BaseDatasetPropertiesType]] = None,
        bootstrap: HyperparameterSearchSpace = HyperparameterSearchSpace(hyperparameter='bootstrap',
                                                                         value_range=(True, False),
                                                                         default_value=True,
                                                                         ),
        n_estimators: HyperparameterSearchSpace = HyperparameterSearchSpace(hyperparameter='n_estimators',
                                                                            value_range=(100,),
                                                                            default_value=100,
                                                                            ),
        max_depth: HyperparameterSearchSpace = HyperparameterSearchSpace(hyperparameter='max_depth',
                                                                         value_range=("none",),
                                                                         default_value="none",
                                                                         ),
        max_features: HyperparameterSearchSpace = HyperparameterSearchSpace(hyperparameter='max_features',
                                                                            value_range=(0.1, 1),
                                                                            default_value=1,
                                                                            ),
        criterion: HyperparameterSearchSpace = HyperparameterSearchSpace(hyperparameter='criterion',
                                                                         value_range=CRITERION_CHOICES,
                                                                         default_value="mse",
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
            value_range=(0,),
            default_value=0),
        max_leaf_nodes: HyperparameterSearchSpace = HyperparameterSearchSpace(hyperparameter='max_leaf_nodes',
                                                                              value_range=("none",),
                                                                              default_value="none",
                                                                              ),
    ) -> ConfigurationSpace:

        cs = ConfigurationSpace()
        add_hyperparameter(cs, bootstrap, CategoricalHyperparameter)
        add_hyperparameter(cs, n_estimators, UniformIntegerHyperparameter)
        add_hyperparameter(cs, max_features, UniformFloatHyperparameter)
        add_hyperparameter(cs, criterion, CategoricalHyperparameter)
        add_hyperparameter(cs, max_depth, UniformIntegerHyperparameter)
        add_hyperparameter(cs, min_samples_split, UniformIntegerHyperparameter)
        add_hyperparameter(cs, min_samples_leaf, UniformIntegerHyperparameter)
        add_hyperparameter(cs, min_weight_fraction_leaf, UniformFloatHyperparameter)
        add_hyperparameter(cs, max_leaf_nodes, UniformIntegerHyperparameter)

        return cs

    @staticmethod
    def get_properties(dataset_properties: Optional[Dict[str, BaseDatasetPropertiesType]] = None) -> Dict[str, Any]:
        return {'shortname': 'ETR',
                'name': 'Extra Trees Regressor Preprocessing',
                'handles_sparse': True,
                'handles_regression': True,
                'handles_classification': False
                }
