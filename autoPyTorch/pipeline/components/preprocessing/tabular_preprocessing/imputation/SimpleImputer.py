from typing import Any, Dict, List, Optional, Union

from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import (
    CategoricalHyperparameter
)

import numpy as np

from sklearn.impute import SimpleImputer as SklearnSimpleImputer

from autoPyTorch.datasets.base_dataset import BaseDatasetPropertiesType
from autoPyTorch.pipeline.components.preprocessing.tabular_preprocessing.imputation.base_imputer import BaseImputer
from autoPyTorch.utils.common import HyperparameterSearchSpace, add_hyperparameter


class SimpleImputer(BaseImputer):
    """
    Impute missing values for categorical columns with '!missing!'
    (In case of numpy data, the constant value is set to -1, under
    the assumption that categorical data is fit with an Ordinal Scaler)
    """

    def __init__(self,
                 random_state: Optional[Union[np.random.RandomState, int]] = None,
                 numerical_strategy: str = 'mean',
                 categorical_strategy: str = 'most_frequent'):
        super().__init__()
        self.random_state = random_state
        self.numerical_strategy = numerical_strategy
        self.categorical_strategy = categorical_strategy

    def fit(self, X: Dict[str, Any], y: Any = None) -> BaseImputer:
        """
        The fit function calls the fit function of the underlying model
        and returns the transformed array.
        Args:
            X (np.ndarray): input features
            y (Optional[np.ndarray]): input labels

        Returns:
            instance of self
        """
        self.check_requirements(X, y)
        categorical_columns = X['dataset_properties']['categorical_columns'] \
            if isinstance(X['dataset_properties']['categorical_columns'], List) else []
        if len(categorical_columns) != 0:
            if self.categorical_strategy == 'constant_!missing!':
                self.preprocessor['categorical'] = SklearnSimpleImputer(strategy='constant',
                                                                        # Train data is numpy
                                                                        # as of this point, where
                                                                        # Ordinal Encoding is using
                                                                        # for categorical. Only
                                                                        # Numbers are allowed
                                                                        # fill_value='!missing!',
                                                                        fill_value=-1,
                                                                        copy=False)
            else:
                self.preprocessor['categorical'] = SklearnSimpleImputer(strategy=self.categorical_strategy,
                                                                        copy=False)
        numerical_columns = X['dataset_properties']['numerical_columns'] \
            if isinstance(X['dataset_properties']['numerical_columns'], List) else []
        if len(numerical_columns) != 0:
            if self.numerical_strategy == 'constant_zero':
                self.preprocessor['numerical'] = SklearnSimpleImputer(strategy='constant',
                                                                      fill_value=0,
                                                                      copy=False)
            else:
                self.preprocessor['numerical'] = SklearnSimpleImputer(strategy=self.numerical_strategy, copy=False)

        return self

    @staticmethod
    def get_hyperparameter_search_space(
        dataset_properties: Optional[Dict[str, BaseDatasetPropertiesType]] = None,
        numerical_strategy: HyperparameterSearchSpace = HyperparameterSearchSpace(hyperparameter='numerical_strategy',
                                                                                  value_range=("mean", "median",
                                                                                               "most_frequent",
                                                                                               "constant_zero"),
                                                                                  default_value="mean",
                                                                                  ),
        categorical_strategy: HyperparameterSearchSpace = HyperparameterSearchSpace(
            hyperparameter='categorical_strategy',
            value_range=("most_frequent",
                         "constant_!missing!"),
            default_value="most_frequent")
    ) -> ConfigurationSpace:
        cs = ConfigurationSpace()
        assert dataset_properties is not None, "To create hyperparameter search space" \
                                               ", dataset_properties should not be None"
        if len(dataset_properties['numerical_columns']) \
                if isinstance(dataset_properties['numerical_columns'], List) else 0 != 0:
            add_hyperparameter(cs, numerical_strategy, CategoricalHyperparameter)

        if len(dataset_properties['categorical_columns']) \
                if isinstance(dataset_properties['categorical_columns'], List) else 0 != 0:
            add_hyperparameter(cs, categorical_strategy, CategoricalHyperparameter)

        return cs

    @staticmethod
    def get_properties(dataset_properties: Optional[Dict[str, BaseDatasetPropertiesType]] = None
                       ) -> Dict[str, Union[str, bool]]:
        return {
            'shortname': 'SimpleImputer',
            'name': 'Simple Imputer',
            'handles_sparse': True
        }
