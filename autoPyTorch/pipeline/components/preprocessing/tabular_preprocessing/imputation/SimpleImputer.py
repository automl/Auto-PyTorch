from typing import Any, Dict, List, Optional, Union

from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import CategoricalHyperparameter

import numpy as np

from sklearn.impute import SimpleImputer as SklearnSimpleImputer

from autoPyTorch.datasets.base_dataset import BaseDatasetPropertiesType
from autoPyTorch.pipeline.components.preprocessing.tabular_preprocessing.imputation.base_imputer import BaseImputer
from autoPyTorch.utils.common import HyperparameterSearchSpace, add_hyperparameter


class SimpleImputer(BaseImputer):
    """An imputer for categorical and numerical columns

    Impute missing values for categorical columns with 'constant_!missing!'

    Note:
        In case of numpy data, the constant value is set to -1, under the assumption
        that categorical data is fit with an Ordinal Scaler.
    """

    def __init__(
        self,
        random_state: Optional[Union[np.random.RandomState, int]] = None,
        numerical_strategy: str = 'mean',
        categorical_strategy: str = 'most_frequent'
    ):
        """
        Note:
            Using 'constant' defaults to fill_value of 0 where 'constant_!missing!'
            uses a fill_value of -1. This behaviour should probably be fixed.

        Args:
            random_state (Optional[Union[np.random.RandomState, int]]):
                The random state to use for the imputer.
            numerical_strategy (str: default='mean'):
                The strategy to use for imputing numerical columns.
                Can be one of ['mean', 'median', 'most_frequent', 'constant', 'constant_!missing!']
            categorical_strategy (str: default='most_frequent')
                The strategy to use for imputing categorical columns.
                Can be one of ['mean', 'median', 'most_frequent', 'constant_zero']
        """
        super().__init__()
        self.random_state = random_state
        self.numerical_strategy = numerical_strategy
        self.categorical_strategy = categorical_strategy

    def fit(self, X: Dict[str, Any], y: Optional[Any] = None) -> BaseImputer:
        """ Fits the underlying model and returns the transformed array.

        Args:
            X (np.ndarray):
                The input features to fit on
            y (Optional[np.ndarray]):
                The labels for the input features `X`

        Returns:
            SimpleImputer:
                returns self
        """
        self.check_requirements(X, y)

        # Choose an imputer for any categorical columns
        categorical_columns = X['dataset_properties']['categorical_columns']

        if isinstance(categorical_columns, List) and len(categorical_columns) != 0:
            if self.categorical_strategy == 'constant_!missing!':
                # Train data is numpy as of this point, where an Ordinal Encoding is used
                # for categoricals. Only Numbers are allowed for `fill_value`
                imputer = SklearnSimpleImputer(strategy='constant', fill_value=-1, copy=False)
                self.preprocessor['categorical'] = imputer
            else:
                imputer = SklearnSimpleImputer(strategy=self.categorical_strategy, copy=False)
                self.preprocessor['categorical'] = imputer

        # Choose an imputer for any numerical columns
        numerical_columns = X['dataset_properties']['numerical_columns']

        if isinstance(numerical_columns, List) and len(numerical_columns) > 0:
            if self.numerical_strategy == 'constant_zero':
                imputer = SklearnSimpleImputer(strategy='constant', fill_value=0, copy=False)
                self.preprocessor['numerical'] = imputer
            else:
                imputer = SklearnSimpleImputer(strategy=self.numerical_strategy, copy=False)
                self.preprocessor['numerical'] = imputer

        return self

    @staticmethod
    def get_hyperparameter_search_space(
        dataset_properties: Optional[Dict[str, BaseDatasetPropertiesType]] = None,
        numerical_strategy: HyperparameterSearchSpace = HyperparameterSearchSpace(
            hyperparameter='numerical_strategy',
            value_range=("mean", "median", "most_frequent", "constant_zero"),
            default_value="mean",
        ),
        categorical_strategy: HyperparameterSearchSpace = HyperparameterSearchSpace(
            hyperparameter='categorical_strategy',
            value_range=("most_frequent", "constant_!missing!"),
            default_value="most_frequent"
        )
    ) -> ConfigurationSpace:
        """Get the hyperparameter search space for the SimpleImputer

        Args:
            dataset_properties (Optional[Dict[str, BaseDatasetPropertiesType]])
                Properties that describe the dataset
                Note: Not actually Optional, just adhering to its supertype
            numerical_strategy (HyperparameterSearchSpace: default = ...)
                The strategy to use for numerical imputation
            caterogical_strategy (HyperparameterSearchSpace: default = ...)
                The strategy to use for categorical imputation

        Returns:
            ConfigurationSpace
                The space of possible configurations for a SimpleImputer with the given
                `dataset_properties`
        """
        cs = ConfigurationSpace()

        if dataset_properties is None:
            raise ValueError("SimpleImputer requires `dataset_properties` for generating"
                             " a search space.")

        if (
            isinstance(dataset_properties['numerical_columns'], List)
            and len(dataset_properties['numerical_columns']) != 0
        ):
            add_hyperparameter(cs, numerical_strategy, CategoricalHyperparameter)

        if (
            isinstance(dataset_properties['categorical_columns'], List)
            and len(dataset_properties['categorical_columns'])
        ):
            add_hyperparameter(cs, categorical_strategy, CategoricalHyperparameter)

        return cs

    @staticmethod
    def get_properties(
        dataset_properties: Optional[Dict[str, BaseDatasetPropertiesType]] = None
    ) -> Dict[str, Union[str, bool]]:
        """Get the properties of the SimpleImputer class and what it can handle

        Returns:
            Dict[str, Union[str, bool]]:
                A dict from property names to values
        """
        return {
            'shortname': 'SimpleImputer',
            'name': 'Simple Imputer',
            'handles_sparse': True
        }
