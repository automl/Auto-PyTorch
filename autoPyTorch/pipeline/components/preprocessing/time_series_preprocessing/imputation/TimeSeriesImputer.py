from typing import Any, Dict, List, Optional

import numpy as np
from sktime.transformations.series.impute import Imputer

from ConfigSpace import ConfigurationSpace
from autoPyTorch.utils.common import FitRequirement

from autoPyTorch.pipeline.components.preprocessing.time_series_preprocessing.imputation. \
    base_time_series_imputer import BaseTimeSeriesImputer

from autoPyTorch.pipeline.components.preprocessing.time_series_preprocessing.base_time_series_preprocessing import (
    autoPyTorchTimeSeriesPreprocessingComponent,
    autoPyTorchTimeSeriesTargetPreprocessingComponent
)
from autoPyTorch.datasets.base_dataset import BaseDatasetPropertiesType
from autoPyTorch.utils.common import HyperparameterSearchSpace


class TimeSeriesFeatureImputer(BaseTimeSeriesImputer, autoPyTorchTimeSeriesPreprocessingComponent):
    def __init__(self,
                 random_state: Optional[np.random.RandomState] = None,
                 imputation_strategy: str = 'mean'):
        super().__init__()
        self.random_state = random_state
        self.imputation_strategy = imputation_strategy
        self.add_fit_requirements([
            FitRequirement('numerical_columns', (List,), user_defined=True, dataset_property=True)])

    def fit(self, X: Dict[str, Any], y: Optional[Any] = None) -> BaseTimeSeriesImputer:
        """
        Builds the preprocessor based on the given fit dictionary 'X'.

        Args:
            X (Dict[str, Any]):
                The fit dictionary
            y (Optional[Any]):
                Not Used -- to comply with API

        Returns:
            self:
                returns an instance of self.
        """
        # Choose an imputer for any numerical columns
        numerical_columns = X['dataset_properties']['numerical_columns']

        if isinstance(numerical_columns, List) and len(numerical_columns) > 0:
            if self.imputation_strategy == 'constant_zero':
                imputer = Imputer(method='constant', random_state=self.random_state, value=0)
                self.preprocessor['numerical'] = imputer
            else:
                imputer = Imputer(method=self.imputation_strategy, random_state=self.random_state)
                self.preprocessor['numerical'] = imputer

        return self

    def transform(self, X: Dict[str, Any]) -> Dict[str, Any]:
        """
        Adds self into the 'X' dictionary and returns it.
        Args:
            X (Dict[str, Any]): 'X' dictionary

        Returns:
            (Dict[str, Any]): the updated 'X' dictionary
        """
        if self.preprocessor['numerical'] is None and len(X["dataset_properties"]["numerical_columns"]) != 0:
            raise ValueError("cant call transform on {} without fitting first."
                             .format(self.__class__.__name__))
        X.update({'imputer': self.preprocessor})
        return X

    @staticmethod
    def get_hyperparameter_search_space(
            dataset_properties: Optional[Dict[str, BaseDatasetPropertiesType]] = None,
            imputation_strategy: HyperparameterSearchSpace = HyperparameterSearchSpace(
                hyperparameter='imputation_strategy',
                value_range=("drift", "linear", "nearest", "constant_zero", "mean", "median", "bfill", "ffill"),
                default_value="drift",
            ),
    ) -> ConfigurationSpace:
        if dataset_properties.get('features_have_missing_values', False):
            cs = super().get_hyperparameter_search_space(dataset_properties, imputation_strategy)
        else:
            cs = ConfigurationSpace()
        return cs


class TimeSeriesTargetImputer(BaseTimeSeriesImputer, autoPyTorchTimeSeriesTargetPreprocessingComponent):
    def __init__(self,
                 random_state: Optional[np.random.RandomState] = None,
                 imputation_strategy: str = 'mean', ):
        super().__init__()
        self.random_state = random_state
        self.imputation_strategy = imputation_strategy
        self.add_fit_requirements([
            FitRequirement('numerical_columns', (List,), user_defined=True, dataset_property=True)])

    def fit(self, X: Dict[str, Any], y: Optional[Any] = None) -> BaseTimeSeriesImputer:
        """
        Builds the preprocessor based on the given fit dictionary 'X'.

        Args:
            X (Dict[str, Any]):
                The fit dictionary
            y (Optional[Any]):
                Not Used -- to comply with API

        Returns:
            self:
                returns an instance of self.
        """
        # Forecasting tasks always have numerical outputs (TODO add support for categorical HPs)
        if self.imputation_strategy == 'constant_zero':
            imputer = Imputer(method='constant', random_state=self.random_state, value=0)
            self.preprocessor['target_numerical'] = imputer
        else:
            imputer = Imputer(method=self.imputation_strategy, random_state=self.random_state)
            self.preprocessor['target_numerical'] = imputer

        return self

    def transform(self, X: Dict[str, Any]) -> Dict[str, Any]:
        """
        Adds self into the 'X' dictionary and returns it.
        Args:
            X (Dict[str, Any]): 'X' dictionary

        Returns:
            (Dict[str, Any]): the updated 'X' dictionary
        """
        if self.preprocessor['target_numerical'] is None and len(X["dataset_properties"]["numerical_columns"]) != 0:
            raise ValueError("cant call transform on {} without fitting first."
                             .format(self.__class__.__name__))
        X.update({'imputer': self.preprocessor})
        return X

    @staticmethod
    def get_hyperparameter_search_space(
            dataset_properties: Optional[Dict[str, BaseDatasetPropertiesType]] = None,
            imputation_strategy: HyperparameterSearchSpace = HyperparameterSearchSpace(
                hyperparameter='imputation_strategy',
                value_range=("linear", "nearest", "constant_zero", "bfill", "ffill"),
                default_value="linear",
            ),
    ) -> ConfigurationSpace:
        """
        Time series imputor, for the sake of speed, we only allow local imputation here (i.e., the filled value only
        depends on its neighbours)
        # TODO: Transformer for mean and median: df.fillna(df.groupby(df.index).agg('mean'))...
        Args:
            dataset_properties (Optional[Dict[str, BaseDatasetPropertiesType]]): dataset properties
            imputation_strategy: which strategy to use, its content is defined by
             sktime.transformations.series.impute.Imputer


        Returns:

        """
        if dataset_properties.get('features_have_missing_values', False):
            cs = super().get_hyperparameter_search_space(dataset_properties, imputation_strategy)
        else:
            cs = ConfigurationSpace()
        return cs
