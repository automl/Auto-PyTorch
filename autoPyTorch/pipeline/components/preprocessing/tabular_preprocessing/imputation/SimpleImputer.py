from typing import Any, Dict, List, Optional, Union

from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import CategoricalHyperparameter

import numpy as np

from sklearn.impute import SimpleImputer as SklearnSimpleImputer

from autoPyTorch.datasets.base_dataset import BaseDatasetPropertiesType
from autoPyTorch.pipeline.components.preprocessing.tabular_preprocessing.imputation.base_imputer import BaseImputer
from autoPyTorch.utils.common import HyperparameterSearchSpace, add_hyperparameter


class SimpleImputer(BaseImputer):
    """
    An imputer for numerical columns

    Attributes:
        random_state (Optional[np.random.RandomState]):
            The random state to use for the imputer.
        numerical_strategy (str: default='mean'):
            The strategy to use for imputing numerical columns.
            Can be one of ['most_frequent', 'constant_!missing!']
    """

    def __init__(
        self,
        random_state: Optional[np.random.RandomState] = None,
        numerical_strategy: str = 'mean',
    ):
        super().__init__()
        self.random_state = random_state
        self.numerical_strategy = numerical_strategy

    def fit(self, X: Dict[str, Any], y: Optional[Any] = None) -> BaseImputer:
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
        self.check_requirements(X, y)

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
    ) -> ConfigurationSpace:
        """Get the hyperparameter search space for the SimpleImputer

        Args:
            dataset_properties (Optional[Dict[str, BaseDatasetPropertiesType]])
                Properties that describe the dataset
                Note: Not actually Optional, just adhering to its supertype
            numerical_strategy (HyperparameterSearchSpace: default = ...)
                The strategy to use for numerical imputation

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
