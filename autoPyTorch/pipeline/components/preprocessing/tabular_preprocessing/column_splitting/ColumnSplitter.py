from typing import Any, Dict, List, Optional, Union

from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import (
    UniformIntegerHyperparameter,
)

import numpy as np

from autoPyTorch.constants import MIN_CATEGORIES_FOR_EMBEDDING_MAX
from autoPyTorch.datasets.base_dataset import BaseDatasetPropertiesType
from autoPyTorch.pipeline.components.preprocessing.tabular_preprocessing.base_tabular_preprocessing import \
    autoPyTorchTabularPreprocessingComponent
from autoPyTorch.utils.common import HyperparameterSearchSpace, add_hyperparameter


class ColumnSplitter(autoPyTorchTabularPreprocessingComponent):
    """
    Splits categorical columns into embed or encode columns based on a hyperparameter.
    """
    def __init__(
        self,
        min_categories_for_embedding: float = 5,
        random_state: Optional[np.random.RandomState] = None
    ):
        self.min_categories_for_embedding = min_categories_for_embedding
        self.random_state = random_state

        self.special_feature_types: Dict[str, List] = dict(encode_columns=[], embed_columns=[])
        self.num_categories_per_col: Optional[List] = None
        super().__init__()

    def fit(self, X: Dict[str, Any], y: Optional[Any] = None) -> 'ColumnSplitter':

        self.check_requirements(X, y)

        if len(X['dataset_properties']['categorical_columns']) > 0:
            self.num_categories_per_col = []
            for categories_per_column, column in zip(X['dataset_properties']['num_categories_per_col'],
                                                     X['dataset_properties']['categorical_columns']):
                if (
                    categories_per_column >= self.min_categories_for_embedding
                ):
                    self.special_feature_types['embed_columns'].append(column)
                    # we only care about the categories for columns to be embedded
                    self.num_categories_per_col.append(categories_per_column)
                else:
                    self.special_feature_types['encode_columns'].append(column)

        return self

    def transform(self, X: Dict[str, Any]) -> Dict[str, Any]:
        if self.num_categories_per_col is not None:
            # update such that only n categories for embedding columns is passed
            X['dataset_properties']['num_categories_per_col'] = self.num_categories_per_col
        X.update(self.special_feature_types)
        return X

    @staticmethod
    def get_properties(
        dataset_properties: Optional[Dict[str, BaseDatasetPropertiesType]] = None
    ) -> Dict[str, Union[str, bool]]:

        return {
            'shortname': 'ColumnSplitter',
            'name': 'Column Splitter',
            'handles_sparse': False,
        }

    @staticmethod
    def get_hyperparameter_search_space(
        dataset_properties: Optional[Dict[str, BaseDatasetPropertiesType]] = None,
        min_categories_for_embedding: HyperparameterSearchSpace = HyperparameterSearchSpace(
            hyperparameter="min_categories_for_embedding",
            value_range=(3, MIN_CATEGORIES_FOR_EMBEDDING_MAX),
            default_value=3,
            log=True),
    ) -> ConfigurationSpace:
        cs = ConfigurationSpace()

        add_hyperparameter(cs, min_categories_for_embedding, UniformIntegerHyperparameter)

        return cs
