from typing import Any, Dict, List, Optional, Tuple


import numpy as np

from sklearn.base import BaseEstimator

from torch import nn

from autoPyTorch.pipeline.components.setup.base_setup import autoPyTorchSetupComponent
from autoPyTorch.utils.common import FitRequirement


class NetworkEmbeddingComponent(autoPyTorchSetupComponent):
    def __init__(self, random_state: Optional[np.random.RandomState] = None):
        super().__init__(random_state=random_state)
        self.add_fit_requirements([
            FitRequirement('num_categories_per_col', (List,), user_defined=True, dataset_property=True),
            FitRequirement('shape_after_preprocessing', (Tuple[int],), user_defined=False, dataset_property=False)])

        self.embedding: Optional[nn.Module] = None

    def fit(self, X: Dict[str, Any], y: Any = None) -> BaseEstimator:

        num_features_excl_embed, num_categories_per_col = self._get_required_info_from_data(X)

        self.embedding = self.build_embedding(
            num_categories_per_col=num_categories_per_col,
            num_features_excl_embed=num_features_excl_embed)

        return self

    def transform(self, X: Dict[str, Any]) -> Dict[str, Any]:
        X.update({'network_embedding': self.embedding})
        return X

    def build_embedding(self, num_categories_per_col: np.ndarray, num_features_excl_embed: int) -> nn.Module:
        raise NotImplementedError

    def _get_required_info_from_data(self, X: Dict[str, Any]) -> Tuple[int, np.ndarray]:
        """
        Returns the number of numerical columns after preprocessing and
        an array of size equal to the number of input features
        containing zeros for numerical data and number of categories
        for categorical data. This is required to build the embedding.

        Args:
            X (Dict[str, Any]):
                Fit dictionary

        Returns:
            Tuple[int, np.ndarray]:
                number of numerical columns and array indicating
                number of categories for categorical columns and
                0 for numerical columns
        """
        num_cols = X['shape_after_preprocessing']
        # only works for 2D(rows, features) tabular data
        num_features_excl_embed = num_cols[0] - len(X['embed_columns'])

        num_categories_per_col = np.zeros(num_cols, dtype=np.int16)

        categories_per_embed_col = X['dataset_properties']['num_categories_per_col']

        # only fill num categories for embedding columns
        for idx, cats in enumerate(categories_per_embed_col, start=num_features_excl_embed):
            num_categories_per_col[idx] = cats

        return num_features_excl_embed, num_categories_per_col
