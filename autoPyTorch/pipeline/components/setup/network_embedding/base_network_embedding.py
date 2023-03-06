import copy
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from sklearn.base import BaseEstimator

from torch import nn

from autoPyTorch.pipeline.components.setup.base_setup import autoPyTorchSetupComponent


class NetworkEmbeddingComponent(autoPyTorchSetupComponent):
    def __init__(self, random_state: Optional[Union[np.random.RandomState, int]] = None):
        super().__init__()
        self.embedding: Optional[nn.Module] = None
        self.random_state = random_state
        self.feature_shapes: Dict[str, int] = {}

    def fit(self, X: Dict[str, Any], y: Any = None) -> BaseEstimator:

        num_numerical_columns, num_categories_per_col = self._get_args(X)

        self.embedding, num_output_features = self.build_embedding(
            num_categories_per_col=num_categories_per_col,
            num_numerical_features=num_numerical_columns
        )
        return self

    def transform(self, X: Dict[str, Any]) -> Dict[str, Any]:
        X.update({'network_embedding': self.embedding})
        return X

    def build_embedding(self,
                        num_categories_per_col: np.ndarray,
                        num_numerical_features: int) -> Tuple[nn.Module, Optional[List[int]]]:
        raise NotImplementedError

    def _get_args(self, X: Dict[str, Any]) -> Tuple[int, np.ndarray]:
        # Feature preprocessors can alter numerical columns
        if len(X['dataset_properties']['numerical_columns']) == 0:
            num_numerical_columns = 0
        else:
            num_numerical_columns = len(X['dataset_properties']['numerical_columns'])
            
        num_cols = num_numerical_columns + len(X['dataset_properties']['categorical_columns'])
        num_categories_per_col = np.zeros(num_cols, dtype=np.int32)

        categories = X['dataset_properties']['num_categories_per_col']
        for idx, cats in enumerate(categories):
            num_categories_per_col[idx] = cats

        return num_numerical_columns, num_categories_per_col