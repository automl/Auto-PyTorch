import copy
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from sklearn.base import BaseEstimator

from torch import nn

from autoPyTorch.pipeline.components.setup.base_setup import autoPyTorchSetupComponent


class NetworkEmbeddingComponent(autoPyTorchSetupComponent):
    def __init__(self, random_state: Optional[np.random.RandomState] = None):
        super().__init__(random_state=random_state)
        self.embedding: Optional[nn.Module] = None
        self.feature_shapes: Dict[str, int] = {}

    def fit(self, X: Dict[str, Any], y: Any = None) -> BaseEstimator:

        num_numerical_columns, num_input_features = self._get_required_info_from_data(X)

        self.embedding, num_output_features = self.build_embedding(
            num_input_features=num_input_features,
            num_numerical_features=num_numerical_columns
        )
        if "feature_shapes" in X['dataset_properties']:
            if num_output_features is not None:
                feature_shapes = X['dataset_properties']['feature_shapes']
                # forecasting tasks
                feature_names = X['dataset_properties']['feature_names']
                for idx_cat, n_output_cat in enumerate(num_output_features[num_numerical_columns:]):
                    cat_feature_name = feature_names[idx_cat + num_numerical_columns]
                    feature_shapes[cat_feature_name] = n_output_cat
                self.feature_shapes = feature_shapes
            else:
                self.feature_shapes = X['dataset_properties']['feature_shapes']

        return self

    def transform(self, X: Dict[str, Any]) -> Dict[str, Any]:
        X.update({'network_embedding': self.embedding})
        if "feature_shapes" in X['dataset_properties']:
            X['dataset_properties'].update({"feature_shapes": self.feature_shapes})
        return X

    def build_embedding(self,
                        num_input_features: np.ndarray,
                        num_numerical_features: int) -> Tuple[nn.Module, Optional[List[int]]]:
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
        # Feature preprocessors can alter numerical columns
        if len(X['dataset_properties']['numerical_columns']) == 0:
            num_numerical_columns = 0
        else:
            X_train = copy.deepcopy(X['backend'].load_datamanager().train_tensors[0][:2])

            numerical_column_transformer = X['tabular_transformer'].preprocessor. \
                named_transformers_['numerical_pipeline']
            num_numerical_columns = numerical_column_transformer.transform(
                X_train[:, X['dataset_properties']['numerical_columns']]).shape[1]

        num_cols = num_numerical_columns + len(X['dataset_properties']['categorical_columns'])
        num_input_feats = np.zeros(num_cols, dtype=np.int32)

        categories = X['dataset_properties']['categories']
        for idx, cats in enumerate(categories, start=num_numerical_columns):
            num_input_feats[idx] = len(cats)

        return num_numerical_columns, num_input_feats
