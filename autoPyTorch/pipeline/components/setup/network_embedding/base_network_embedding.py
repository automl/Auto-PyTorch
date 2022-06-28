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

        num_numerical_columns, num_input_features = self._get_args(X)

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

    def _get_args(self, X: Dict[str, Any]) -> Tuple[int, np.ndarray]:
        # Feature preprocessors can alter numerical columns
        if len(X['dataset_properties']['numerical_columns']) == 0:
            num_numerical_columns = 0
        else:
            X_train = copy.deepcopy(X['backend'].load_datamanager().train_tensors[0][:2])

            if 'tabular_transformer' in X:
                numerical_column_transformer = X['tabular_transformer'].preprocessor. \
                    named_transformers_['numerical_pipeline']
            elif 'time_series_feature_transformer' in X:
                numerical_column_transformer = X['time_series_feature_transformer'].preprocessor. \
                    named_transformers_['numerical_pipeline']
            else:
                raise ValueError("Either a tabular or time_series transformer must be contained!")
            if hasattr(X_train, 'iloc'):
                num_numerical_columns = numerical_column_transformer.transform(
                    X_train.iloc[:, X['dataset_properties']['numerical_columns']]).shape[1]
            else:
                num_numerical_columns = numerical_column_transformer.transform(
                    X_train[:, X['dataset_properties']['numerical_columns']]).shape[1]
        num_input_features = np.zeros((num_numerical_columns + len(X['dataset_properties']['categorical_columns'])),
                                      dtype=np.int32)
        categories = X['dataset_properties']['categories']

        for i, category in enumerate(categories):
            num_input_features[num_numerical_columns + i, ] = len(category)
        return num_numerical_columns, num_input_features
