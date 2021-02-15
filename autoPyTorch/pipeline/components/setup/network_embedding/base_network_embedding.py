import copy
from typing import Any, Dict, Optional, Union

import numpy as np

from sklearn.base import BaseEstimator

from torch import nn

from autoPyTorch.pipeline.components.setup.base_setup import autoPyTorchSetupComponent
from autoPyTorch.utils.common import subsampler

class NetworkEmbeddingComponent(autoPyTorchSetupComponent):
    def __init__(self, random_state: Optional[Union[np.random.RandomState, int]] = None):
        super().__init__()
        self.embedding: Optional[nn.Module] = None

    def fit(self, X: Dict[str, Any], y: Any = None) -> BaseEstimator:

        num_numerical_columns, num_input_features = self._get_args(X)

        self.embedding = self.build_embedding(
            num_input_features=num_input_features,
            num_numerical_features=num_numerical_columns)
        return self

    def transform(self, X: Dict[str, Any]) -> Dict[str, Any]:
        X.update({'network_embedding': self.embedding})
        return X

    def build_embedding(self, num_input_features, num_numerical_features) -> nn.Module:
        raise NotImplementedError

    def _get_args(self, X: Dict[str, Any]) -> Union[int, np.ndarray]:
        # Feature preprocessors can alter numerical columns
        if len(X['dataset_properties']['numerical_columns']) == 0:
            num_numerical_columns = 0
        else:
            X_train = copy.deepcopy(X['backend'].load_datamanager().train_tensors[0][:2])
            # as numerical pipeline will always be the first pipeline
            numerical_column_transformer = X['tabular_transformer'].preprocessor.named_transformers_['numerical_pipeline']
            num_numerical_columns = numerical_column_transformer.transform(
                X_train.iloc[:, X['dataset_properties']['numerical_columns']]).shape[1]
        num_input_features = np.zeros((num_numerical_columns +
                                       len(X['dataset_properties']['categorical_columns'])), dtype=int)
        categories = X['dataset_properties']['categories']

        for i, category in enumerate(categories):
            num_input_features[num_numerical_columns + i, ] = len(category)
        return num_numerical_columns, num_input_features
