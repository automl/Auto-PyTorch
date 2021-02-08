from typing import Any, Dict, Optional

from sklearn.base import BaseEstimator
from torch import nn

from autoPyTorch.pipeline.components.setup.base_setup import autoPyTorchSetupComponent


class NetworkEmbeddingComponent(autoPyTorchSetupComponent):
    def __init__(self,
                 **kwargs):
        super().__init__()
        self.config = kwargs
        self.embedding: Optional[nn.Module] = None

    def fit(self, X: Dict[str, Any], y: Any = None) -> BaseEstimator:
        in_features = X['X_train'].shape[1:]

        self.embedding = self.build_embedding(
            in_features=in_features,
            num_numerical_features=len(X['numerical_features']))
        return self

    def build_embedding(self, in_features, num_numerical_features) -> nn.Module:
        raise NotImplementedError