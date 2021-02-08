from typing import Optional, Any

from sklearn.base import BaseEstimator
from torch import nn

from autoPyTorch.pipeline.components.setup.base_setup import autoPyTorchSetupComponent


class NetworkEmbeddingComponent(autoPyTorchSetupComponent):
    def __init__(self,
                 **kwargs):
        super().__init__()
        self.config = kwargs
        self.embedding: Optional[nn.Module] = None

    # def fit(self, X: Dict[str, Any], y: Any = None) -> BaseEstimator: