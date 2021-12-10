from abc import abstractmethod
from autoPyTorch.pipeline.components.setup.network_backbone.base_network_backbone import NetworkBackboneComponent

from abc import abstractmethod
from typing import Any, Dict, Iterable, Optional, Tuple, List


from autoPyTorch.pipeline.components.base_component import BaseEstimator


class BaseForecastingNetworkBackbone(NetworkBackboneComponent):
    """
    Base forecasting network, its output needs to be a 3-d Tensor:
    """
    def fit(self, X: Dict[str, Any], y: Any = None) -> BaseEstimator:
        return super().fit(X, y)

    def transform(self, X: Dict[str, Any]) -> Dict[str, Any]:
        X = super().transform(X)
        X.update({'encoder_properties': self.encoder_properties})
        return X

    @property
    @abstractmethod
    def encoder_properties(self):
        raise NotImplementedError
