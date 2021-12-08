from abc import abstractmethod
from autoPyTorch.pipeline.components.setup.network_backbone.base_network_backbone import NetworkBackboneComponent

from abc import abstractmethod
from typing import Any, Dict, Iterable, Optional, Tuple, List



from autoPyTorch.pipeline.components.base_component import BaseEstimator


class BaseForecastingNetworkBackbone(NetworkBackboneComponent):
    def fit(self, X: Dict[str, Any], y: Any = None) -> BaseEstimator:
        X['backbone_properties'] = self.backbone_properties
        return super().fit(X, y)

    @property
    @abstractmethod
    def backbone_properties(self):
        raise NotImplementedError
