from typing import Any, Dict, List, Optional, Union


from typing import Tuple
from autoPyTorch.pipeline.components.setup.network_backbone.MLPBackbone import MLPBackbone

import torch
from torch import nn


from autoPyTorch.pipeline.components.base_component import BaseEstimator
from autoPyTorch.datasets.base_dataset import BaseDatasetPropertiesType
from autoPyTorch.utils.common import FitRequirement


class _TimeSeriesMLP(nn.Module):
    def __init__(self,
                 module_layers: nn.Module,
                 ):
        super().__init__()
        self.module_layers = module_layers

    def forward(self, x: torch.Tensor):
        x = x.view(x.shape[0], -1)
        return self.module_layers(x)


class TimeSeriesMLPBackbone(MLPBackbone):
    _fixed_seq_length = True
    window_size = 1

    @property
    def _required_fit_arguments(self) -> List[FitRequirement]:
        requirements_list = super()._required_fit_arguments
        requirements_list.append(FitRequirement('window_size', (int,), user_defined=False, dataset_property=False))
        return requirements_list

    def fit(self, X: Dict[str, Any], y: Any = None) -> BaseEstimator:
        self.window_size = X["window_size"]
        X['MLP_backbone'] = True
        return super().fit(X, y)

    def build_backbone(self, input_shape: Tuple[int, ...]) -> nn.Module:
        in_features = input_shape[-1] * self.window_size
        return _TimeSeriesMLP(self._build_backbone(in_features))

    @staticmethod
    def get_properties(dataset_properties: Optional[Dict[str, BaseDatasetPropertiesType]] = None
                       ) -> Dict[str, Union[str, bool]]:
        return {
            'shortname': 'TSMLPBackbone',
            'name': 'TimeSeriesMLPBackbone',
            'handles_tabular': False,
            'handles_image': False,
            'handles_time_series': True,
        }
