import os
from collections import OrderedDict
from typing import Dict, List, Optional

import ConfigSpace.hyperparameters as CSH
from ConfigSpace.configuration_space import ConfigurationSpace

import numpy as np

from autoPyTorch.datasets.base_dataset import BaseDatasetPropertiesType
from autoPyTorch.pipeline.components.base_choice import autoPyTorchChoice
from autoPyTorch.pipeline.components.base_component import (
    ThirdPartyComponents,
    autoPyTorchComponent,
    find_components,
)
from autoPyTorch.pipeline.components.setup.network_backbone import NetworkBackboneChoice
from autoPyTorch.pipeline.components.setup.network_backbone.forecasting_network_backbone.base_forecasting_backbone\
    import (
    BaseForecastingNetworkBackbone,
)

directory = os.path.split(__file__)[0]
_backbones = find_components(__package__,
                             directory,
                             BaseForecastingNetworkBackbone)
_addons = ThirdPartyComponents(BaseForecastingNetworkBackbone)


def add_backbone(backbone: BaseForecastingNetworkBackbone) -> None:
    _addons.add_component(backbone)


class ForecastingNetworkBackboneChoice(NetworkBackboneChoice):
    def get_components(self) -> Dict[str, autoPyTorchComponent]:
        """Returns the available backbone components

        Args:
            None

        Returns:
            Dict[str, autoPyTorchComponent]: all basebackbone components available
                as choices for learning rate scheduling
        """
        components = OrderedDict()
        components.update(_backbones)
        components.update(_addons.components)
        return components

    @property
    def _defaults_network(self):
        return ['RNNBackbone', 'TSMLPBackbone']