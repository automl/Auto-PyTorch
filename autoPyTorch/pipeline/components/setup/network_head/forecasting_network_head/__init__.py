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
from autoPyTorch.pipeline.components.setup.network_head.base_network_head import (
    NetworkHeadComponent,
)

from autoPyTorch.pipeline.components.setup.network_head import NetworkHeadChoice
from autoPyTorch.pipeline.components.setup.network_head.forecasting_network_head.distributed_network_head import (
    DistributionNetworkHeadComponents,
)

from autoPyTorch.pipeline.components.base_component import (
    ThirdPartyComponents,
    find_components,
)

directory = os.path.split(__file__)[0]
_distributed_heads = find_components(__package__,
                                     directory,
                                     DistributionNetworkHeadComponents)

_distributed_addons = ThirdPartyComponents(DistributionNetworkHeadComponents)


class ForecastingNetworkHeadChoice(NetworkHeadChoice):
    def get_components(self) -> Dict[str, autoPyTorchComponent]:
        """Returns the available head components

        Args:
            None

        Returns:
            Dict[str, autoPyTorchComponent]: all NetworkHeadComponents available
                as choices for learning rate scheduling
        """
        components = OrderedDict()
        components.update(_distributed_heads)
        components.update(_distributed_addons.components)

        return components
