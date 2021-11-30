import os

from autoPyTorch.pipeline.components.setup.network_head.distributed_network_head.distributed_network_head import (
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
