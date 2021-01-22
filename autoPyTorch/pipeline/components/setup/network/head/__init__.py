from collections import OrderedDict
from typing import Dict, Type

from autoPyTorch.pipeline.components.base_component import (
    ThirdPartyComponents
)
from autoPyTorch.pipeline.components.setup.network.head.base_head import BaseHead
from autoPyTorch.pipeline.components.setup.network.head.fully_connected import FullyConnectedHead
from autoPyTorch.pipeline.components.setup.network.head.fully_convolutional import FullyConvolutional2DHead

_heads = {
    FullyConnectedHead.get_name(): FullyConnectedHead,
    FullyConvolutional2DHead.get_name(): FullyConvolutional2DHead
}
_addons = ThirdPartyComponents(BaseHead)


def add_head(head: BaseHead) -> None:
    _addons.add_component(head)


def get_available_heads() -> Dict[str, Type[BaseHead]]:
    heads = OrderedDict()
    heads.update(_heads)
    heads.update(_addons.components)
    return heads
