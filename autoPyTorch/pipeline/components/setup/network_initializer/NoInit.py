from typing import Callable

import torch

from autoPyTorch.pipeline.components.setup.network_initializer.base_network_initializer import (
    BaseNetworkInitializerComponent
)


class NoInit(BaseNetworkInitializerComponent):
    """
    No initialization on the weights/bias
    """

    def weights_init(self) -> Callable:
        """Returns the actual PyTorch model, that is dynamically created
        from a self.config object.

        self.config is a dictionary created form a given config in the config space.
        It contains the necessary information to build a network.
        """
        def initialization(m: torch.nn.Module) -> None:
            pass
        return initialization
