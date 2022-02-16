from typing import Callable

from torch import nn

from autoPyTorch.pipeline.components.setup.network_initializer.base_network_initializer import (
    BaseNetworkInitializerComponent
)


class KaimingInit(BaseNetworkInitializerComponent):
    """
    Fills the input Tensor with values according to the method described in
    Delving deep into rectifiers: Surpassing human-level performance on
    ImageNet classification
    """

    def weights_init(self) -> Callable:
        """Returns the actual PyTorch model, that is dynamically created
        from a self.config object.

        self.config is a dictionary created form a given config in the config space.
        It contains the necessary information to build a network.
        """
        def initialization(m: nn.Module) -> None:
            if isinstance(m, (nn.Conv1d,
                              nn.Conv2d,
                              nn.Conv3d,
                              nn.Linear)):
                nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None and self.bias_strategy == 'Zero':
                    nn.init.constant_(m.bias.data, 0.0)
        return initialization
