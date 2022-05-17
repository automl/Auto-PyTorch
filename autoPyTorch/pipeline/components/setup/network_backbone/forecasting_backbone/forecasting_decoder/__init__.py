import os
from collections import OrderedDict
from typing import Dict, List, Optional

import ConfigSpace.hyperparameters as CSH
from ConfigSpace.configuration_space import ConfigurationSpace

from autoPyTorch.datasets.base_dataset import BaseDatasetPropertiesType
from autoPyTorch.pipeline.components.base_component import (
    autoPyTorchComponent,
)

from autoPyTorch.pipeline.components.setup.network_backbone import NetworkBackboneChoice
from autoPyTorch.pipeline.components.setup.network_backbone.\
    forecasting_backbone.forecasting_decoder.base_forecasting_decoder import BaseForecastingDecoder


from autoPyTorch.pipeline.components.base_component import (
    ThirdPartyComponents,
    find_components,
)

directory = os.path.split(__file__)[0]
decoders = find_components(__package__,
                         directory,
                         BaseForecastingDecoder)

decoder_addons = ThirdPartyComponents(BaseForecastingDecoder)


def add_decoder(encoder: BaseForecastingDecoder) -> None:
    decoder_addons.add_component(encoder)


class ForecastingDecoderChoice(NetworkBackboneChoice):
    def get_components(self) -> Dict[str, autoPyTorchComponent]:
        """Returns the available head components

        Args:
            None

        Returns:
            Dict[str, autoPyTorchComponent]: all NetworkHeadComponents available
                as choices for learning rate scheduling
        """
        components = OrderedDict()

        components.update(decoders)
        components.update(decoder_addons.components)

        return components


