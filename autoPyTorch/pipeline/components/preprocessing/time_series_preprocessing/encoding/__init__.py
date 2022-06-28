import os
from collections import OrderedDict
from typing import Dict

from autoPyTorch.pipeline.components.base_component import (
    ThirdPartyComponents, autoPyTorchComponent, find_components)
from autoPyTorch.pipeline.components.preprocessing.tabular_preprocessing.encoding import \
    EncoderChoice
from autoPyTorch.pipeline.components.preprocessing.time_series_preprocessing.encoding.time_series_base_encoder import \
    TimeSeriesBaseEncoder

encoding_directory = os.path.split(__file__)[0]
_encoders = find_components(__package__,
                            encoding_directory,
                            TimeSeriesBaseEncoder)
_addons = ThirdPartyComponents(TimeSeriesBaseEncoder)


def add_encoder(encoder: TimeSeriesBaseEncoder) -> None:
    _addons.add_component(encoder)


class TimeSeriesEncoderChoice(EncoderChoice):
    """
    Allows for dynamically choosing encoding component at runtime
    """

    def get_components(self) -> Dict[str, autoPyTorchComponent]:
        """Returns the available encoder components

        Args:
            None

        Returns:
            Dict[str, autoPyTorchComponent]: all BaseEncoder components available
                as choices for encoding the categorical columns
        """
        components = OrderedDict()
        components.update(_encoders)
        components.update(_addons.components)
        return components
