import os

from autoPyTorch.pipeline.components.base_component import (
    ThirdPartyComponents, find_components)
from autoPyTorch.pipeline.components.setup.network_backbone.forecasting_backbone.forecasting_decoder.\
    base_forecasting_decoder import BaseForecastingDecoder

directory = os.path.split(__file__)[0]
decoders = find_components(__package__,
                           directory,
                           BaseForecastingDecoder)

decoder_addons = ThirdPartyComponents(BaseForecastingDecoder)


def add_decoder(decoder: BaseForecastingDecoder) -> None:
    decoder_addons.add_component(decoder)
