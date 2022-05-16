import unittest

from autoPyTorch.pipeline.components.setup.network_backbone.forecasting_backbone.forecasting_encoder.flat_encoder \
    import FlatForecastingEncoderChoice
from autoPyTorch.pipeline.components.setup.network_backbone.forecasting_backbone.forecasting_encoder.flat_encoder.\
    MLPEncoder import MLPEncoder
from autoPyTorch.pipeline.components.setup.network_backbone.forecasting_backbone.forecasting_encoder.flat_encoder.\
    NBEATSEncoder import NBEATSEncoder
from autoPyTorch.pipeline.components.setup.network_backbone.forecasting_backbone.forecasting_decoder.MLPDecoder import (
    ForecastingMLPDecoder
)
from autoPyTorch.pipeline.components.setup.network_backbone.forecasting_backbone.forecasting_decoder.NBEATSDecoder \
    import NBEATSDecoder


