import os
from collections import OrderedDict
from typing import Dict, Optional, Type, Union

from autoPyTorch.datasets.base_dataset import BaseDatasetPropertiesType
from autoPyTorch.pipeline.components.base_component import (
    ThirdPartyComponents,
    autoPyTorchComponent,
    find_components
)
from autoPyTorch.pipeline.components.setup.network_backbone.forecasting_backbone.forecasting_encoder import \
    AbstractForecastingEncoderChoice
from autoPyTorch.pipeline.components.setup.network_backbone.forecasting_backbone.forecasting_encoder.\
    base_forecasting_encoder import BaseForecastingEncoder

directory = os.path.split(__file__)[0]
_encoders = find_components(__package__,
                            directory,
                            BaseForecastingEncoder)
_addons = ThirdPartyComponents(BaseForecastingEncoder)


def add_encoder(encoder: BaseForecastingEncoder) -> None:
    _addons.add_component(encoder)


class FlatForecastingEncoderChoice(AbstractForecastingEncoderChoice):
    def get_components(self) -> Dict[str, Type[autoPyTorchComponent]]:  # type: ignore[override]
        """Returns the available backbone components

        Args:
            None

        Returns:
            Dict[str, autoPyTorchComponent]: all basebackbone components available
                as choices for learning rate scheduling
        """
        components = OrderedDict()
        components.update(_encoders)
        components.update(_addons.components)
        return components

    @staticmethod
    def get_properties(dataset_properties: Optional[Dict[str, BaseDatasetPropertiesType]] = None
                       ) -> Dict[str, Union[str, bool]]:
        return {
            'shortname': 'FlatEncoder',
            'name': 'FlatEncoder',
            'handles_tabular': False,
            'handles_image': False,
            'handles_time_series': True,
        }
