from autoPyTorch.pipeline.components.setup.network_backbone.forecasting_backbone.forecasting_encoder import \
    AbstractForecastingEncoderChoice


import os
from collections import OrderedDict
from typing import Dict, Union, Optional, List, Type

from autoPyTorch.pipeline.components.base_component import (
    ThirdPartyComponents,
    autoPyTorchComponent,
    find_components,
)
from autoPyTorch.pipeline.components.setup.network_backbone import NetworkBackboneChoice
from autoPyTorch.datasets.base_dataset import BaseDatasetPropertiesType
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
    def get_available_components(
        self,
        dataset_properties: Optional[Dict[str, BaseDatasetPropertiesType]] = None,
        include: List[str] = None,
        exclude: List[str] = None,
        components: Optional[Dict[str, autoPyTorchComponent]] = None
    ) -> Dict[str, Type[autoPyTorchComponent]]:
        """Filters out components based on user provided
        include/exclude directives, as well as the dataset properties

        Args:
         include (Optional[Dict[str, Any]]): what hyper-parameter configurations
            to honor when creating the configuration space
         exclude (Optional[Dict[str, Any]]): what hyper-parameter configurations
             to remove from the configuration space
         dataset_properties (Optional[Dict[str, Union[str, int]]]): Caracteristics
             of the dataset to guide the pipeline choices of components

        Returns:
            Dict[str, autoPyTorchComponent]: A filtered dict of learning
                rate backbones

        """
        if dataset_properties is None:
            dataset_properties = {}

        if include is not None and exclude is not None:
            raise ValueError(
                "The argument include and exclude cannot be used together.")

        if components is None:
            available_comp = self.get_components()
        else:
            available_comp = components

        if include is not None:
            for incl in include:
                if incl not in available_comp:
                    raise ValueError("Trying to include unknown component: "
                                     "%s" % incl)

        components_dict = OrderedDict()
        for name in available_comp:
            if include is not None and name not in include:
                continue
            elif exclude is not None and name in exclude:
                continue

            entry = available_comp[name]

            # Exclude itself to avoid infinite loop
            if entry == NetworkBackboneChoice or hasattr(entry, 'get_components'):
                continue

            task_type = str(dataset_properties['task_type'])
            properties = entry.get_properties()
            if 'tabular' in task_type and not bool(properties['handles_tabular']):
                continue
            elif 'image' in task_type and not bool(properties['handles_image']):
                continue
            elif 'time_series' in task_type and not bool(properties['handles_time_series']):
                continue

            # target_type = dataset_properties['target_type']
            # Apply some automatic filtering here for
            # backbones based on the dataset!
            # TODO: Think if there is any case where a backbone
            # is not recommended for a certain dataset

            components_dict[name] = entry

        return components_dict


    def get_components(self) -> Dict[str, autoPyTorchComponent]:
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
            'shortname': 'SeqEncoder',
            'name': 'SeqEncoder',
            'handles_tabular': False,
            'handles_image': False,
            'handles_time_series': True,
        }

