import os
from collections import OrderedDict
from typing import Dict, List, Optional

import ConfigSpace.hyperparameters as CSH
from ConfigSpace.configuration_space import ConfigurationSpace

from autoPyTorch.datasets.base_dataset import BaseDatasetPropertiesType
from autoPyTorch.pipeline.components.base_choice import autoPyTorchChoice
from autoPyTorch.pipeline.components.base_component import (
    ThirdPartyComponents,
    autoPyTorchComponent,
    find_components,
)
from autoPyTorch.pipeline.components.preprocessing.image_preprocessing.normalise.base_normalizer import BaseNormalizer


normalise_directory = os.path.split(__file__)[0]
_normalizers = find_components(__package__,
                               normalise_directory,
                               BaseNormalizer)

_addons = ThirdPartyComponents(BaseNormalizer)


def add_normalizer(normalizer: BaseNormalizer) -> None:
    _addons.add_component(normalizer)


class NormalizerChoice(autoPyTorchChoice):
    """
    Allows for dynamically choosing normalizer component at runtime
    """

    def get_components(self) -> Dict[str, autoPyTorchComponent]:
        """Returns the available normalizer components

        Args:
            None

        Returns:
            Dict[str, autoPyTorchComponent]: all BaseNormalizer components available
                as choices for encoding the categorical columns
        """
        components = OrderedDict()
        components.update(_normalizers)
        components.update(_addons.components)
        return components

    def get_hyperparameter_search_space(self,
                                        dataset_properties: Optional[Dict[str, BaseDatasetPropertiesType]] = None,
                                        default: Optional[str] = None,
                                        include: Optional[List[str]] = None,
                                        exclude: Optional[List[str]] = None) -> ConfigurationSpace:
        cs = ConfigurationSpace()

        if dataset_properties is None:
            dataset_properties = dict()

        dataset_properties = {**self.dataset_properties, **dataset_properties}

        available_preprocessors = self.get_available_components(dataset_properties=dataset_properties,
                                                                include=include,
                                                                exclude=exclude)

        if len(available_preprocessors) == 0:
            raise ValueError("no image normalizers found, please add an image normalizer")

        if default is None:
            defaults = ['ImageNormalizer', 'NoNormalizer']
            for default_ in defaults:
                if default_ in available_preprocessors:
                    if include is not None and default_ not in include:
                        continue
                    if exclude is not None and default_ in exclude:
                        continue
                    default = default_
                    break

        updates = self._get_search_space_updates()
        if '__choice__' in updates.keys():
            choice_hyperparameter = updates['__choice__']
            if not set(choice_hyperparameter.value_range).issubset(available_preprocessors):
                raise ValueError("Expected given update for {} to have "
                                 "choices in {} got {}".format(self.__class__.__name__,
                                                               available_preprocessors,
                                                               choice_hyperparameter.value_range))
            preprocessor = CSH.CategoricalHyperparameter('__choice__',
                                                         choice_hyperparameter.value_range,
                                                         default_value=choice_hyperparameter.default_value)
        else:
            preprocessor = CSH.CategoricalHyperparameter('__choice__',
                                                         list(available_preprocessors.keys()),
                                                         default_value=default)
        cs.add_hyperparameter(preprocessor)

        # add only child hyperparameters of preprocessor choices
        for name in preprocessor.choices:
            preprocessor_configuration_space = available_preprocessors[name].\
                get_hyperparameter_search_space(dataset_properties)
            parent_hyperparameter = {'parent': preprocessor, 'value': name}
            cs.add_configuration_space(name, preprocessor_configuration_space,
                                       parent_hyperparameter=parent_hyperparameter)

        self.configuration_space = cs
        self.dataset_properties = dataset_properties
        return cs
